// Fase 2: contagem de suporte e pré-processamento na GPU.

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "gpu/gpu.cuh"

namespace fpg {
namespace gpu {

namespace {

constexpr int kThreads = 256;
// Itens por faixa no histograma em memória compartilhada (48 KB = 12 K contadores).
constexpr int kSharedBins = 12 * 1024;

// 2.2 Contagem v1: um atomicAdd global por ocorrência de item.
__global__ void count_v1_kernel(const int32_t* __restrict__ items, int64_t nnz, int32_t* __restrict__ support) {
    for (int64_t p = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; p < nnz; p += int64_t(gridDim.x) * blockDim.x)
        atomicAdd(&support[items[p]], 1);
}

// 2.3 Contagem v2: histograma privado por bloco em memória compartilhada.
// Quando os ids não cabem, gridDim.y particiona os ids em faixas de kSharedBins;
// o bloco (x, y) conta só os itens da faixa y.
__global__ void count_v2_kernel(const int32_t* __restrict__ items, int64_t nnz, int32_t n_ids,
                                int32_t* __restrict__ support) {
    __shared__ int32_t hist[kSharedBins];
    const int32_t lo = blockIdx.y * kSharedBins;
    const int32_t bins = min(kSharedBins, n_ids - lo);
    for (int i = threadIdx.x; i < bins; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    for (int64_t p = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; p < nnz; p += int64_t(gridDim.x) * blockDim.x) {
        int32_t x = items[p] - lo;
        if (x >= 0 && x < bins) atomicAdd(&hist[x], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < bins; i += blockDim.x)
        if (hist[i]) atomicAdd(&support[lo + i], hist[i]);
}

// 2.5 Filtragem e remapeamento: uma thread por item.
__global__ void filter_remap_kernel(const int32_t* __restrict__ items, int64_t nnz, const int32_t* __restrict__ id_to_rank,
                                    int32_t* __restrict__ mapped, int32_t* __restrict__ flag) {
    int64_t p = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (p >= nnz) return;
    int32_t r = id_to_rank[items[p]];
    mapped[p] = r;
    flag[p] = r >= 0;
}

__global__ void scatter_flagged_kernel(const int32_t* __restrict__ mapped, const int32_t* __restrict__ flag,
                                       const int32_t* __restrict__ pos, int64_t nnz, int32_t* __restrict__ out) {
    int64_t p = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (p < nnz && flag[p]) out[pos[p]] = mapped[p];
}

struct SupportRankLess {
    const int32_t* sup;
    __device__ bool operator()(int32_t a, int32_t b) const {
        return sup[a] != sup[b] ? sup[a] > sup[b] : a < b;
    }
};

struct IsPositive {
    __device__ bool operator()(int32_t x) const { return x > 0; }
};

int grid_for(int64_t n) { return static_cast<int>(std::min<int64_t>(div_up(n, kThreads), 1 << 16)); }

}  // namespace

DeviceCsr upload(const Csr& db, StageTimes& times) {
    GpuTimer t;
    DeviceCsr d;
    d.items.resize(db.items.size());
    d.offsets.resize(db.offsets.size());
    CUDA_CHECK(cudaMemcpy(raw(d.items), db.items.data(), db.items.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(raw(d.offsets), db.offsets.data(), db.offsets.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    times.add("transferencia_h2d", t.ms());
    return d;
}

thrust::device_vector<int32_t> count_support(const DeviceCsr& db, int32_t max_id, CountKernel kernel) {
    const int32_t n_ids = max_id + 1;
    const int64_t nnz = static_cast<int64_t>(db.items.size());
    thrust::device_vector<int32_t> support(n_ids, 0);
    if (nnz == 0) return support;
    if (kernel == CountKernel::V1) {
        count_v1_kernel<<<grid_for(nnz), kThreads>>>(raw(db.items), nnz, raw(support));
    } else {
        // Poucos blocos por faixa: cada bloco varre muitos itens e amortiza o
        // custo de zerar e descarregar o histograma.
        int sms = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        int bx = static_cast<int>(std::min<int64_t>(div_up(nnz, kThreads * 16), sms * 4));
        dim3 grid(std::max(bx, 1), div_up(n_ids, kSharedBins));
        count_v2_kernel<<<grid, kThreads>>>(raw(db.items), nnz, n_ids, raw(support));
    }
    CUDA_CHECK_KERNEL();
    return support;
}

Preprocessed preprocess(const DeviceCsr& db, int32_t max_id, int32_t minsup, CountKernel kernel, StageTimes& times) {
    Preprocessed out;
    const int32_t n_ids = max_id + 1;
    const int32_t n_tx = db.n();
    const int64_t nnz = static_cast<int64_t>(db.items.size());

    GpuTimer t;
    thrust::device_vector<int32_t> support = count_support(db, max_id, kernel);
    times.add("contagem", t.ms());

    t.start();
    // 2.4 Tabela de ranks: ids frequentes ordenados por (suporte desc, id asc).
    thrust::device_vector<int32_t> ids(n_ids);
    auto ids_end = thrust::copy_if(thrust::device, thrust::counting_iterator<int32_t>(0),
                                   thrust::counting_iterator<int32_t>(n_ids), support.begin(), ids.begin(),
                                   [minsup] __device__(int32_t s) { return s >= minsup; });
    ids.resize(ids_end - ids.begin());
    out.n_frequent = static_cast<int32_t>(ids.size());
    thrust::sort(thrust::device, ids.begin(), ids.end(), SupportRankLess{raw(support)});
    thrust::device_vector<int32_t> id_to_rank(n_ids, -1);
    thrust::scatter(thrust::device, thrust::counting_iterator<int32_t>(0),
                    thrust::counting_iterator<int32_t>(out.n_frequent), ids.begin(), id_to_rank.begin());
    out.rank_support.resize(out.n_frequent);
    thrust::gather(thrust::device, ids.begin(), ids.end(), support.begin(), out.rank_support.begin());

    // 2.5 Filtragem e remapeamento.
    thrust::device_vector<int32_t> mapped(nnz), flag(nnz), pos(nnz);
    if (nnz > 0) {
        filter_remap_kernel<<<div_up(nnz, kThreads), kThreads>>>(raw(db.items), nnz, raw(id_to_rank), raw(mapped),
                                                                  raw(flag));
        CUDA_CHECK_KERNEL();
    }

    // 2.6 Compactação: posição de cada item mantido e novo tamanho de cada transação.
    thrust::exclusive_scan(thrust::device, flag.begin(), flag.end(), pos.begin());
    thrust::device_vector<int32_t> new_len(n_tx);
    {
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, raw(flag), raw(new_len), n_tx, raw(db.offsets),
                                                   raw(db.offsets) + 1));
        thrust::device_vector<char> tmp(bytes);
        CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(raw(tmp), bytes, raw(flag), raw(new_len), n_tx, raw(db.offsets),
                                                   raw(db.offsets) + 1));
    }
    const int32_t kept = nnz ? pos.back() + flag.back() : 0;
    thrust::device_vector<int32_t> filtered(kept);
    if (nnz > 0) {
        scatter_flagged_kernel<<<div_up(nnz, kThreads), kThreads>>>(raw(mapped), raw(flag), raw(pos), nnz,
                                                                     raw(filtered));
        CUDA_CHECK_KERNEL();
    }
    // Descarta transações vazias: offsets = scan dos tamanhos não nulos.
    thrust::device_vector<int32_t> lens(n_tx);
    auto lens_end = thrust::copy_if(thrust::device, new_len.begin(), new_len.end(), lens.begin(), IsPositive{});
    const int32_t n_out = static_cast<int32_t>(lens_end - lens.begin());
    out.db.offsets.resize(n_out + 1);
    out.db.offsets[0] = 0;
    thrust::inclusive_scan(thrust::device, lens.begin(), lens_end, out.db.offsets.begin() + 1);

    // 2.7 Ordena cada transação por rank.
    out.db.items.resize(kept);
    if (kept > 0) {
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(nullptr, bytes, raw(filtered), raw(out.db.items), kept, n_out,
                                                      raw(out.db.offsets), raw(out.db.offsets) + 1));
        thrust::device_vector<char> tmp(bytes);
        CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(raw(tmp), bytes, raw(filtered), raw(out.db.items), kept, n_out,
                                                      raw(out.db.offsets), raw(out.db.offsets) + 1));
    }
    times.add("preprocessamento", t.ms());

    out.rank_to_id.resize(out.n_frequent);
    thrust::copy(ids.begin(), ids.end(), out.rank_to_id.begin());
    out.h_support.resize(n_ids);
    thrust::copy(support.begin(), support.end(), out.h_support.begin());
    return out;
}

Csr download(const DeviceCsr& db) {
    Csr c;
    c.items.resize(db.items.size());
    c.offsets.resize(db.offsets.size());
    thrust::copy(db.items.begin(), db.items.end(), c.items.begin());
    thrust::copy(db.offsets.begin(), db.offsets.end(), c.offsets.begin());
    return c;
}

}  // namespace gpu
}  // namespace fpg
