// Fase 3: construção da FP-tree por primitivas paralelas (sort, LCP, scan,
// sparse table), sem inserções concorrentes.

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "gpu/gpu.cuh"
#include "gpu/tree_core.cuh"

namespace fpg {
namespace gpu {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int64_t grid_tid() { return blockIdx.x * int64_t(blockDim.x) + threadIdx.x; }
__device__ __forceinline__ int64_t grid_threads() { return int64_t(gridDim.x) * blockDim.x; }

int grid_for(int64_t n) { return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(div_up(n, kThreads), 1 << 20))); }

struct PathLess {
    const int32_t* items;
    const int32_t* off;
    __device__ bool operator()(int32_t a, int32_t b) const { return path_less(items, off, a, b); }
};

__global__ void gather_lens_kernel(const int32_t* order, const int32_t* off, int32_t n, int32_t* lens) {
    for (int64_t i = grid_tid(); i < n; i += grid_threads()) lens[i] = off[order[i] + 1] - off[order[i]];
}

__global__ void permute_kernel(const int32_t* items, const int32_t* off, const int32_t* order, const int32_t* new_off,
                               int32_t n, int32_t* new_items) {
    for (int64_t i = grid_tid(); i < n; i += grid_threads()) {
        int32_t src = off[order[i]], len = new_off[i + 1] - new_off[i];
        for (int32_t d = 0; d < len; ++d) new_items[new_off[i] + d] = items[src + d];
    }
}

// 3.2: LCP e número de nós novos por caminho.
__global__ void lcp_kernel(const int32_t* items, const int32_t* off, int32_t n, int32_t* lcp, int32_t* new_nodes) {
    for (int64_t i = grid_tid(); i < n; i += grid_threads()) {
        int32_t l = path_lcp(items, off, int32_t(i));
        lcp[i] = l;
        new_nodes[i] = off[i + 1] - off[i] - l;
    }
}

__global__ void sparse_level_kernel(int32_t* st, int32_t n, int j) { sparse_level(st, n, j, grid_tid(), grid_threads()); }

__global__ void write_nodes_kernel(const int32_t* items, const int32_t* off, const int32_t* lcp, const int32_t* base,
                                   int32_t n, int32_t* node_item, int32_t* node_depth, int32_t* node_tx) {
    write_nodes(items, off, lcp, base, n, node_item, node_depth, node_tx, grid_tid(), grid_threads());
}

__global__ void parents_kernel(const int32_t* st, int levels, const int32_t* lcp, const int32_t* base, int32_t n,
                               int32_t n_nodes, const int32_t* node_depth, const int32_t* node_tx,
                               int32_t* node_parent) {
    node_parents(st, levels, lcp, base, n, n_nodes, node_depth, node_tx, node_parent, grid_tid(), grid_threads());
}

__global__ void counts_kernel(const int32_t* st, int levels, const int32_t* W, int32_t n, int32_t n_nodes,
                              const int32_t* node_depth, const int32_t* node_tx, int32_t* node_count) {
    node_counts(st, levels, W, n, n_nodes, node_depth, node_tx, node_count, grid_tid(), grid_threads());
}

// 3.7: id do nó percorrido por cada posição (i, d), com o peso do caminho i.
__global__ void traversal_kernel(const int32_t* st, int levels, const int32_t* off, const int32_t* lcp,
                                 const int32_t* base, const int32_t* weights, int32_t n, int32_t* key, int32_t* val) {
    for (int64_t i = grid_tid(); i < n; i += grid_threads()) {
        int32_t len = off[i + 1] - off[i], l = lcp[i];
        int32_t w = weights ? weights[i] : 1;
        for (int32_t d = 0; d < len; ++d) {
            int32_t k = d >= l ? int32_t(i) : prev_smaller_eq(st, n, levels, int32_t(i), d);
            key[off[i] + d] = base[k] + d - lcp[k];
            val[off[i] + d] = w;
        }
    }
}

// Amostra a memória livre para estimar o pico de uso.
struct MemProbe {
    size_t start_free = 0, min_free = SIZE_MAX;
    MemProbe() { sample(); start_free = min_free; }
    void sample() {
        size_t f = 0, t = 0;
        CUDA_CHECK(cudaMemGetInfo(&f, &t));
        min_free = std::min(min_free, f);
        MemTracker::sample();
    }
    size_t peak() const { return start_free - min_free; }
};

}  // namespace

DeviceTree build_tree(DeviceCsr& db, const thrust::device_vector<int32_t>& weights, int32_t n_items,
                      const BuildOptions& opt, StageTimes& times, BuildStats* stats) {
    DeviceTree tree;
    tree.n_items = n_items;
    const int32_t n = db.n();
    const bool weighted = !weights.empty();
    MemProbe mem;
    double total = 0;
    auto stage = [&](const char* name, GpuTimer& t) {
        double ms = t.ms();
        times.add(name, ms);
        total += ms;
        mem.sample();
        t.start();
    };
    if (n == 0) {
        tree.header_offsets.assign(n_items + 1, 0);
        tree.item_support.assign(n_items, 0);
        times.add("construcao", 0.0);
        return tree;
    }
    GpuTimer t;

    // 3.1 Ordenação lexicográfica dos caminhos e reescrita do CSR.
    thrust::device_vector<int32_t> order(n);
    thrust::sequence(thrust::device, order.begin(), order.end());
    thrust::sort(thrust::device, order.begin(), order.end(), PathLess{raw(db.items), raw(db.offsets)});
    thrust::device_vector<int32_t> new_off(n + 1);
    {
        thrust::device_vector<int32_t> lens(n);
        gather_lens_kernel<<<grid_for(n), kThreads>>>(raw(order), raw(db.offsets), n, raw(lens));
        CUDA_CHECK_KERNEL();
        new_off[0] = 0;
        thrust::inclusive_scan(thrust::device, lens.begin(), lens.end(), new_off.begin() + 1);
    }
    thrust::device_vector<int32_t> new_items(db.items.size());
    permute_kernel<<<grid_for(n), kThreads>>>(raw(db.items), raw(db.offsets), raw(order), raw(new_off), n,
                                              raw(new_items));
    CUDA_CHECK_KERNEL();
    thrust::device_vector<int32_t> w_sorted;
    if (weighted) {
        w_sorted.resize(n);
        thrust::gather(thrust::device, order.begin(), order.end(), weights.begin(), w_sorted.begin());
    }
    mem.sample();
    db.items.swap(new_items);
    db.offsets.swap(new_off);
    new_items.clear();
    new_items.shrink_to_fit();
    new_off.clear();
    new_off.shrink_to_fit();
    order.clear();
    order.shrink_to_fit();
    const int32_t* items = raw(db.items);
    const int32_t* off = raw(db.offsets);
    stage("construcao_sort", t);

    // 3.2 LCP.
    thrust::device_vector<int32_t> lcp(n), base(n + 1);
    lcp_kernel<<<grid_for(n), kThreads>>>(items, off, n, raw(lcp), raw(base));
    CUDA_CHECK_KERNEL();
    stage("construcao_lcp", t);

    // 3.3 Nós novos: node_base = scan exclusivo; base[n] = N. Pesos: W = scan exclusivo.
    thrust::exclusive_scan(thrust::device, base.begin(), base.end(), base.begin());
    const int32_t N = base[n];
    thrust::device_vector<int32_t> W;
    if (weighted) {
        W.resize(n + 1);
        W[n] = 0;  // valor neutro na última posição antes do scan
        thrust::copy(w_sorted.begin(), w_sorted.end(), W.begin());
        thrust::exclusive_scan(thrust::device, W.begin(), W.end(), W.begin());
    }
    tree.n_nodes = N;
    stage("construcao_scan", t);

    // 3.4 Item, profundidade e caminho criador de cada nó.
    tree.item.resize(N);
    tree.depth.resize(N);
    tree.parent.resize(N);
    tree.count.resize(N);
    thrust::device_vector<int32_t> node_tx(N);
    write_nodes_kernel<<<grid_for(n), kThreads>>>(items, off, raw(lcp), raw(base), n, raw(tree.item), raw(tree.depth),
                                                  raw(node_tx));
    CUDA_CHECK_KERNEL();
    stage("construcao_nos", t);

    // Sparse table de mínimos sobre o LCP: log n passos paralelos.
    const int levels = sparse_levels(n);
    thrust::device_vector<int32_t> st(int64_t(n) * levels);
    thrust::copy(lcp.begin(), lcp.end(), st.begin());
    for (int j = 1; j < levels; ++j) {
        sparse_level_kernel<<<grid_for(n), kThreads>>>(raw(st), n, j);
        CUDA_CHECK_KERNEL();
    }
    stage("construcao_sparse", t);

    // 3.5 Pais (previous smaller value).
    parents_kernel<<<grid_for(N), kThreads>>>(raw(st), levels, raw(lcp), raw(base), n, N, raw(tree.depth),
                                              raw(node_tx), raw(tree.parent));
    CUDA_CHECK_KERNEL();
    stage("construcao_pais", t);

    // 3.6 Contadores (next smaller or equal), sem atômicos.
    counts_kernel<<<grid_for(N), kThreads>>>(raw(st), levels, weighted ? raw(W) : nullptr, n, N, raw(tree.depth),
                                             raw(node_tx), raw(tree.count));
    CUDA_CHECK_KERNEL();
    stage("construcao_contadores", t);

    // 3.8 Tabela de cabeçalho.
    {
        thrust::device_vector<int32_t> keys = tree.item;
        tree.header_nodes.resize(N);
        thrust::sequence(thrust::device, tree.header_nodes.begin(), tree.header_nodes.end());
        thrust::stable_sort_by_key(thrust::device, keys.begin(), keys.end(), tree.header_nodes.begin());
        tree.header_offsets.resize(n_items + 1);
        thrust::lower_bound(thrust::device, keys.begin(), keys.end(), thrust::counting_iterator<int32_t>(0),
                            thrust::counting_iterator<int32_t>(n_items + 1), tree.header_offsets.begin());
        thrust::device_vector<int32_t> cnt(N), ukeys(n_items), sums(n_items);
        thrust::gather(thrust::device, tree.header_nodes.begin(), tree.header_nodes.end(), tree.count.begin(),
                       cnt.begin());
        auto ends = thrust::reduce_by_key(thrust::device, keys.begin(), keys.end(), cnt.begin(), ukeys.begin(),
                                          sums.begin());
        tree.item_support.assign(n_items, 0);
        thrust::scatter(thrust::device, sums.begin(), ends.second, ukeys.begin(), tree.item_support.begin());
        mem.sample();
    }
    stage("construcao_cabecalho", t);
    times.add("construcao", total);

    // 3.7 Verificação cruzada dos contadores (fora da medição da construção).
    if (opt.cross_check) {
        const int64_t nnz = static_cast<int64_t>(db.items.size());
        thrust::device_vector<int32_t> key(nnz), val(nnz), ukey(N), usum(N);
        traversal_kernel<<<grid_for(n), kThreads>>>(raw(st), levels, off, raw(lcp), raw(base),
                                                    weighted ? raw(w_sorted) : nullptr, n, raw(key), raw(val));
        CUDA_CHECK_KERNEL();
        thrust::sort_by_key(thrust::device, key.begin(), key.end(), val.begin());
        auto ends = thrust::reduce_by_key(thrust::device, key.begin(), key.end(), val.begin(), ukey.begin(),
                                          usum.begin());
        bool ok = (ends.first - ukey.begin()) == N &&
                  thrust::equal(thrust::device, usum.begin(), usum.end(), tree.count.begin());
        if (stats) stats->counts_match = ok;
        mem.sample();
    }
    if (stats) stats->peak_bytes = mem.peak();
    return tree;
}

void download_tree(const DeviceTree& t, std::vector<int32_t>& item, std::vector<int32_t>& count,
                   std::vector<int32_t>& parent) {
    item.resize(t.n_nodes);
    count.resize(t.n_nodes);
    parent.resize(t.n_nodes);
    thrust::copy(t.item.begin(), t.item.end(), item.begin());
    thrust::copy(t.count.begin(), t.count.end(), count.begin());
    thrust::copy(t.parent.begin(), t.parent.end(), parent.begin());
}

}  // namespace gpu
}  // namespace fpg
