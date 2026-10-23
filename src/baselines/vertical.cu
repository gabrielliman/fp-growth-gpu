// Apriori (estilo GPApriori) e Eclat (estilo Frontier Expansion) em GPU com
// bitsets verticais. Ver vertical.cuh.

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cub/block/block_reduce.cuh>

#include "baselines/vertical.cuh"
#include "common/cuda_check.cuh"

namespace fpg {
namespace baselines {

namespace {

constexpr int kThreads = 256;  // tamanho de bloco usado nos dois artigos
constexpr int kMaxK = 128;     // tamanho máximo de candidato no Apriori
using BlockReduceT = cub::BlockReduce<int32_t, kThreads>;

// Palavras de 32 bits por bitset, alinhadas a 64 bytes (16 palavras) para
// leituras coalescidas, como nos dois artigos.
size_t words_for(int32_t n) { return ((size_t(n) + 31) / 32 + 15) / 16 * 16; }

class Timer {
public:
    Timer() {
        CUDA_CHECK(cudaEventCreate(&a_));
        CUDA_CHECK(cudaEventCreate(&b_));
        CUDA_CHECK(cudaEventRecord(a_));
    }
    ~Timer() {
        cudaEventDestroy(a_);
        cudaEventDestroy(b_);
    }
    float ms() {
        CUDA_CHECK(cudaEventRecord(b_));
        CUDA_CHECK(cudaEventSynchronize(b_));
        float t = 0;
        CUDA_CHECK(cudaEventElapsedTime(&t, a_, b_));
        return t;
    }

private:
    cudaEvent_t a_, b_;
};

// Bitset vertical de cada item: linha `row_of_rank[r]` recebe as transações que
// contêm o item de rank r.
std::vector<uint32_t> build_bitsets(const Csr& pre, int32_t n_items, const std::vector<int32_t>& row_of_rank,
                                    size_t W) {
    std::vector<uint32_t> bits(size_t(n_items) * W, 0);
    for (int32_t t = 0; t < pre.n_transactions(); ++t)
        for (int32_t p = pre.offsets[t]; p < pre.offsets[t + 1]; ++p)
            bits[size_t(row_of_rank[pre.items[p]]) * W + (t >> 5)] |= 1u << (t & 31);
    return bits;
}

// ---------------------------------------------------------------------------
// Apriori: interseção completa dos k bitsets de cada candidato.

__global__ void __launch_bounds__(kThreads)
    apriori_count_kernel(const uint32_t* __restrict__ bits, size_t W, const int32_t* __restrict__ cand, int k,
                         int64_t n_cand, int32_t* __restrict__ sup) {
    __shared__ int32_t items[kMaxK];
    __shared__ typename BlockReduceT::TempStorage tmp;
    for (int64_t c = blockIdx.x; c < n_cand; c += gridDim.x) {
        // Pré-carrega os itens do candidato em memória compartilhada.
        for (int j = threadIdx.x; j < k; j += kThreads) items[j] = cand[c * k + j];
        __syncthreads();
        int32_t cnt = 0;
        for (size_t w = threadIdx.x; w < W; w += kThreads) {
            uint32_t x = bits[size_t(items[0]) * W + w];
            for (int j = 1; j < k; ++j) x &= bits[size_t(items[j]) * W + w];
            cnt += __popc(x);
        }
        int32_t total = BlockReduceT(tmp).Sum(cnt);
        if (threadIdx.x == 0) sup[c] = total;
        __syncthreads();
    }
}

uint64_t hash_items(const int32_t* a, int k, int skip) {
    uint64_t h = 1469598103934665603ull;
    for (int j = 0; j < k; ++j) {
        if (j == skip) continue;
        h ^= uint64_t(uint32_t(a[j]));
        h *= 1099511628211ull;
        h ^= h >> 29;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Eclat: interseção de dois bitsets do pool, gravando o resultado no pool.

struct Work {
    int32_t a, b, dst;
};

__global__ void __launch_bounds__(kThreads)
    eclat_intersect_kernel(uint32_t* __restrict__ pool, size_t W, const Work* __restrict__ work, int n,
                           int32_t* __restrict__ sup) {
    __shared__ Work wk;
    __shared__ typename BlockReduceT::TempStorage tmp;
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        if (threadIdx.x == 0) wk = work[i];  // pré-carga da lista de trabalho
        __syncthreads();
        const uint32_t* a = pool + size_t(wk.a) * W;
        const uint32_t* b = pool + size_t(wk.b) * W;
        uint32_t* d = pool + size_t(wk.dst) * W;
        int32_t cnt = 0;
        for (size_t w = threadIdx.x; w < W; w += kThreads) {
            uint32_t x = a[w] & b[w];
            d[w] = x;
            cnt += __popc(x);
        }
        int32_t total = BlockReduceT(tmp).Sum(cnt);
        if (threadIdx.x == 0) sup[i] = total;
        __syncthreads();
    }
}

int resident_blocks(const void* kernel) {
    int dev = 0, sms = 0, per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
    return std::max(1, per_sm) * sms;
}

}  // namespace

// ===========================================================================

ItemsetList mine_apriori_gpu(const Csr& pre, int32_t n_items, const std::vector<int32_t>& rank_support,
                             int32_t minsup, StageTimes& times, VerticalStats& stats) {
    ItemsetList out;
    if (n_items == 0) return out;
    const size_t W = words_for(pre.n_transactions());

    CpuTimer ct;
    std::vector<int32_t> row(n_items);
    for (int32_t r = 0; r < n_items; ++r) row[r] = r;
    std::vector<uint32_t> h_bits = build_bitsets(pre, n_items, row, W);
    times.add("preprocessamento", ct.ms());

    Timer th;
    uint32_t* d_bits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, h_bits.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_bits, h_bits.data(), h_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    times.add("transferencia_h2d", th.ms());
    stats.gpu_bytes = h_bits.size() * sizeof(uint32_t);
    h_bits.clear();
    h_bits.shrink_to_fit();

    Timer tm;
    const int grid_cap = resident_blocks(reinterpret_cast<const void*>(apriori_count_kernel)) * 4;
    // Nível 1: itens frequentes (suporte vem da contagem).
    std::vector<int32_t> freq;  // itemsets frequentes do nível atual, k inteiros cada, em ordem lexicográfica
    for (int32_t r = 0; r < n_items; ++r) {
        out.add(&r, 1, rank_support[r]);
        freq.push_back(r);
    }
    int k = 1;
    stats.levels = 1;
    const int64_t batch = int64_t(1) << 22;
    int32_t *d_cand = nullptr, *d_sup = nullptr;
    size_t cand_cap = 0;
    std::vector<int32_t> cand, sup;
    while (!freq.empty()) {
        const int64_t m = int64_t(freq.size()) / k;
        // Poda de Apriori: conjunto (hash) dos k-itemsets frequentes. Uma colisão
        // só deixa de podar um candidato, cujo suporte é contado de qualquer forma.
        std::unordered_set<uint64_t> fset;
        if (k >= 2) {
            fset.reserve(size_t(m) * 2);
            for (int64_t i = 0; i < m; ++i) fset.insert(hash_items(&freq[i * k], k, -1));
        }
        // Junção por classes de equivalência: itemsets com o mesmo prefixo de k-1 itens.
        cand.clear();
        std::vector<int32_t> c(k + 1);
        for (int64_t g = 0; g < m;) {
            int64_t e = g + 1;
            while (e < m && std::equal(&freq[g * k], &freq[g * k] + (k - 1), &freq[e * k])) ++e;
            for (int64_t i = g; i < e; ++i)
                for (int64_t j = i + 1; j < e; ++j) {
                    std::copy(&freq[i * k], &freq[i * k] + k, c.begin());
                    c[k] = freq[j * k + k - 1];
                    bool ok = true;
                    for (int s = 0; s < k - 1 && ok && k >= 2; ++s)  // subconjuntos sem o item s
                        ok = fset.count(hash_items(c.data(), k + 1, s)) > 0;
                    if (ok) cand.insert(cand.end(), c.begin(), c.end());
                }
            g = e;
        }
        const int kk = k + 1;
        if (kk > kMaxK) throw std::runtime_error("candidato maior que kMaxK");
        const int64_t n_cand = int64_t(cand.size()) / kk;
        if (n_cand == 0) break;
        stats.candidates += n_cand;
        sup.assign(n_cand, 0);
        // Contagem na GPU em lotes de candidatos.
        for (int64_t lo = 0; lo < n_cand; lo += batch) {
            const int64_t nb = std::min(batch, n_cand - lo);
            if (size_t(nb * kk) > cand_cap) {
                if (d_cand) CUDA_CHECK(cudaFree(d_cand));
                if (d_sup) CUDA_CHECK(cudaFree(d_sup));
                cand_cap = size_t(nb * kk);
                CUDA_CHECK(cudaMalloc(&d_cand, cand_cap * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_sup, size_t(batch) * sizeof(int32_t)));
            }
            CUDA_CHECK(cudaMemcpy(d_cand, &cand[lo * kk], nb * kk * sizeof(int32_t), cudaMemcpyHostToDevice));
            const int grid = int(std::min<int64_t>(nb, grid_cap));
            apriori_count_kernel<<<grid, kThreads>>>(d_bits, W, d_cand, kk, nb, d_sup);
            CUDA_CHECK_KERNEL();
            ++stats.kernel_launches;
            CUDA_CHECK(cudaMemcpy(&sup[lo], d_sup, nb * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }
        // Mantém os frequentes (a ordem lexicográfica é preservada).
        freq.clear();
        for (int64_t i = 0; i < n_cand; ++i)
            if (sup[i] >= minsup) {
                out.add(&cand[i * kk], kk, sup[i]);
                freq.insert(freq.end(), &cand[i * kk], &cand[i * kk] + kk);
            }
        k = kk;
        ++stats.levels;
    }
    if (d_cand) CUDA_CHECK(cudaFree(d_cand));
    if (d_sup) CUDA_CHECK(cudaFree(d_sup));
    CUDA_CHECK(cudaFree(d_bits));
    times.add("mineracao", tm.ms());
    return out;
}

// ===========================================================================

ItemsetList mine_eclat_gpu(const Csr& pre, int32_t n_items, const std::vector<int32_t>& rank_support,
                           int32_t minsup, StageTimes& times, VerticalStats& stats, int64_t pool_slots) {
    ItemsetList out;
    if (n_items == 0) return out;
    const size_t W = words_for(pre.n_transactions());

    // Itens em ordem de frequência crescente: a linha p guarda o rank n_items-1-p.
    CpuTimer ct;
    std::vector<int32_t> row(n_items);
    for (int32_t r = 0; r < n_items; ++r) row[r] = n_items - 1 - r;
    std::vector<uint32_t> h_bits = build_bitsets(pre, n_items, row, W);
    times.add("preprocessamento", ct.ms());

    // Pool de listas verticais (gerenciador do host com pilha de posições livres).
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    int64_t n_slots = pool_slots ? pool_slots
                                 : int64_t(std::min<size_t>(free_b * 35 / 100, size_t(8) << 30) /
                                           (W * sizeof(uint32_t)));
    if (n_slots < n_items + 2) n_slots = n_items + 2;
    stats.gpu_bytes = size_t(n_slots) * W * sizeof(uint32_t);

    Timer th;
    uint32_t* pool = nullptr;
    CUDA_CHECK(cudaMalloc(&pool, size_t(n_slots) * W * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(pool, h_bits.data(), h_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    times.add("transferencia_h2d", th.ms());
    h_bits.clear();
    h_bits.shrink_to_fit();

    Timer tm;
    std::vector<int32_t> free_slots;
    for (int64_t s = n_slots - 1; s >= n_items; --s) free_slots.push_back(int32_t(s));

    struct Member {
        int32_t item, slot, sup;
    };
    struct Class {
        std::vector<int32_t> prefix;  // ranks
        std::vector<Member> members;
        size_t next = 0;              // membro i sendo expandido
        size_t next_j = 0;            // próximo parceiro j de i
        std::unique_ptr<Class> open;  // classe filha de i em construção
    };
    std::vector<std::unique_ptr<Class>> stack;
    stack.push_back(std::make_unique<Class>());
    for (int32_t p = 0; p < n_items; ++p) {
        int32_t r = n_items - 1 - p;
        out.add(&r, 1, rank_support[r]);
        stack[0]->members.push_back(Member{r, p, rank_support[r]});
    }

    const int eps = resident_blocks(reinterpret_cast<const void*>(eclat_intersect_kernel));
    Work* d_work = nullptr;
    int32_t* d_sup = nullptr;
    size_t work_cap = 0;
    std::vector<Work> work;
    std::vector<int32_t> sup;
    struct Origin {
        Class* child;   // classe filha que recebe o candidato
        int32_t item;   // item acrescentado
    };
    std::vector<Origin> origin;
    std::vector<std::unique_ptr<Class>> closed;  // filhas completas neste lote
    std::vector<int32_t> to_free;
    std::vector<int32_t> buf;

    while (!stack.empty()) {
        // Desempilha classes até juntar ε candidatos (Frontier Expansion). Um
        // membro pode ter seus candidatos gerados em vários lotes quando o pool
        // não tem posições livres para todos: a classe filha fica aberta.
        work.clear();
        origin.clear();
        closed.clear();
        to_free.clear();
        while (!stack.empty() && int(work.size()) < eps) {
            Class& cls = *stack.back();
            if (cls.next >= cls.members.size()) {
                for (auto& mb : cls.members) to_free.push_back(mb.slot);
                stack.pop_back();
                continue;
            }
            const size_t i = cls.next;
            if (cls.next_j <= i) cls.next_j = i + 1;
            if (!cls.open) {
                cls.open = std::make_unique<Class>();
                cls.open->prefix = cls.prefix;
                cls.open->prefix.push_back(cls.members[i].item);
            }
            while (cls.next_j < cls.members.size() && int(work.size()) < eps && !free_slots.empty()) {
                const size_t j = cls.next_j++;
                int32_t dst = free_slots.back();
                free_slots.pop_back();
                work.push_back(Work{cls.members[i].slot, cls.members[j].slot, dst});
                origin.push_back(Origin{cls.open.get(), cls.members[j].item});
            }
            if (cls.next_j >= cls.members.size()) {
                closed.push_back(std::move(cls.open));
                ++cls.next;
                continue;
            }
            if (free_slots.empty()) {
                if (work.empty()) throw std::runtime_error("pool de listas verticais esgotado");
                break;  // processa o lote; posições voltam ao pool
            }
        }
        if (work.empty()) {
            for (int32_t s2 : to_free) free_slots.push_back(s2);
            for (auto& ch : closed)
                for (auto& mb : ch->members) free_slots.push_back(mb.slot);
            continue;
        }

        // Contagem na GPU.
        const int n = int(work.size());
        if (size_t(n) > work_cap) {
            if (d_work) CUDA_CHECK(cudaFree(d_work));
            if (d_sup) CUDA_CHECK(cudaFree(d_sup));
            work_cap = size_t(std::max(n, eps * 2));
            CUDA_CHECK(cudaMalloc(&d_work, work_cap * sizeof(Work)));
            CUDA_CHECK(cudaMalloc(&d_sup, work_cap * sizeof(int32_t)));
        }
        CUDA_CHECK(cudaMemcpy(d_work, work.data(), n * sizeof(Work), cudaMemcpyHostToDevice));
        eclat_intersect_kernel<<<n, kThreads>>>(pool, W, d_work, n, d_sup);
        CUDA_CHECK_KERNEL();
        ++stats.kernel_launches;
        stats.candidates += n;
        sup.resize(n);
        CUDA_CHECK(cudaMemcpy(sup.data(), d_sup, n * sizeof(int32_t), cudaMemcpyDeviceToHost));

        // Frequentes viram membros das classes filhas; os outros liberam a posição.
        for (int w = 0; w < n; ++w) {
            Class& ch = *origin[w].child;
            if (sup[w] >= minsup) {
                buf.assign(ch.prefix.begin(), ch.prefix.end());
                buf.push_back(origin[w].item);
                out.add(buf.data(), int(buf.size()), sup[w]);
                ch.members.push_back(Member{origin[w].item, work[w].dst, sup[w]});
            } else {
                free_slots.push_back(work[w].dst);
            }
        }
        for (int32_t s2 : to_free) free_slots.push_back(s2);
        // Filhas completas com 2+ membros vão para a pilha (ordenadas por suporte
        // decrescente, como no artigo); as de 1 membro não geram candidatos.
        for (auto& ch : closed) {
            if (ch->members.size() < 2) {
                for (auto& mb : ch->members) free_slots.push_back(mb.slot);
                continue;
            }
            std::stable_sort(ch->members.begin(), ch->members.end(),
                             [](const Member& x, const Member& y) { return x.sup > y.sup; });
            stack.push_back(std::move(ch));
        }
    }
    if (d_work) CUDA_CHECK(cudaFree(d_work));
    if (d_sup) CUDA_CHECK(cudaFree(d_sup));
    CUDA_CHECK(cudaFree(pool));
    times.add("mineracao", tm.ms());
    return out;
}

}  // namespace baselines
}  // namespace fpg
