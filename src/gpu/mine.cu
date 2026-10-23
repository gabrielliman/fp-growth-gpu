// Fases 4 e 5: mineração na GPU.
//
// Uma tarefa é o par (sufixo S, árvore linearizada T). Processar uma tarefa
// significa, para cada item frequente y de T (do menos para o mais frequente),
// emitir S ∪ {y}, construir a árvore condicional de y e continuar a recursão.
// A recursão é uma pilha explícita de quadros na arena do bloco.
//
// - Estática (Fase 4): um bloco por item da árvore raiz; o bloco resolve
//   sozinho toda a recursão do seu item.
// - Dinâmica (Fase 5): blocos persistentes consomem uma fila global de tarefas;
//   árvores condicionais com base >= grain viram tarefas novas na fila.
//
// Memória de um bloco: uma arena com duas pontas. Por baixo ficam os quadros
// da pilha e as árvores condicionais (LIFO); por cima, os temporários da
// construção da árvore, liberados ao fim de cada construção.

#include <algorithm>
#include <memory>
#include <cstdio>
#include <stdexcept>

#include <cub/block/block_scan.cuh>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "gpu/gpu.cuh"
#include "gpu/tree_core.cuh"

namespace fpg {
namespace gpu {

namespace {

constexpr int kBlock = 256;
using BlockScanT = cub::BlockScan<int32_t, kBlock>;

// Árvore linearizada (a global, uma projetada ou uma condicional numa arena/pool).
struct TreeView {
    int32_t n_nodes = 0, n_items = 0, n_fitems = 0, single = 0, base_items = 0;
    int32_t *item = nullptr, *count = nullptr, *parent = nullptr, *depth = nullptr;
    int32_t *hdr_off = nullptr, *hdr_nodes = nullptr, *support = nullptr;
    int32_t* fitems = nullptr;  // itens frequentes em ordem decrescente de rank
};

struct Frame {
    TreeView t;
    int32_t next;   // próximo índice em t.fitems
    int32_t depth;  // tamanho do sufixo associado a t
    size_t mark;    // fundo da arena antes de t (liberado ao desempilhar)
};

enum TaskKind : int32_t { kRootItem = 0, kSubtree = 1 };

struct Task {
    TreeView t;          // kSubtree: árvore no pool
    int32_t* suffix;     // kSubtree: sufixo no pool
    int32_t depth;       // tamanho do sufixo
    int32_t root;        // índice da tarefa raiz que originou esta (contagem por raiz)
    int32_t kind;
};

struct MineCtx {
    TreeView root;
    const int32_t* prefix;   // sufixo comum a todas as tarefas desta execução
    int32_t prefix_len;
    int32_t root_lo, root_hi;  // faixa de índices em root.fitems processada nesta chamada
    int32_t max_depth;
    int32_t minsup;
    int32_t grain;
    int32_t write;  // 0: passada de contagem, 1: passada de escrita

    // Contagem por raiz (passada 1) e reserva global (passada 2).
    unsigned long long *root_sets, *root_items;
    unsigned long long *cnt_sets, *cnt_items;
    unsigned long long cap_sets, cap_items;
    int32_t* out_items;
    int64_t* out_start;
    int32_t* out_len;
    int32_t* out_sup;
    int* out_overflow;

    // Tarefas que estouraram a arena (só na passada 2).
    int32_t* ovf_suffix;  // max_ovf * max_depth
    int32_t* ovf_len;
    int* ovf_count;
    int max_ovf;

    // Arenas.
    char* arenas;
    size_t arena_bytes;
    int* slot_lock;
    int n_slots;
    int hist_cap;  // capacidade do histograma em memória compartilhada (itens)

    // Fila e pool (dinâmica).
    Task* q_task;
    unsigned long long* q_seq;
    unsigned long long *q_head, *q_tail;
    unsigned long long q_cap;
    unsigned long long* pending;
    char* pool;
    unsigned long long* pool_used;
    unsigned long long pool_bytes;
    unsigned long long* n_enqueued;

    // Instrumentação por bloco.
    long long* blk_cycles;
    long long* blk_busy;
    int* blk_tasks;
    long long* blk_sets;
    int* blk_slot;
    int* error_flag;
};

struct Shared {
    typename BlockScanT::TempStorage scan;
    unsigned long long u0, u1;
    int32_t i0, i1;
    Task task;
};

// ---------------------------------------------------------------------------
// Arena: todas as threads mantêm cópias idênticas dos ponteiros, porque todas
// executam a mesma sequência de alocações com os mesmos tamanhos.

struct Arena {
    char* base;
    size_t bottom, top;
    bool failed;

    static __device__ size_t align(size_t b) { return (b + 15) & ~size_t(15); }
    template <typename T>
    __device__ T* push(int64_t n) {
        size_t b = align(size_t(n) * sizeof(T));
        if (failed || bottom + b > top) {
            failed = true;
            return nullptr;
        }
        T* p = reinterpret_cast<T*>(base + bottom);
        bottom += b;
        return p;
    }
    template <typename T>
    __device__ T* scratch(int64_t n) {
        size_t b = align(size_t(n) * sizeof(T));
        if (failed || top < bottom + b) {
            failed = true;
            return nullptr;
        }
        top -= b;
        return reinterpret_cast<T*>(base + top);
    }
};

// Scan exclusivo de bloco sobre n valores gerados por value(i); escreve out[0..n]
// (out[n] = total) e devolve o total a todas as threads. out pode ser a origem
// de value(i) (cada i é lido e escrito pela mesma thread).
template <typename F>
__device__ int32_t block_scan(F value, int32_t n, int32_t* out, Shared& sh) {
    int32_t carry = 0;
    for (int32_t base = 0; base < n; base += kBlock) {
        int32_t i = base + threadIdx.x;
        int32_t v = i < n ? value(i) : 0;
        int32_t ex, agg;
        BlockScanT(sh.scan).ExclusiveSum(v, ex, agg);
        if (i < n) out[i] = carry + ex;
        carry += agg;
        __syncthreads();
    }
    if (threadIdx.x == 0) out[n] = carry;
    __syncthreads();
    return carry;
}

// Merge sort de bloco de índices de caminhos. Cada passo posiciona cada elemento
// no resultado da intercalação por busca binária na outra metade (estável).
__device__ int32_t* block_merge_sort(int32_t* a, int32_t* b, int32_t n, const int32_t* items, const int32_t* off) {
    for (int32_t w = 1; w < n; w *= 2) {
        for (int32_t i = threadIdx.x; i < n; i += kBlock) {
            int32_t start = (i / (2 * w)) * (2 * w);
            int32_t mid = min(start + w, n), end = min(start + 2 * w, n);
            int32_t x = a[i], dest;
            if (i < mid) {
                int32_t lo = mid, hi = end;  // elementos da direita estritamente menores
                while (lo < hi) {
                    int32_t m = (lo + hi) >> 1;
                    if (path_less(items, off, a[m], x)) lo = m + 1; else hi = m;
                }
                dest = i + (lo - mid);
            } else {
                int32_t lo = start, hi = mid;  // elementos da esquerda menores ou iguais
                while (lo < hi) {
                    int32_t m = (lo + hi) >> 1;
                    if (!path_less(items, off, x, a[m])) lo = m + 1; else hi = m;
                }
                dest = start + (i - mid) + (lo - start);
            }
            b[dest] = x;
        }
        __syncthreads();
        int32_t* t = a;
        a = b;
        b = t;
    }
    return a;
}

// ---------------------------------------------------------------------------
// 4.2–4.4: árvore condicional de T para o item y, construída pelo bloco.
// Devolve false se a arena estourar (nada fica alocado nesse caso).

__device__ bool build_conditional(const TreeView& T, int32_t y, int32_t minsup, Arena& ar, TreeView& C,
                                  int32_t* smem_hist, int hist_cap, Shared& sh) {
    const int tid = threadIdx.x;
    const size_t top_mark = ar.top, bottom_mark = ar.bottom;
    auto fail = [&]() {
        ar.top = top_mark;
        ar.bottom = bottom_mark;
        ar.failed = false;
        return false;
    };
    C = TreeView{};
    C.n_items = y;

    // 4.2 Extração da base: offsets pelas profundidades e subida pelos pais.
    const int32_t h0 = T.hdr_off[y], m = T.hdr_off[y + 1] - h0;
    const int32_t* vs = T.hdr_nodes + h0;
    int32_t* poff = ar.scratch<int32_t>(m + 1);
    int32_t* cnt = y <= hist_cap ? smem_hist : ar.scratch<int32_t>(y);
    if (ar.failed) return fail();
    const int32_t P = block_scan([&](int32_t j) { return T.depth[vs[j]]; }, m, poff, sh);
    int32_t* buf = ar.scratch<int32_t>(P);
    if (ar.failed) return fail();
    for (int32_t i = tid; i < y; i += kBlock) cnt[i] = 0;
    __syncthreads();
    // 4.3 Histograma ponderado durante a extração.
    for (int32_t j = tid; j < m; j += kBlock) {
        int32_t v = vs[j], w = T.count[v], b = poff[j];
        for (int32_t u = T.parent[v]; u != -1; u = T.parent[u]) {
            int32_t it = T.item[u];
            buf[b + T.depth[u]] = it;
            atomicAdd(&cnt[it], w);
        }
    }
    __syncthreads();

    // 4.3 Filtro e compactação dos caminhos (removendo os vazios).
    int32_t* noff = ar.scratch<int32_t>(m + 1);
    int32_t* pidx = ar.scratch<int32_t>(m + 1);
    if (ar.failed) return fail();
    const int32_t P2 = block_scan(
        [&](int32_t j) {
            int32_t c = 0;
            for (int32_t p = poff[j]; p < poff[j + 1]; ++p) c += cnt[buf[p]] >= minsup;
            return c;
        },
        m, noff, sh);
    const int32_t n2 = block_scan([&](int32_t j) { return int32_t(noff[j + 1] > noff[j]); }, m, pidx, sh);
    C.base_items = P2;
    if (n2 == 0) {
        ar.top = top_mark;
        return true;  // árvore vazia
    }
    int32_t* citems = ar.scratch<int32_t>(P2);
    int32_t* coff = ar.scratch<int32_t>(n2 + 1);
    int32_t* cw = ar.scratch<int32_t>(n2);
    int32_t* ord = ar.scratch<int32_t>(n2);
    int32_t* tmp = ar.scratch<int32_t>(n2);
    if (ar.failed) return fail();
    for (int32_t j = tid; j < m; j += kBlock) {
        if (noff[j + 1] == noff[j]) continue;
        int32_t k = pidx[j], o = noff[j];
        coff[k] = o;
        cw[k] = T.count[vs[j]];
        ord[k] = k;
        for (int32_t p = poff[j]; p < poff[j + 1]; ++p)
            if (cnt[buf[p]] >= minsup) citems[o++] = buf[p];
    }
    if (tid == 0) coff[n2] = P2;
    __syncthreads();

    // 4.4 Ordenação lexicográfica dos caminhos e reescrita em ordem.
    int32_t* sorted = block_merge_sort(ord, tmp, n2, citems, coff);
    int32_t* soff = ar.scratch<int32_t>(n2 + 1);
    int32_t* sitems = ar.scratch<int32_t>(P2);
    int32_t* lcp = ar.scratch<int32_t>(n2);
    int32_t* nbase = ar.scratch<int32_t>(n2 + 1);
    int32_t* W = ar.scratch<int32_t>(n2 + 1);
    if (ar.failed) return fail();
    block_scan([&](int32_t i) { return coff[sorted[i] + 1] - coff[sorted[i]]; }, n2, soff, sh);
    for (int32_t i = tid; i < n2; i += kBlock) {
        int32_t src = coff[sorted[i]], dst = soff[i], len = soff[i + 1] - dst;
        for (int32_t d = 0; d < len; ++d) sitems[dst + d] = citems[src + d];
    }
    __syncthreads();
    block_scan([&](int32_t i) { return cw[sorted[i]]; }, n2, W, sh);

    // LCP, nós novos e teste de caminho único (cada caminho estende o anterior).
    bool not_single = false;
    for (int32_t i = tid; i < n2; i += kBlock) {
        int32_t l = path_lcp(sitems, soff, i);
        lcp[i] = l;
        nbase[i] = soff[i + 1] - soff[i] - l;
        if (i > 0 && l != soff[i] - soff[i - 1]) not_single = true;
    }
    C.single = !__syncthreads_or(not_single);
    const int32_t N = block_scan([&](int32_t i) { return nbase[i]; }, n2, nbase, sh);

    // Árvore no fundo da arena.
    C.n_nodes = N;
    C.item = ar.push<int32_t>(N);
    C.count = ar.push<int32_t>(N);
    C.parent = ar.push<int32_t>(N);
    C.depth = ar.push<int32_t>(N);
    C.hdr_off = ar.push<int32_t>(y + 1);
    C.hdr_nodes = ar.push<int32_t>(N);
    C.support = ar.push<int32_t>(y);
    C.fitems = ar.push<int32_t>(y);
    const int levels = sparse_levels(n2);
    int32_t* node_tx = ar.scratch<int32_t>(N);
    int32_t* st = ar.scratch<int32_t>(int64_t(n2) * levels);
    int32_t* cursor = ar.scratch<int32_t>(y + 1);
    if (ar.failed) return fail();

    write_nodes(sitems, soff, lcp, nbase, n2, C.item, C.depth, node_tx, tid, kBlock);
    for (int32_t i = tid; i < n2; i += kBlock) st[i] = lcp[i];
    for (int32_t i = tid; i <= y; i += kBlock) cursor[i] = 0;
    __syncthreads();
    for (int j = 1; j < levels; ++j) {
        sparse_level(st, n2, j, tid, kBlock);
        __syncthreads();
    }
    node_parents(st, levels, lcp, nbase, n2, N, C.depth, node_tx, C.parent, tid, kBlock);
    node_counts(st, levels, W, n2, N, C.depth, node_tx, C.count, tid, kBlock);

    // Cabeçalho: contagem de nós por item, scan e posicionamento.
    for (int32_t v = tid; v < N; v += kBlock) atomicAdd(&cursor[C.item[v]], 1);
    __syncthreads();
    block_scan([&](int32_t i) { return cursor[i]; }, y, C.hdr_off, sh);
    for (int32_t i = tid; i < y; i += kBlock) {
        cursor[i] = C.hdr_off[i];
        C.support[i] = cnt[i] >= minsup ? cnt[i] : 0;
    }
    __syncthreads();
    for (int32_t v = tid; v < N; v += kBlock) C.hdr_nodes[atomicAdd(&cursor[C.item[v]], 1)] = v;
    // Itens frequentes em ordem decrescente.
    C.n_fitems = block_scan([&](int32_t i) { return int32_t(cnt[y - 1 - i] >= minsup); }, y, cursor, sh);
    for (int32_t i = tid; i < y; i += kBlock)
        if (cnt[y - 1 - i] >= minsup) C.fitems[cursor[i]] = y - 1 - i;
    __syncthreads();

    ar.top = top_mark;
    return true;
}

// ---------------------------------------------------------------------------
// Emissão de itemsets (4.5 e 4.7).

__device__ __forceinline__ unsigned long long popcount_prefix(unsigned long long m) {
    // Total de bits 1 nos inteiros 0 .. m-1.
    unsigned long long s = 0;
    for (int b = 0; b < 63; ++b) {
        unsigned long long cycle = 1ull << (b + 1), half = 1ull << b;
        if (half > m) break;
        s += (m / cycle) * half;
        unsigned long long r = m % cycle;
        s += r > half ? r - half : 0;
    }
    return s;
}

// Thread 0: emite suffix[0..len) com suporte sup.
__device__ void emit_one(const MineCtx& cx, int32_t root, const int32_t* suffix, int32_t len, int32_t sup,
                         long long& sets) {
    ++sets;
    if (!cx.write) {
        atomicAdd(&cx.root_sets[root], 1ull);
        atomicAdd(&cx.root_items[root], (unsigned long long)len);
        return;
    }
    unsigned long long s = atomicAdd(cx.cnt_sets, 1ull);
    unsigned long long it = atomicAdd(cx.cnt_items, (unsigned long long)len);
    if (s >= cx.cap_sets || it + len > cx.cap_items) {
        atomicExch(cx.out_overflow, 1);
        return;
    }
    for (int32_t k = 0; k < len; ++k) cx.out_items[it + k] = suffix[k];
    cx.out_start[s] = int64_t(it);
    cx.out_len[s] = len;
    cx.out_sup[s] = sup;
}

// 4.5 Caminho único: todas as combinações não vazias dos L itens do caminho,
// com suporte do nó mais profundo escolhido. Em C.single os ids dos nós
// coincidem com a profundidade.
__device__ void emit_single_path(const MineCtx& cx, int32_t root, const int32_t* suffix, int32_t k, const TreeView& C,
                                 long long& sets, Shared& sh) {
    const int32_t L = C.n_nodes;
    if (L > 40) {  // 2^40 itemsets: não cabe em lugar nenhum
        if (threadIdx.x == 0) atomicExch(cx.error_flag, 1);
        return;
    }
    const unsigned long long total = (1ull << L) - 1;
    const unsigned long long total_items = total * k + (unsigned long long)L * (1ull << (L - 1));
    if (threadIdx.x == 0) sets += total;
    if (!cx.write) {
        if (threadIdx.x == 0) {
            atomicAdd(&cx.root_sets[root], total);
            atomicAdd(&cx.root_items[root], total_items);
        }
        return;
    }
    if (threadIdx.x == 0) {
        sh.u0 = atomicAdd(cx.cnt_sets, total);
        sh.u1 = atomicAdd(cx.cnt_items, total_items);
        if (sh.u0 + total > cx.cap_sets || sh.u1 + total_items > cx.cap_items) atomicExch(cx.out_overflow, 1);
    }
    __syncthreads();
    const unsigned long long s0 = sh.u0, i0 = sh.u1;
    __syncthreads();
    if (s0 + total > cx.cap_sets || i0 + total_items > cx.cap_items) return;
    for (unsigned long long mask = threadIdx.x + 1; mask <= total; mask += kBlock) {
        unsigned long long s = s0 + mask - 1;
        unsigned long long it = i0 + (mask - 1) * k + popcount_prefix(mask);
        cx.out_start[s] = int64_t(it);
        int32_t len = k + __popcll(mask);
        cx.out_len[s] = len;
        cx.out_sup[s] = C.count[63 - __clzll(mask)];
        for (int32_t j = 0; j < k; ++j) cx.out_items[it++] = suffix[j];
        for (int32_t b = 0; b < L; ++b)
            if (mask >> b & 1) cx.out_items[it++] = C.item[b];
    }
}

// ---------------------------------------------------------------------------
// Fila global (Fase 5): vetor circular com head/tail atômicos e um número de
// sequência por posição (q_seq = posição + 1 quando a tarefa está publicada,
// 0 quando a posição está livre).

__device__ bool try_enqueue(const MineCtx& cx, const Task& t) {
    volatile unsigned long long* vtail = cx.q_tail;
    volatile unsigned long long* vhead = cx.q_head;
    for (int attempt = 0; attempt < 64; ++attempt) {
        unsigned long long tl = *vtail, hd = *vhead;
        if (tl - hd >= cx.q_cap) return false;  // 5.6 fila cheia
        if (atomicCAS(cx.q_tail, tl, tl + 1) != tl) continue;
        unsigned long long slot = tl % cx.q_cap;
        volatile unsigned long long* vseq = cx.q_seq + slot;
        while (*vseq != 0) __nanosleep(32);  // consumidor anterior ainda lendo
        cx.q_task[slot] = t;
        __threadfence();
        atomicExch(cx.q_seq + slot, tl + 1);
        return true;
    }
    return false;
}

__device__ bool try_dequeue(const MineCtx& cx, Task& out) {
    volatile unsigned long long* vtail = cx.q_tail;
    volatile unsigned long long* vhead = cx.q_head;
    while (true) {
        unsigned long long hd = *vhead, tl = *vtail;
        if (hd >= tl) return false;
        if (atomicCAS(cx.q_head, hd, hd + 1) != hd) continue;
        unsigned long long slot = hd % cx.q_cap;
        volatile unsigned long long* vseq = cx.q_seq + slot;
        while (*vseq != hd + 1) __nanosleep(32);  // produtor ainda escrevendo
        __threadfence();
        out = cx.q_task[slot];
        __threadfence();
        atomicExch(cx.q_seq + slot, 0ull);
        return true;
    }
}

// Copia a árvore C e o sufixo para o pool e enfileira. Devolve false se não houver
// espaço no pool ou na fila (o chamador processa localmente).
__device__ bool publish_subtree(const MineCtx& cx, const TreeView& C, const int32_t* suffix, int32_t depth,
                                int32_t root, Shared& sh) {
    const int32_t N = C.n_nodes, y = C.n_items;
    auto al = [](size_t n) { return (n * 4 + 15) & ~size_t(15); };
    const size_t bytes = 5 * al(N) + al(y + 1) + 2 * al(y) + al(depth);
    if (threadIdx.x == 0) {
        unsigned long long off = atomicAdd(cx.pool_used, (unsigned long long)bytes);
        sh.u0 = off + bytes <= cx.pool_bytes ? off : ~0ull;
    }
    __syncthreads();
    const unsigned long long off = sh.u0;
    __syncthreads();
    if (off == ~0ull) return false;
    char* p = cx.pool + off;
    auto take = [&](size_t n) {
        int32_t* r = reinterpret_cast<int32_t*>(p);
        p += al(n);
        return r;
    };
    TreeView D = C;
    D.item = take(N);
    D.count = take(N);
    D.parent = take(N);
    D.depth = take(N);
    D.hdr_nodes = take(N);
    D.hdr_off = take(y + 1);
    D.support = take(y);
    D.fitems = take(y);
    int32_t* suf = take(depth);
    for (int32_t i = threadIdx.x; i < N; i += kBlock) {
        D.item[i] = C.item[i];
        D.count[i] = C.count[i];
        D.parent[i] = C.parent[i];
        D.depth[i] = C.depth[i];
        D.hdr_nodes[i] = C.hdr_nodes[i];
    }
    for (int32_t i = threadIdx.x; i <= y; i += kBlock) D.hdr_off[i] = C.hdr_off[i];
    for (int32_t i = threadIdx.x; i < y; i += kBlock) D.support[i] = C.support[i];
    for (int32_t i = threadIdx.x; i < C.n_fitems; i += kBlock) D.fitems[i] = C.fitems[i];
    for (int32_t i = threadIdx.x; i < depth; i += kBlock) suf[i] = suffix[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        Task t{D, suf, depth, root, kSubtree};
        atomicAdd(cx.pending, 1ull);  // 5.5: antes de publicar
        bool ok = try_enqueue(cx, t);
        if (ok) atomicAdd(cx.n_enqueued, 1ull);
        else atomicAdd(cx.pending, ~0ull);  // desfaz o incremento
        sh.i0 = ok;
    }
    __syncthreads();
    const bool ok = sh.i0;
    __syncthreads();
    return ok;
}

// ---------------------------------------------------------------------------
// Processamento de tarefas com pilha explícita (4.6).

struct BlockWork {
    long long sets = 0;  // significativo só na thread 0
    int tasks = 0;
};

// Expande o item y da árvore T (sufixo com depth itens já em suffix): emite
// suffix ∪ {y}, constrói a árvore condicional e decide o destino dela.
// Devolve true se um quadro novo foi empilhado.
__device__ bool expand(const MineCtx& cx, const TreeView& T, int32_t y, int32_t depth, int32_t root, int32_t* suffix,
                       Frame* frames, int32_t& n_frames, Arena& ar, int32_t* smem_hist, bool dynamic, BlockWork& bw,
                       Shared& sh) {
    if (threadIdx.x == 0) {
        suffix[depth] = y;
        emit_one(cx, root, suffix, depth + 1, T.support[y], bw.sets);
    }
    __syncthreads();
    const size_t mark = ar.bottom;
    TreeView C;
    ++bw.tasks;
    if (!build_conditional(T, y, cx.minsup, ar, C, smem_hist, cx.hist_cap, sh)) {
        // 4.8 Estouro: registra suffix ∪ {y} para o host.
        if (threadIdx.x == 0 && cx.write) {
            int k = atomicAdd(cx.ovf_count, 1);
            if (k < cx.max_ovf) {
                for (int32_t j = 0; j <= depth; ++j) cx.ovf_suffix[int64_t(k) * cx.max_depth + j] = suffix[j];
                cx.ovf_len[k] = depth + 1;
            } else {
                atomicExch(cx.error_flag, 2);
            }
        }
        __syncthreads();
        return false;
    }
    if (C.n_nodes == 0) {
        ar.bottom = mark;
        return false;
    }
    if (C.single) {
        emit_single_path(cx, root, suffix, depth + 1, C, bw.sets, sh);
        __syncthreads();
        ar.bottom = mark;
        return false;
    }
    // 5.3/5.4 Na dinâmica, árvores com base grande viram tarefas da fila.
    if (dynamic && C.base_items >= cx.grain && publish_subtree(cx, C, suffix, depth + 1, root, sh)) {
        ar.bottom = mark;
        return false;
    }
    if (threadIdx.x == 0) frames[n_frames] = Frame{C, 0, depth + 1, mark};
    ++n_frames;
    __syncthreads();
    return true;
}

// Esvazia a pilha de quadros (busca em profundidade local).
__device__ void run_stack(const MineCtx& cx, int32_t root, int32_t* suffix, Frame* frames, int32_t n_frames,
                          Arena& ar, int32_t* smem_hist, bool dynamic, BlockWork& bw, Shared& sh) {
    while (n_frames > 0) {
        Frame& F = frames[n_frames - 1];
        const int32_t next = F.next;
        if (next >= F.t.n_fitems) {
            ar.bottom = F.mark;
            --n_frames;
            __syncthreads();
            continue;
        }
        const TreeView t = F.t;
        const int32_t depth = F.depth;
        __syncthreads();
        if (threadIdx.x == 0) F.next = next + 1;
        __syncthreads();
        expand(cx, t, t.fitems[next], depth, root, suffix, frames, n_frames, ar, smem_hist, dynamic, bw, sh);
    }
}

// Reserva no fundo da arena a pilha de quadros e o sufixo, e copia o prefixo.
__device__ Arena open_arena(const MineCtx& cx, int slot, Frame*& frames, int32_t*& suffix) {
    Arena ar{cx.arenas + size_t(slot) * cx.arena_bytes, 0, cx.arena_bytes, false};
    frames = ar.push<Frame>(cx.max_depth + 1);
    suffix = ar.push<int32_t>(cx.max_depth + 1);
    for (int32_t i = threadIdx.x; i < cx.prefix_len; i += kBlock) suffix[i] = cx.prefix[i];
    __syncthreads();
    return ar;
}

// ---------------------------------------------------------------------------
// Fase 4: um bloco por item da árvore raiz.

__global__ void __launch_bounds__(kBlock) mine_static_kernel(MineCtx cx) {
    extern __shared__ int32_t smem_hist[];
    __shared__ Shared sh;
    const int32_t root = cx.root_lo + blockIdx.x;

    // Reserva uma arena livre (no máximo n_slots blocos residentes ao mesmo tempo).
    if (threadIdx.x == 0) {
        int s = blockIdx.x % cx.n_slots;
        while (atomicCAS(&cx.slot_lock[s], 0, 1) != 0) s = (s + 1) % cx.n_slots;
        sh.i1 = s;
    }
    __syncthreads();
    const int slot = sh.i1;
    const long long t0 = clock64();

    Frame* frames;
    int32_t* suffix;
    Arena ar = open_arena(cx, slot, frames, suffix);
    BlockWork bw;
    int32_t n_frames = 0;
    expand(cx, cx.root, cx.root.fitems[root], cx.prefix_len, root, suffix, frames, n_frames, ar, smem_hist, false, bw,
           sh);
    run_stack(cx, root, suffix, frames, n_frames, ar, smem_hist, false, bw, sh);

    __syncthreads();
    if (threadIdx.x == 0) {
        long long dt = clock64() - t0;
        int b = blockIdx.x;
        cx.blk_cycles[b] = dt;
        cx.blk_busy[b] = dt;
        cx.blk_tasks[b] = bw.tasks;
        cx.blk_sets[b] = bw.sets;
        cx.blk_slot[b] = slot;
        __threadfence();
        atomicExch(&cx.slot_lock[slot], 0);
    }
}

// ---------------------------------------------------------------------------
// Fase 5: blocos persistentes consumindo a fila global.

__global__ void __launch_bounds__(kBlock) mine_dynamic_kernel(MineCtx cx) {
    extern __shared__ int32_t smem_hist[];
    __shared__ Shared sh;
    const int slot = blockIdx.x;
    const long long t0 = clock64();
    long long busy = 0;
    Frame* frames;
    int32_t* suffix;
    Arena ar = open_arena(cx, slot, frames, suffix);
    const size_t base_bottom = ar.bottom;
    BlockWork bw;
    while (true) {
        if (threadIdx.x == 0) {
            int state;
            if (try_dequeue(cx, sh.task)) state = 1;
            else if (*(volatile unsigned long long*)cx.pending == 0) state = 2;
            else state = 0;
            sh.i0 = state;
        }
        __syncthreads();
        const int state = sh.i0;
        const Task task = sh.task;
        __syncthreads();
        if (state == 2) break;
        if (state == 0) {
            __nanosleep(256);
            continue;
        }
        const long long tb = clock64();
        ar.bottom = base_bottom;
        int32_t n_frames = 0;
        if (task.kind == kRootItem) {
            expand(cx, cx.root, cx.root.fitems[task.root], cx.prefix_len, task.root, suffix, frames, n_frames, ar,
                   smem_hist, true, bw, sh);
        } else {
            for (int32_t i = threadIdx.x; i < task.depth; i += kBlock) suffix[i] = task.suffix[i];
            if (threadIdx.x == 0) frames[0] = Frame{task.t, 0, task.depth, ar.bottom};
            n_frames = 1;
            __syncthreads();
        }
        run_stack(cx, task.root, suffix, frames, n_frames, ar, smem_hist, true, bw, sh);
        busy += clock64() - tb;
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(cx.pending, ~0ull);  // 5.5: tarefa concluída
        }
    }
    if (threadIdx.x == 0) {
        cx.blk_cycles[slot] = clock64() - t0;
        cx.blk_busy[slot] = busy;
        cx.blk_tasks[slot] = bw.tasks;
        cx.blk_sets[slot] = bw.sets;
        cx.blk_slot[slot] = slot;
    }
}

// ---------------------------------------------------------------------------
// Projeção de uma tarefa estourada X a partir da base pré-processada (4.8):
// {t ∩ [0, min X) : t ⊇ X}, filtrada por frequência. Como as transações estão
// em ranks crescentes, os itens < min X formam um prefixo de cada transação.

__global__ void project_len_kernel(const int32_t* items, const int32_t* off, int32_t n, const int32_t* X, int32_t k,
                                   int32_t* plen, int32_t* hist) {
    int32_t lo = X[0];
    for (int32_t j = 1; j < k; ++j) lo = min(lo, X[j]);
    for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n; t += int64_t(gridDim.x) * blockDim.x) {
        const int32_t b = off[t], e = off[t + 1];
        bool all = true;
        for (int32_t j = 0; j < k && all; ++j) {  // busca binária de cada item de X
            int32_t a = b, z = e;
            while (a < z) {
                int32_t m = (a + z) >> 1;
                if (items[m] < X[j]) a = m + 1; else z = m;
            }
            all = a < e && items[a] == X[j];
        }
        int32_t len = 0;
        if (all)
            for (int32_t p = b; p < e && items[p] < lo; ++p) {
                atomicAdd(&hist[items[p]], 1);
                ++len;
            }
        plen[t] = len;
    }
}

__global__ void project_count_kernel(const int32_t* items, const int32_t* off, int32_t n, const int32_t* plen,
                                     const int32_t* hist, int32_t minsup, int32_t* flen) {
    for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n; t += int64_t(gridDim.x) * blockDim.x) {
        int32_t c = 0;
        for (int32_t p = off[t]; p < off[t] + plen[t]; ++p) c += hist[items[p]] >= minsup;
        flen[t] = c;
    }
}

__global__ void project_write_kernel(const int32_t* items, const int32_t* off, int32_t n, const int32_t* plen,
                                     const int32_t* hist, int32_t minsup, const int32_t* fpos, const int32_t* tpos,
                                     int32_t* out_items, int32_t* out_off) {
    for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n; t += int64_t(gridDim.x) * blockDim.x) {
        if (fpos[t + 1] == fpos[t]) continue;
        int32_t o = fpos[t];
        out_off[tpos[t]] = o;
        for (int32_t p = off[t]; p < off[t] + plen[t]; ++p)
            if (hist[items[p]] >= minsup) out_items[o++] = items[p];
    }
}

struct IsNonEmpty {
    __device__ int32_t operator()(int32_t x) const { return x > 0; }
};

DeviceCsr project(const DeviceCsr& db, const std::vector<int32_t>& X, int32_t minsup, int32_t& n_items) {
    n_items = *std::min_element(X.begin(), X.end());
    const int32_t n = db.n();
    thrust::device_vector<int32_t> dX(X.begin(), X.end()), plen(n), hist(std::max(n_items, 1), 0), flen(n + 1),
        tpos(n + 1);
    const int grid = std::max(1, std::min(div_up(n, 256), 1 << 16));
    project_len_kernel<<<grid, 256>>>(raw(db.items), raw(db.offsets), n, raw(dX), int32_t(X.size()), raw(plen),
                                      raw(hist));
    CUDA_CHECK_KERNEL();
    project_count_kernel<<<grid, 256>>>(raw(db.items), raw(db.offsets), n, raw(plen), raw(hist), minsup, raw(flen));
    CUDA_CHECK_KERNEL();
    thrust::transform(thrust::device, flen.begin(), flen.begin() + n, tpos.begin(), IsNonEmpty{});
    flen[n] = 0;
    tpos[n] = 0;
    thrust::exclusive_scan(thrust::device, flen.begin(), flen.end(), flen.begin());
    thrust::exclusive_scan(thrust::device, tpos.begin(), tpos.end(), tpos.begin());
    const int32_t total = flen[n], n_out = tpos[n];
    DeviceCsr out;
    out.items.resize(total);
    out.offsets.resize(n_out + 1);
    out.offsets[n_out] = total;
    project_write_kernel<<<grid, 256>>>(raw(db.items), raw(db.offsets), n, raw(plen), raw(hist), minsup, raw(flen),
                                        raw(tpos), raw(out.items), raw(out.offsets));
    CUDA_CHECK_KERNEL();
    return out;
}

// ---------------------------------------------------------------------------
// Host

// Árvore raiz de uma execução (a global ou a projetada de uma tarefa estourada).
struct RootJob {
    DeviceTree tree;
    std::vector<int32_t> prefix;
};

TreeView make_view(DeviceTree& t, thrust::device_vector<int32_t>& fitems, int32_t minsup) {
    std::vector<int32_t> sup(t.n_items), f;
    thrust::copy(t.item_support.begin(), t.item_support.end(), sup.begin());
    for (int32_t i = t.n_items - 1; i >= 0; --i)
        if (sup[i] >= minsup) f.push_back(i);
    fitems.assign(f.begin(), f.end());
    TreeView v;
    v.n_nodes = t.n_nodes;
    v.n_items = t.n_items;
    v.n_fitems = int32_t(f.size());
    v.item = raw(t.item);
    v.count = raw(t.count);
    v.parent = raw(t.parent);
    v.depth = raw(t.depth);
    v.hdr_off = raw(t.header_offsets);
    v.hdr_nodes = raw(t.header_nodes);
    v.support = raw(t.item_support);
    v.fitems = raw(fitems);
    return v;
}

template <typename T>
T* dalloc(thrust::device_vector<T>& v, size_t n) {
    v.assign(std::max<size_t>(n, 1), T(0));
    return raw(v);
}

struct BlockRecord {
    int launch, block, slot, tasks;
    long long cycles, busy, sets;
};

// Memória de dispositivo sem inicialização.
struct DevBuf {
    char* p = nullptr;
    void alloc(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(bytes, 1))); }
    ~DevBuf() {
        if (p) cudaFree(p);
    }
};

class Miner {
public:
    Miner(int32_t minsup, const MineOptions& opt, const DeviceCsr& pre_db, MineStats& stats, int32_t n_items)
        : minsup_(minsup), opt_(opt), pre_db_(pre_db), stats_(stats) {
        dynamic_ = opt.mode == MineMode::Dynamic;
        int dev = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz_, cudaDevAttrClockRate, dev));
        // Memória compartilhada dinâmica: histograma condicional e limite de
        // ocupação (2 blocos por SM), para que cada arena seja grande.
        smem_bytes_ = 40 * 1024;
        auto kern = dynamic_ ? mine_dynamic_kernel : mine_static_kernel;
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, kBlock, smem_bytes_));
        n_slots_ = std::max(1, per_sm) * sms;
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        // Máquina compartilhada: usamos no máximo ~35 % da memória livre para as
        // arenas e ~15 % para o pool da fila dinâmica.
        size_t arena_total = opt.arena_bytes ? opt.arena_bytes * n_slots_
                                             : std::min<size_t>(free_b * 35 / 100, size_t(6) << 30);
        arena_bytes_ = (arena_total / n_slots_) & ~size_t(255);
        // A arena precisa ao menos da pilha de quadros e do sufixo de profundidade
        // máxima (n_items + 1), mais uma folga mínima para trabalhar.
        const size_t stack_bytes = (size_t(n_items) + 2) * (sizeof(Frame) + sizeof(int32_t)) + 64;
        arena_bytes_ = std::max(arena_bytes_, (stack_bytes + (size_t(16) << 10) + 255) & ~size_t(255));
        pool_bytes_ = dynamic_ ? (opt.pool_bytes ? opt.pool_bytes : std::min<size_t>(free_b * 15 / 100, size_t(3) << 30))
                               : 0;
        // Sem inicialização: cada arena e o pool são sempre escritos antes de lidos.
        // (Zerar 6–9 GB com device_vector::resize custava ~10–20 ms por execução.)
        arenas_.alloc(arena_bytes_ * n_slots_);
        slot_lock_.assign(n_slots_, 0);
        if (dynamic_) pool_.alloc(pool_bytes_);
        stats_.n_blocks = n_slots_;
        stats_.peak_bytes = arena_bytes_ * n_slots_ + pool_bytes_;
    }

    ItemsetList run(DeviceTree& global, StageTimes& times) {
        ItemsetList out;
        std::vector<RootJob> jobs;
        jobs.push_back(RootJob{});
        bool first = true;
        while (!jobs.empty()) {
            RootJob job = std::move(jobs.back());
            jobs.pop_back();
            DeviceTree& tree = first ? global : job.tree;
            first = false;
            thrust::device_vector<int32_t> fitems;
            TreeView view = make_view(tree, fitems, minsup_);
            if (view.n_fitems == 0) continue;
            std::vector<std::vector<int32_t>> overflow;
            run_root(view, job.prefix, out, overflow, times);
            for (auto& X : overflow) {
                // Constrói a árvore projetada de X com a rotina de dispositivo inteiro.
                GpuTimer t;
                int32_t n_items = 0;
                DeviceCsr proj = project(pre_db_, X, minsup_, n_items);
                StageTimes dummy("", "", 0);
                RootJob sub;
                sub.prefix = X;
                if (proj.n() > 0) {
                    sub.tree = build_tree(proj, {}, n_items, BuildOptions{}, dummy);
                    jobs.push_back(std::move(sub));
                }
                times.add("mineracao", t.ms());
            }
        }
        if (!opt_.block_stats_path.empty()) write_block_stats();
        stats_.max_over_mean_block_time = imbalance();
        return out;
    }

private:
    MineCtx base_ctx(const TreeView& view, const thrust::device_vector<int32_t>& d_prefix, int32_t prefix_len) {
        MineCtx cx{};
        cx.root = view;
        cx.prefix = raw(d_prefix);
        cx.prefix_len = prefix_len;
        cx.max_depth = prefix_len + view.n_items + 1;
        cx.minsup = minsup_;
        cx.grain = opt_.grain;
        cx.arenas = arenas_.p;
        cx.arena_bytes = arena_bytes_;
        cx.slot_lock = raw(slot_lock_);
        cx.n_slots = n_slots_;
        cx.hist_cap = int(smem_bytes_ / sizeof(int32_t));
        cx.pool = dynamic_ ? pool_.p : nullptr;
        cx.pool_bytes = pool_bytes_;
        cx.error_flag = dalloc(error_, 1);
        cx.max_ovf = 1 << 16;
        return cx;
    }

    // Lança um kernel sobre as raízes [lo, hi) e devolve o tempo em ms.
    float launch(MineCtx& cx, int32_t lo, int32_t hi) {
        cx.root_lo = lo;
        cx.root_hi = hi;
        const int32_t n = hi - lo;
        int n_blocks = dynamic_ ? n_slots_ : n;
        cx.blk_cycles = dalloc(blk_cycles_, n_blocks);
        cx.blk_busy = dalloc(blk_busy_, n_blocks);
        cx.blk_tasks = dalloc(blk_tasks_, n_blocks);
        cx.blk_sets = dalloc(blk_sets_, n_blocks);
        cx.blk_slot = dalloc(blk_slot_, n_blocks);
        if (dynamic_) {
            // Fila inicial: uma tarefa por item raiz.
            unsigned long long cap = std::max<unsigned long long>(opt_.queue_capacity, 2ull * n);
            std::vector<Task> init(n);
            std::vector<unsigned long long> seq(cap, 0);
            for (int32_t i = 0; i < n; ++i) {
                init[i] = Task{TreeView{}, nullptr, cx.prefix_len, lo + i, kRootItem};
                seq[i] = i + 1;
            }
            q_task_.resize(cap * sizeof(Task));
            CUDA_CHECK(cudaMemcpy(raw(q_task_), init.data(), n * sizeof(Task), cudaMemcpyHostToDevice));
            q_seq_.assign(seq.begin(), seq.end());
            std::vector<unsigned long long> ctl = {0, (unsigned long long)n, (unsigned long long)n, 0, 0};
            q_ctl_.assign(ctl.begin(), ctl.end());  // head, tail, pending, pool_used, enqueued
            cx.q_task = reinterpret_cast<Task*>(raw(q_task_));
            cx.q_seq = raw(q_seq_);
            cx.q_cap = cap;
            cx.q_head = raw(q_ctl_);
            cx.q_tail = raw(q_ctl_) + 1;
            cx.pending = raw(q_ctl_) + 2;
            cx.pool_used = raw(q_ctl_) + 3;
            cx.n_enqueued = raw(q_ctl_) + 4;
        }
        GpuTimer t;
        if (dynamic_) mine_dynamic_kernel<<<n_slots_, kBlock, smem_bytes_>>>(cx);
        else mine_static_kernel<<<n, kBlock, smem_bytes_>>>(cx);
        CUDA_CHECK_KERNEL();
        float ms = t.ms();
        MemTracker::sample();
        int err = error_[0];
        if (err) throw std::runtime_error(err == 1 ? "caminho único longo demais (> 40 itens)"
                                                   : "lista de estouro cheia");
        if (dynamic_ && cx.write) {  // contamos só a passada de escrita
            unsigned long long enq = q_ctl_[4];
            stats_.n_enqueued += (long long)enq;
        }
        return ms;
    }

    void record_blocks(int n_blocks) {
        std::vector<long long> cyc(n_blocks), busy(n_blocks), sets(n_blocks);
        std::vector<int> tasks(n_blocks), slot(n_blocks);
        thrust::copy(blk_cycles_.begin(), blk_cycles_.begin() + n_blocks, cyc.begin());
        thrust::copy(blk_busy_.begin(), blk_busy_.begin() + n_blocks, busy.begin());
        thrust::copy(blk_sets_.begin(), blk_sets_.begin() + n_blocks, sets.begin());
        thrust::copy(blk_tasks_.begin(), blk_tasks_.begin() + n_blocks, tasks.begin());
        thrust::copy(blk_slot_.begin(), blk_slot_.begin() + n_blocks, slot.begin());
        for (int b = 0; b < n_blocks; ++b) {
            blocks_.push_back(BlockRecord{launch_id_, b, slot[b], tasks[b], cyc[b], busy[b], sets[b]});
            stats_.n_tasks += tasks[b];
        }
        ++launch_id_;
    }

    void run_root(const TreeView& view, const std::vector<int32_t>& prefix, ItemsetList& out,
                  std::vector<std::vector<int32_t>>& overflow, StageTimes& times) {
        thrust::device_vector<int32_t> d_prefix(prefix.begin(), prefix.end());
        if (d_prefix.empty()) d_prefix.resize(1);
        MineCtx cx = base_ctx(view, d_prefix, int32_t(prefix.size()));
        const int32_t n_roots = view.n_fitems;

        // 4.7 Passada 1: só contagem, por raiz.
        cx.write = 0;
        cx.root_sets = dalloc(root_sets_, n_roots);
        cx.root_items = dalloc(root_items_, n_roots);
        const float p1 = launch(cx, 0, n_roots);
        times.add("mineracao_passada1", p1);
        times.add("mineracao", p1);
        std::vector<unsigned long long> rs(n_roots), ri(n_roots);
        thrust::copy(root_sets_.begin(), root_sets_.begin() + n_roots, rs.begin());
        thrust::copy(root_items_.begin(), root_items_.begin() + n_roots, ri.begin());

        // Passada 2 em lotes de raízes cuja saída cabe no orçamento.
        const unsigned long long budget_items = 256ull << 20, budget_sets = 64ull << 20;
        int32_t lo = 0;
        while (lo < n_roots) {
            int32_t hi = lo;
            unsigned long long s = 0, it = 0;
            while (hi < n_roots && (hi == lo || (s + rs[hi] <= budget_sets && it + ri[hi] <= budget_items))) {
                s += rs[hi];
                it += ri[hi];
                ++hi;
            }
            run_batch(cx, lo, hi, s, it, out, overflow, times);
            lo = hi;
        }
    }

    void run_batch(MineCtx cx, int32_t lo, int32_t hi, unsigned long long sets, unsigned long long items,
                   ItemsetList& out, std::vector<std::vector<int32_t>>& overflow, StageTimes& times) {
        cx.write = 1;
        // A dinâmica pode estourar menos tarefas na passada 2 que na 1 (o destino
        // de cada filho depende do estado da fila); por isso há folga e nova
        // tentativa se o buffer encher.
        unsigned long long cap_s = sets, cap_i = items;
        if (dynamic_) {
            cap_s += cap_s / 8 + 1024;
            cap_i += cap_i / 8 + 8192;
        }
        while (true) {
            cx.cap_sets = cap_s;
            cx.cap_items = cap_i;
            cx.cnt_sets = dalloc(ctr_, 2);
            cx.cnt_items = cx.cnt_sets + 1;
            cx.out_items = dalloc(out_items_, cap_i);
            cx.out_start = dalloc(out_start_, cap_s);
            cx.out_len = dalloc(out_len_, cap_s);
            cx.out_sup = dalloc(out_sup_, cap_s);
            cx.out_overflow = dalloc(out_ovf_, 1);
            cx.ovf_count = dalloc(ovf_count_, 1);
            cx.ovf_suffix = dalloc(ovf_suffix_, size_t(cx.max_ovf) * cx.max_depth);
            cx.ovf_len = dalloc(ovf_len_, cx.max_ovf);
            const float p2 = launch(cx, lo, hi);
            times.add("mineracao_passada2", p2);
            times.add("mineracao", p2);
            if (out_ovf_[0] == 0) break;
            cap_s *= 2;
            cap_i *= 2;
        }
        record_blocks(dynamic_ ? n_slots_ : hi - lo);

        // Transferência D2H da saída.
        GpuTimer t;
        unsigned long long ns = ctr_[0], ni = ctr_[1];
        std::vector<int32_t> h_items(ni), len(ns), sup(ns);
        std::vector<int64_t> start(ns);
        if (ni) CUDA_CHECK(cudaMemcpy(h_items.data(), raw(out_items_), ni * 4, cudaMemcpyDeviceToHost));
        if (ns) {
            CUDA_CHECK(cudaMemcpy(start.data(), raw(out_start_), ns * 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(len.data(), raw(out_len_), ns * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(sup.data(), raw(out_sup_), ns * 4, cudaMemcpyDeviceToHost));
        }
        int n_ovf = ovf_count_[0];
        std::vector<int32_t> osuf(size_t(n_ovf) * cx.max_depth), olen(n_ovf);
        if (n_ovf) {
            CUDA_CHECK(cudaMemcpy(osuf.data(), raw(ovf_suffix_), osuf.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(olen.data(), raw(ovf_len_), n_ovf * 4, cudaMemcpyDeviceToHost));
        }
        times.add("transferencia_d2h", t.ms());
        stats_.n_itemsets_gpu += (long long)ns;
        stats_.n_overflow_tasks += n_ovf;
        for (int k = 0; k < n_ovf; ++k)
            overflow.emplace_back(osuf.begin() + int64_t(k) * cx.max_depth,
                                  osuf.begin() + int64_t(k) * cx.max_depth + olen[k]);
        // Monta o CSR de itemsets no host (a ordenação canônica vem depois).
        CpuTimer ct;
        out.items.reserve(out.items.size() + ni);
        for (unsigned long long s = 0; s < ns; ++s) out.add(h_items.data() + start[s], len[s], sup[s]);
        times.add("transferencia_d2h", ct.ms());
    }

    void write_block_stats() {
        FILE* f = std::fopen(opt_.block_stats_path.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "mode,launch,block,slot,tasks,itemsets,cycles,busy_cycles,ms,busy_ms\n");
        for (auto& b : blocks_)
            std::fprintf(f, "%s,%d,%d,%d,%d,%lld,%lld,%lld,%.6f,%.6f\n", dynamic_ ? "dynamic" : "static", b.launch,
                         b.block, b.slot, b.tasks, b.sets, b.cycles, b.busy, b.cycles / double(clock_khz_),
                         b.busy / double(clock_khz_));
        std::fclose(f);
    }

    // Razão entre o tempo ocupado do bloco mais lento e o tempo médio (só a
    // primeira execução, que minera a árvore global).
    double imbalance() const {
        double mx = 0, sum = 0;
        int n = 0;
        for (auto& b : blocks_)
            if (b.launch == 0) {
                mx = std::max(mx, double(b.busy));
                sum += b.busy;
                ++n;
            }
        return n && sum > 0 ? mx / (sum / n) : 0;
    }

    int32_t minsup_;
    MineOptions opt_;
    const DeviceCsr& pre_db_;
    MineStats& stats_;
    bool dynamic_ = false;
    int clock_khz_ = 0;
    size_t smem_bytes_ = 0, arena_bytes_ = 0, pool_bytes_ = 0;
    int n_slots_ = 0;
    DevBuf arenas_, pool_;
    thrust::device_vector<char> q_task_;
    thrust::device_vector<int> slot_lock_, error_, blk_tasks_, blk_slot_, out_ovf_, ovf_count_;
    thrust::device_vector<long long> blk_cycles_, blk_busy_, blk_sets_;
    thrust::device_vector<unsigned long long> root_sets_, root_items_, ctr_, q_seq_, q_ctl_;
    thrust::device_vector<int32_t> out_items_, out_len_, out_sup_, ovf_suffix_, ovf_len_;
    thrust::device_vector<int64_t> out_start_;
    std::vector<BlockRecord> blocks_;
    int launch_id_ = 0;
};

}  // namespace

ItemsetList mine(DeviceTree& tree, int32_t minsup, const MineOptions& opt, const DeviceCsr& pre_db, StageTimes& times,
                 MineStats& stats) {
    // A preparação (reserva de arenas, pool e travas) e a liberação no fim
    // também contam como mineração, na subetapa "mineracao_preparacao".
    GpuTimer prep;
    auto m = std::make_unique<Miner>(minsup, opt, pre_db, stats, tree.n_items);
    float setup = prep.ms();
    ItemsetList out = m->run(tree, times);
    prep.start();
    m.reset();
    setup += prep.ms();
    times.add("mineracao_preparacao", setup);
    times.add("mineracao", setup);
    return out;
}

}  // namespace gpu
}  // namespace fpg
