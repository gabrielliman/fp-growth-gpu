#pragma once
// Funções de dispositivo da construção da FP-tree por primitivas (Fase 3).
// São usadas pelos kernels de grade inteira (build_tree.cu) e pela versão de
// bloco da mineração (mine.cu). Todas recebem um par (tid, nthreads) para que o
// mesmo laço sirva a uma grade inteira ou a um único bloco.
//
// Notação: n caminhos (transações) já ordenados lexicograficamente, o caminho i
// ocupa items[off[i] .. off[i+1]). LCP[i] = maior prefixo comum entre i-1 e i.

#include <cstdint>

namespace fpg {
namespace gpu {

// Comparação lexicográfica de dois caminhos (prefixo menor vem antes).
__device__ __forceinline__ bool path_less(const int32_t* items, const int32_t* off, int32_t a, int32_t b) {
    int32_t pa = off[a], ea = off[a + 1], pb = off[b], eb = off[b + 1];
    for (; pa < ea && pb < eb; ++pa, ++pb)
        if (items[pa] != items[pb]) return items[pa] < items[pb];
    return (ea - pa) < (eb - pb);
}

// LCP entre os caminhos i-1 e i (0 para i = 0).
__device__ __forceinline__ int32_t path_lcp(const int32_t* items, const int32_t* off, int32_t i) {
    if (i == 0) return 0;
    int32_t pa = off[i - 1], ea = off[i], pb = off[i], eb = off[i + 1];
    int32_t l = 0;
    while (pa + l < ea && pb + l < eb && items[pa + l] == items[pb + l]) ++l;
    return l;
}

// Sparse table de mínimos: nível j em st[j * n + i] = min(LCP[i .. i + 2^j)).
// O nível 0 é o próprio LCP. Calcula o nível j (j >= 1) a partir do j-1.
__device__ __forceinline__ void sparse_level(int32_t* st, int32_t n, int j, int64_t tid, int64_t nthreads) {
    const int32_t half = 1 << (j - 1);
    const int32_t* prev = st + int64_t(j - 1) * n;
    int32_t* cur = st + int64_t(j) * n;
    for (int64_t i = tid; i + (int64_t(1) << j) <= n; i += nthreads) cur[i] = min(prev[i], prev[i + half]);
}

// Previous smaller value: maior k < i com LCP[k] < v. Pressupõe LCP[0] = 0 < v.
__device__ __forceinline__ int32_t prev_smaller(const int32_t* st, int32_t n, int levels, int32_t i, int32_t v) {
    int32_t pos = i;  // invariante: LCP[pos .. i) >= v
    for (int j = levels - 1; j >= 0; --j) {
        int32_t w = 1 << j;
        if (pos >= w && st[int64_t(j) * n + pos - w] >= v) pos -= w;
    }
    return pos - 1;
}

// Next smaller or equal: menor m > k com LCP[m] <= d, ou n se não existir.
__device__ __forceinline__ int32_t next_smaller_eq(const int32_t* st, int32_t n, int levels, int32_t k, int32_t d) {
    int32_t pos = k + 1;  // invariante: LCP(k .. pos) > d
    for (int j = levels - 1; j >= 0; --j) {
        int32_t w = 1 << j;
        if (pos + w <= n && st[int64_t(j) * n + pos] > d) pos += w;
    }
    return pos;
}

// Maior k <= i com LCP[k] <= d (quem criou o nó que (i, d) percorre).
__device__ __forceinline__ int32_t prev_smaller_eq(const int32_t* st, int32_t n, int levels, int32_t i, int32_t d) {
    int32_t pos = i + 1;  // invariante: LCP[pos .. i] > d
    for (int j = levels - 1; j >= 0; --j) {
        int32_t w = 1 << j;
        if (pos >= w && st[int64_t(j) * n + pos - w] > d) pos -= w;
    }
    return pos - 1;
}

__host__ __device__ __forceinline__ int sparse_levels(int32_t n) {
    int l = 1;
    while ((int64_t(1) << l) <= n) ++l;
    return l;
}

// 3.4: para cada caminho i, escreve item, profundidade e caminho criador dos nós
// novos (d >= LCP[i]).
__device__ __forceinline__ void write_nodes(const int32_t* items, const int32_t* off, const int32_t* lcp,
                                            const int32_t* base, int32_t n, int32_t* node_item, int32_t* node_depth,
                                            int32_t* node_tx, int64_t tid, int64_t nthreads) {
    for (int64_t i = tid; i < n; i += nthreads) {
        int32_t b = base[i], l = lcp[i], len = off[i + 1] - off[i];
        for (int32_t d = l; d < len; ++d) {
            node_item[b + d - l] = items[off[i] + d];
            node_depth[b + d - l] = d;
            node_tx[b + d - l] = int32_t(i);
        }
    }
}

// 3.5: pai de cada nó.
__device__ __forceinline__ void node_parents(const int32_t* st, int levels, const int32_t* lcp, const int32_t* base,
                                             int32_t n, int32_t n_nodes, const int32_t* node_depth,
                                             const int32_t* node_tx, int32_t* node_parent, int64_t tid,
                                             int64_t nthreads) {
    for (int64_t v = tid; v < n_nodes; v += nthreads) {
        int32_t i = node_tx[v], d = node_depth[v], l = lcp[i];
        int32_t p;
        if (d > l) {
            p = int32_t(v) - 1;
        } else if (l == 0) {
            p = -1;
        } else {
            int32_t k = prev_smaller(st, n, levels, i, l);
            p = base[k] + (l - 1 - lcp[k]);
        }
        node_parent[v] = p;
    }
}

// 3.6: contador de cada nó sem atômicos. W é o scan exclusivo dos pesos
// (tamanho n + 1); se for nulo, todos os pesos valem 1.
__device__ __forceinline__ void node_counts(const int32_t* st, int levels, const int32_t* W, int32_t n,
                                            int32_t n_nodes, const int32_t* node_depth, const int32_t* node_tx,
                                            int32_t* node_count, int64_t tid, int64_t nthreads) {
    for (int64_t v = tid; v < n_nodes; v += nthreads) {
        int32_t i = node_tx[v];
        int32_t m = next_smaller_eq(st, n, levels, i, node_depth[v]);
        node_count[v] = W ? W[m] - W[i] : m - i;
    }
}

}  // namespace gpu
}  // namespace fpg
