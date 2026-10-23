#pragma once
// Estruturas e interfaces do FP-Growth em GPU.

#include <cstdint>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "common/cuda_check.cuh"
#include "common/fimi.hpp"
#include "common/output.hpp"
#include "common/timer.hpp"

namespace fpg {
namespace gpu {

template <typename T>
T* raw(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}
template <typename T>
const T* raw(const thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

inline int div_up(int64_t a, int64_t b) { return static_cast<int>((a + b - 1) / b); }

// Cronômetro com cudaEvent na stream padrão.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&a_));
        CUDA_CHECK(cudaEventCreate(&b_));
        start();
    }
    ~GpuTimer() {
        cudaEventDestroy(a_);
        cudaEventDestroy(b_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(a_)); }
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

// Pico de memória da GPU usada pelo processo, estimado pela menor memória livre
// observada em relação à memória livre no início. Numa GPU compartilhada,
// alocações de outros processos no mesmo intervalo entram como ruído.
struct MemTracker {
    static size_t& baseline() { static size_t b = 0; return b; }
    static size_t& min_free() { static size_t m = SIZE_MAX; return m; }
    static void init() {
        size_t f = 0, t = 0;
        CUDA_CHECK(cudaMemGetInfo(&f, &t));
        baseline() = f;
        min_free() = f;
    }
    static void sample() {
        size_t f = 0, t = 0;
        CUDA_CHECK(cudaMemGetInfo(&f, &t));
        if (f < min_free()) min_free() = f;
    }
    static size_t peak() { return baseline() > min_free() ? baseline() - min_free() : 0; }
};

// Base em CSR na GPU.
struct DeviceCsr {
    thrust::device_vector<int32_t> items;
    thrust::device_vector<int32_t> offsets;
    int32_t n() const { return static_cast<int32_t>(offsets.size()) - 1; }
};

// ---------------------------------------------------------------------------
// Fase 2: contagem e pré-processamento

enum class CountKernel { V1, V2 };

struct Preprocessed {
    DeviceCsr db;                       // transações filtradas, em ranks, ordenadas
    thrust::device_vector<int32_t> rank_support;
    std::vector<int32_t> rank_to_id;    // cópia no host, para a saída
    std::vector<int32_t> h_support;     // suporte por id (cópia no host, para depuração)
    int32_t n_frequent = 0;
};

// Copia a base para a GPU (2.1). Registra "transferencia_h2d".
DeviceCsr upload(const Csr& db, StageTimes& times);

// Contagem de suporte por id (2.2 / 2.3). Retorna o vetor de suportes na GPU.
thrust::device_vector<int32_t> count_support(const DeviceCsr& db, int32_t max_id, CountKernel kernel);

// Pré-processamento completo (2.2–2.7). Registra "contagem" e "preprocessamento".
Preprocessed preprocess(const DeviceCsr& db, int32_t max_id, int32_t minsup, CountKernel kernel,
                        StageTimes& times);

Csr download(const DeviceCsr& db);

// ---------------------------------------------------------------------------
// Fase 3: FP-tree linearizada

struct DeviceTree {
    thrust::device_vector<int32_t> item, count, parent, depth;
    thrust::device_vector<int32_t> header_nodes;    // ids de nó agrupados por item
    thrust::device_vector<int32_t> header_offsets;  // n_items + 1
    thrust::device_vector<int32_t> item_support;    // suporte total por item
    int32_t n_nodes = 0;
    int32_t n_items = 0;
};

struct BuildOptions {
    bool lsd_sort = false;      // ordenação lexicográfica por colunas (LSD) em vez de comparador
    bool cross_check = false;   // 3.7: confere contadores com reduce_by_key
    bool count_only_sort = false;
};

struct BuildStats {
    size_t peak_bytes = 0;      // memória temporária máxima estimada da construção
    bool counts_match = true;   // resultado da verificação cruzada 3.7
};

// Constrói a FP-tree da base pré-processada. weights pode ser vazio (peso 1).
// Registra as subetapas "construcao_sort", "construcao_lcp", "construcao_scan",
// "construcao_nos", "construcao_pais", "construcao_contadores",
// "construcao_cabecalho" e o total "construcao".
DeviceTree build_tree(DeviceCsr& db, const thrust::device_vector<int32_t>& weights, int32_t n_items,
                      const BuildOptions& opt, StageTimes& times, BuildStats* stats = nullptr);

// Copia os vetores da árvore para o host (depuração / validação).
void download_tree(const DeviceTree& t, std::vector<int32_t>& item, std::vector<int32_t>& count,
                   std::vector<int32_t>& parent);

// ---------------------------------------------------------------------------
// Fases 4 e 5: mineração

enum class MineMode { Static, Dynamic };

struct MineOptions {
    MineMode mode = MineMode::Static;
    int grain = 64;                   // limiar de granularidade (itens da base) da versão dinâmica
    size_t arena_bytes = 0;           // bytes por arena de bloco (0 = automático)
    size_t pool_bytes = 0;            // bytes do pool global de tarefas (0 = automático)
    int queue_capacity = 1 << 16;
    std::string block_stats_path;     // CSV com instrumentação por bloco
};

struct MineStats {
    int64_t n_itemsets_gpu = 0;
    int64_t n_overflow_tasks = 0;
    int64_t n_tasks = 0;
    int64_t n_enqueued = 0;
    int n_blocks = 0;
    size_t peak_bytes = 0;
    double max_over_mean_block_time = 0;
};

// Minera todos os itemsets frequentes. Itens da saída em ranks.
// pre_db é a base pré-processada na GPU, usada para projetar as tarefas que
// estouram a arena de um bloco. Registra "mineracao" (com as subetapas
// "mineracao_passada1" e "mineracao_passada2") e "transferencia_d2h".
ItemsetList mine(DeviceTree& tree, int32_t minsup, const MineOptions& opt, const DeviceCsr& pre_db,
                 StageTimes& times, MineStats& stats);

}  // namespace gpu
}  // namespace fpg
