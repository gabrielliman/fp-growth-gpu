#pragma once
// Baselines de mineração em GPU com representação vertical em bitset, usadas
// para comparação com o FP-Growth em GPU:
//
// - mine_apriori_gpu: no estilo do GPApriori (Zhang, Zhang e Bakos, CLUSTER 2011).
//   Candidatos gerados nível a nível na CPU (junção por classes de equivalência
//   + poda de Apriori); suporte por "interseção completa" na GPU: só os bitsets
//   dos itens frequentes ficam na GPU e cada candidato de tamanho k faz o AND
//   dos k bitsets. Um bloco por candidato, uma palavra por thread, popcount e
//   redução em memória compartilhada; itens do candidato pré-carregados em shared.
//
// - mine_eclat_gpu: no estilo do Frontier Expansion (Zhang, Zhang e Bakos,
//   J. Supercomputing 2013). Pilha de classes de equivalência na CPU; a cada
//   passo desempilha classes até juntar ε candidatos (ε = blocos residentes na
//   GPU); cada candidato é o AND de dois bitsets já na GPU e o resultado fica
//   num pool de listas verticais gerenciado pelo host. Itens ordenados por
//   frequência crescente.
//
// Ambas recebem a base pré-processada (ranks, ordem por frequência decrescente)
// e devolvem itemsets em ranks.

#include <cstdint>

#include "common/fimi.hpp"
#include "common/output.hpp"
#include "common/timer.hpp"

namespace fpg {
namespace baselines {

struct VerticalStats {
    int64_t candidates = 0;   // candidatos cujo suporte foi contado na GPU
    int64_t kernel_launches = 0;
    int levels = 0;           // Apriori: número de níveis
    size_t gpu_bytes = 0;     // memória reservada na GPU (bitsets + pool)
};

// Registram "preprocessamento" (conversão para bitsets verticais),
// "transferencia_h2d", "mineracao" e "transferencia_d2h".
ItemsetList mine_apriori_gpu(const Csr& pre, int32_t n_items, const std::vector<int32_t>& rank_support,
                             int32_t minsup, StageTimes& times, VerticalStats& stats);

// pool_slots: número de listas verticais do pool (0 = automático, até 35 % da
// memória livre e no máximo 8 GB). Com n itens, n(n+1)/2 + n + 2 posições
// garantem progresso no pior caso da busca em profundidade.
ItemsetList mine_eclat_gpu(const Csr& pre, int32_t n_items, const std::vector<int32_t>& rank_support,
                           int32_t minsup, StageTimes& times, VerticalStats& stats, int64_t pool_slots = 0);

}  // namespace baselines
}  // namespace fpg
