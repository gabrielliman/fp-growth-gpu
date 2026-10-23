#pragma once
// Leitura de arquivos FIMI para o formato CSR.

#include <cstdint>
#include <string>
#include <vector>

namespace fpg {

// Base de transações em CSR: os itens da transação t estão em
// items[offsets[t] .. offsets[t+1]).
struct Csr {
    std::vector<int32_t> items;
    std::vector<int32_t> offsets{0};

    int32_t n_transactions() const { return static_cast<int32_t>(offsets.size()) - 1; }
    int32_t len(int32_t t) const { return offsets[t + 1] - offsets[t]; }
};

struct DatasetStats {
    int64_t n_transactions = 0;
    int64_t n_distinct_items = 0;
    int64_t total_items = 0;
    int64_t max_len = 0;
    int32_t max_item_id = -1;
    double avg_len() const { return n_transactions ? double(total_items) / n_transactions : 0.0; }
};

// Lê um arquivo FIMI (uma transação por linha, inteiros não negativos separados
// por espaço). Itens repetidos dentro de uma transação são removidos.
// Linhas vazias viram transações vazias (contam no total de transações).
Csr read_fimi(const std::string& path);

DatasetStats compute_stats(const Csr& db);

// Converte o suporte relativo em contagem absoluta: ceil(minsup * n).
// O epsilon evita que erros de arredondamento (ex.: 0.01 * 100000) subam uma unidade.
int32_t minsup_count(double minsup, int64_t n_transactions);

// Nome do dataset a partir do caminho (basename sem extensão).
std::string dataset_name(const std::string& path);

}  // namespace fpg
