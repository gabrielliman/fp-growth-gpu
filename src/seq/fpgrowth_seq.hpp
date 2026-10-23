#pragma once
// FP-Growth sequencial clássico (baseline e oráculo de corretude).

#include <cstdint>
#include <string>
#include <vector>

#include "common/fimi.hpp"
#include "common/output.hpp"

namespace fpg {

// Tabela de ranks: itens frequentes ordenados por suporte decrescente,
// desempate pelo id original crescente. Rank 0 = item mais frequente.
struct RankTable {
    std::vector<int32_t> rank_to_id;
    std::vector<int32_t> rank_support;
    std::vector<int32_t> id_to_rank;  // -1 para itens infrequentes
    int32_t n_frequent() const { return static_cast<int32_t>(rank_to_id.size()); }
};

// 1.1 Contagem de suporte por id de item (vetor de tamanho max_id + 1).
std::vector<int32_t> count_support(const Csr& db);

// 1.2 Remapeamento por suporte decrescente.
RankTable build_ranks(const std::vector<int32_t>& support, int32_t minsup);

// 1.3 Filtra itens infrequentes, troca id por rank e ordena cada transação por
// rank crescente. Transações que ficam vazias são descartadas.
Csr preprocess(const Csr& db, const RankTable& ranks);

// FP-tree clássica com nós em vetor (índices no lugar de ponteiros). O nó 0 é
// a raiz. Filhos ficam numa lista de irmãos; a busca de filho usa uma tabela
// hash (pai, item) para não varrer listas longas.
class FPTree {
public:
    explicit FPTree(int32_t n_items);

    // Insere um caminho de ranks crescentes com peso w.
    void insert(const int32_t* items, int len, int32_t w);

    int32_t n_items() const { return n_items_; }
    int32_t n_nodes() const { return static_cast<int32_t>(item_.size()) - 1; }  // sem a raiz
    bool single_path() const { return single_path_; }

    // Campos dos nós (índice 0 = raiz).
    std::vector<int32_t> item_, count_, parent_, first_child_, next_sibling_, next_same_;
    // Tabela de cabeçalho: primeiro nó de cada item e suporte total.
    std::vector<int32_t> head_, support_;

private:
    int32_t find_or_add_child(int32_t parent, int32_t item);
    void rehash();

    int32_t n_items_;
    bool single_path_ = true;
    std::vector<int64_t> hkeys_;
    std::vector<int32_t> hvals_;
    size_t hmask_ = 0, hused_ = 0;
};

// 1.4 Construção da FP-tree global a partir do CSR pré-processado.
FPTree build_tree(const Csr& pre, int32_t n_items);

// 1.5 Mineração. Itens dos itemsets retornados estão em ranks.
ItemsetList mine(const FPTree& tree, int32_t minsup);

// Minera a partir de um sufixo X (ranks): todos os itemsets X ∪ Z, Z não vazio,
// com Z formado por itens de rank menor que min(X). Usa a base projetada
// {t ∩ [0, min X) : t ⊇ X}. Usado para processar tarefas que estouram a
// memória da GPU.
ItemsetList mine_extensions(const Csr& pre, const std::vector<int32_t>& suffix, int32_t minsup);

// 1.9 FP-tree em formato textual canônico: um nó por linha, "r0 r1 ... rk (contador)",
// caminho de ranks da raiz até o nó, linhas ordenadas.
std::vector<std::string> dump_tree(const FPTree& tree);

// Caminhos (item, contador, pai) linearizados -> mesmo formato de dump_tree.
std::vector<std::string> dump_linear_tree(const std::vector<int32_t>& item, const std::vector<int32_t>& count,
                                          const std::vector<int32_t>& parent);

// Compara dois CSR (usado para validar o pré-processamento da GPU).
bool csr_equal(const Csr& a, const Csr& b);

}  // namespace fpg
