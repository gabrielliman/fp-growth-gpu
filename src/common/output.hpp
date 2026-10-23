#pragma once
// Representação de itemsets minerados e escrita no formato canônico.
//
// Formato canônico: uma linha por itemset, itens em ordem crescente de id
// original separados por espaço, seguidos de " (suporte)". As linhas são
// ordenadas lexicograficamente (ordem de bytes, igual a `LC_ALL=C sort`).

#include <cstdint>
#include <string>
#include <vector>

namespace fpg {

// Itemsets em CSR. Os itens podem estar como ranks ou ids originais, conforme
// quem produziu; to_canonical_lines recebe a tabela rank -> id quando preciso.
struct ItemsetList {
    std::vector<int32_t> items;
    std::vector<int64_t> offsets{0};
    std::vector<int32_t> support;

    size_t size() const { return support.size(); }
    void add(const int32_t* it, int len, int32_t sup) {
        items.insert(items.end(), it, it + len);
        offsets.push_back(static_cast<int64_t>(items.size()));
        support.push_back(sup);
    }
    void append(const ItemsetList& o) {
        int64_t base = static_cast<int64_t>(items.size());
        items.insert(items.end(), o.items.begin(), o.items.end());
        for (size_t i = 1; i < o.offsets.size(); ++i) offsets.push_back(base + o.offsets[i]);
        support.insert(support.end(), o.support.begin(), o.support.end());
    }
};

// Converte para linhas canônicas ordenadas. Se rank_to_id não for vazio, os
// itens são ranks e são traduzidos para ids originais.
std::vector<std::string> to_canonical_lines(const ItemsetList& sets, const std::vector<int32_t>& rank_to_id);

void write_lines(const std::string& path, const std::vector<std::string>& lines);

}  // namespace fpg
