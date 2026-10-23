#include "seq/fpgrowth_seq.hpp"

#include <algorithm>
#include <cstdio>

namespace fpg {

std::vector<int32_t> count_support(const Csr& db) {
    int32_t max_id = -1;
    for (int32_t x : db.items) max_id = std::max(max_id, x);
    std::vector<int32_t> sup(size_t(max_id + 1), 0);
    for (int32_t x : db.items) ++sup[x];
    return sup;
}

RankTable build_ranks(const std::vector<int32_t>& support, int32_t minsup) {
    RankTable r;
    for (int32_t id = 0; id < int32_t(support.size()); ++id)
        if (support[id] >= minsup) r.rank_to_id.push_back(id);
    std::sort(r.rank_to_id.begin(), r.rank_to_id.end(), [&](int32_t a, int32_t b) {
        return support[a] != support[b] ? support[a] > support[b] : a < b;
    });
    r.id_to_rank.assign(support.size(), -1);
    for (int32_t k = 0; k < r.n_frequent(); ++k) {
        r.id_to_rank[r.rank_to_id[k]] = k;
        r.rank_support.push_back(support[r.rank_to_id[k]]);
    }
    return r;
}

Csr preprocess(const Csr& db, const RankTable& ranks) {
    Csr out;
    out.items.reserve(db.items.size());
    for (int32_t t = 0; t < db.n_transactions(); ++t) {
        size_t b = out.items.size();
        for (int32_t p = db.offsets[t]; p < db.offsets[t + 1]; ++p) {
            int32_t id = db.items[p];
            int32_t r = id < int32_t(ranks.id_to_rank.size()) ? ranks.id_to_rank[id] : -1;
            if (r >= 0) out.items.push_back(r);
        }
        if (out.items.size() == b) continue;
        std::sort(out.items.begin() + b, out.items.end());
        out.offsets.push_back(static_cast<int32_t>(out.items.size()));
    }
    return out;
}

// ---------------------------------------------------------------------------
// FPTree

FPTree::FPTree(int32_t n_items) : n_items_(n_items) {
    item_ = {-1};
    count_ = {0};
    parent_ = {-1};
    first_child_ = {-1};
    next_sibling_ = {-1};
    next_same_ = {-1};
    head_.assign(size_t(n_items), -1);
    support_.assign(size_t(n_items), 0);
    hkeys_.assign(64, -1);
    hvals_.assign(64, -1);
    hmask_ = 63;
}

void FPTree::rehash() {
    std::vector<int64_t> ok = std::move(hkeys_);
    std::vector<int32_t> ov = std::move(hvals_);
    size_t cap = ok.size() * 2;
    hkeys_.assign(cap, -1);
    hvals_.assign(cap, -1);
    hmask_ = cap - 1;
    for (size_t i = 0; i < ok.size(); ++i) {
        if (ok[i] < 0) continue;
        size_t h = size_t(ok[i] * 0x9E3779B97F4A7C15ull >> 20) & hmask_;
        while (hkeys_[h] >= 0) h = (h + 1) & hmask_;
        hkeys_[h] = ok[i];
        hvals_[h] = ov[i];
    }
}

int32_t FPTree::find_or_add_child(int32_t parent, int32_t item) {
    int64_t key = int64_t(parent) * n_items_ + item;
    size_t h = size_t(key * 0x9E3779B97F4A7C15ull >> 20) & hmask_;
    while (hkeys_[h] >= 0) {
        if (hkeys_[h] == key) return hvals_[h];
        h = (h + 1) & hmask_;
    }
    int32_t v = static_cast<int32_t>(item_.size());
    item_.push_back(item);
    count_.push_back(0);
    parent_.push_back(parent);
    first_child_.push_back(-1);
    if (first_child_[parent] != -1) single_path_ = false;
    next_sibling_.push_back(first_child_[parent]);
    first_child_[parent] = v;
    next_same_.push_back(head_[item]);
    head_[item] = v;
    hkeys_[h] = key;
    hvals_[h] = v;
    if (++hused_ * 2 > hkeys_.size()) rehash();
    return v;
}

void FPTree::insert(const int32_t* items, int len, int32_t w) {
    int32_t cur = 0;
    for (int i = 0; i < len; ++i) {
        cur = find_or_add_child(cur, items[i]);
        count_[cur] += w;
        support_[items[i]] += w;
    }
}

FPTree build_tree(const Csr& pre, int32_t n_items) {
    FPTree t(n_items);
    for (int32_t i = 0; i < pre.n_transactions(); ++i) t.insert(&pre.items[pre.offsets[i]], pre.len(i), 1);
    return t;
}

// ---------------------------------------------------------------------------
// Mineração

namespace {

struct Miner {
    int32_t minsup;
    ItemsetList& out;
    std::vector<int32_t> suffix;
    std::vector<int32_t> buf;

    void emit(const std::vector<int32_t>& extra, int32_t sup) {
        buf.assign(suffix.begin(), suffix.end());
        buf.insert(buf.end(), extra.begin(), extra.end());
        out.add(buf.data(), int(buf.size()), sup);
    }

    // Árvore com um único caminho: cada subconjunto não vazio dos itens do
    // caminho, com suporte igual ao contador do nó mais profundo escolhido.
    void emit_single_path(const FPTree& t) {
        std::vector<int32_t> path, cnt;
        for (int32_t v = t.first_child_[0]; v != -1; v = t.first_child_[v]) {
            path.push_back(t.item_[v]);
            cnt.push_back(t.count_[v]);
        }
        std::vector<int32_t> chosen;
        for (size_t j = 0; j < path.size(); ++j) {
            // j é o nó mais profundo; enumera subconjuntos de path[0..j).
            uint64_t combos = uint64_t(1) << j;
            for (uint64_t mask = 0; mask < combos; ++mask) {
                chosen.clear();
                for (size_t b = 0; b < j; ++b)
                    if (mask >> b & 1) chosen.push_back(path[b]);
                chosen.push_back(path[j]);
                emit(chosen, cnt[j]);
            }
        }
    }

    // Árvore condicional de t para o item y, já filtrada por minsup.
    FPTree conditional(const FPTree& t, int32_t y) {
        std::vector<int32_t> cnt(size_t(y), 0);
        for (int32_t v = t.head_[y]; v != -1; v = t.next_same_[v]) {
            int32_t w = t.count_[v];
            for (int32_t u = t.parent_[v]; u != 0; u = t.parent_[u]) cnt[t.item_[u]] += w;
        }
        FPTree c(y);
        std::vector<int32_t> path;
        for (int32_t v = t.head_[y]; v != -1; v = t.next_same_[v]) {
            path.clear();
            for (int32_t u = t.parent_[v]; u != 0; u = t.parent_[u])
                if (cnt[t.item_[u]] >= minsup) path.push_back(t.item_[u]);
            if (path.empty()) continue;
            std::reverse(path.begin(), path.end());
            c.insert(path.data(), int(path.size()), t.count_[v]);
        }
        return c;
    }

    void mine_tree(const FPTree& t) {
        if (t.n_nodes() == 0) return;
        if (t.single_path()) {
            emit_single_path(t);
            return;
        }
        for (int32_t y = t.n_items() - 1; y >= 0; --y) {
            if (t.support_[y] < minsup) continue;
            suffix.push_back(y);
            out.add(suffix.data(), int(suffix.size()), t.support_[y]);
            FPTree c = conditional(t, y);
            mine_tree(c);
            suffix.pop_back();
        }
    }
};

}  // namespace

ItemsetList mine(const FPTree& tree, int32_t minsup) {
    ItemsetList out;
    Miner m{minsup, out, {}, {}};
    m.mine_tree(tree);
    return out;
}

ItemsetList mine_extensions(const Csr& pre, const std::vector<int32_t>& suffix, int32_t minsup) {
    int32_t lo = *std::min_element(suffix.begin(), suffix.end());
    std::vector<int32_t> cnt(size_t(lo), 0);
    auto contains = [&](int32_t t) {
        auto b = pre.items.begin() + pre.offsets[t], e = pre.items.begin() + pre.offsets[t + 1];
        for (int32_t x : suffix)
            if (!std::binary_search(b, e, x)) return false;
        return true;
    };
    std::vector<int32_t> match;
    for (int32_t t = 0; t < pre.n_transactions(); ++t) {
        if (!contains(t)) continue;
        match.push_back(t);
        for (int32_t p = pre.offsets[t]; p < pre.offsets[t + 1] && pre.items[p] < lo; ++p) ++cnt[pre.items[p]];
    }
    FPTree c(lo);
    std::vector<int32_t> path;
    for (int32_t t : match) {
        path.clear();
        for (int32_t p = pre.offsets[t]; p < pre.offsets[t + 1] && pre.items[p] < lo; ++p)
            if (cnt[pre.items[p]] >= minsup) path.push_back(pre.items[p]);
        if (!path.empty()) c.insert(path.data(), int(path.size()), 1);
    }
    ItemsetList out;
    Miner m{minsup, out, suffix, {}};
    m.mine_tree(c);
    return out;
}

// ---------------------------------------------------------------------------
// Dump textual da árvore

std::vector<std::string> dump_linear_tree(const std::vector<int32_t>& item, const std::vector<int32_t>& count,
                                          const std::vector<int32_t>& parent) {
    std::vector<std::string> lines(item.size());
    std::vector<int32_t> path;
    char num[16];
    for (size_t v = 0; v < item.size(); ++v) {
        path.clear();
        for (int32_t u = int32_t(v); u != -1; u = parent[u]) path.push_back(item[u]);
        std::string& s = lines[v];
        for (size_t i = path.size(); i-- > 0;) {
            int n = std::snprintf(num, sizeof num, i + 1 == path.size() ? "%d" : " %d", path[i]);
            s.append(num, n);
        }
        int n = std::snprintf(num, sizeof num, " (%d)", count[v]);
        s.append(num, n);
    }
    std::sort(lines.begin(), lines.end());
    return lines;
}

std::vector<std::string> dump_tree(const FPTree& t) {
    // Remove a raiz e desloca os índices em 1.
    std::vector<int32_t> item(t.item_.begin() + 1, t.item_.end());
    std::vector<int32_t> count(t.count_.begin() + 1, t.count_.end());
    std::vector<int32_t> parent(t.parent_.begin() + 1, t.parent_.end());
    for (auto& p : parent) p -= 1;  // raiz (0) vira -1
    return dump_linear_tree(item, count, parent);
}

bool csr_equal(const Csr& a, const Csr& b) { return a.items == b.items && a.offsets == b.offsets; }

}  // namespace fpg
