#include "common/fimi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace fpg {

Csr read_fimi(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("não foi possível abrir " + path);
    std::vector<char> buf;
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    buf.resize(size > 0 ? size_t(size) : 0);
    if (size > 0 && std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fclose(f);
        throw std::runtime_error("erro lendo " + path);
    }
    std::fclose(f);

    Csr db;
    db.items.reserve(buf.size() / 3);
    bool line_has_content = false;
    int64_t cur = -1;  // inteiro sendo lido (-1 = nenhum)
    auto flush_tx = [&]() {
        auto b = db.items.begin() + db.offsets.back();
        std::sort(b, db.items.end());
        db.items.erase(std::unique(b, db.items.end()), db.items.end());
        db.offsets.push_back(static_cast<int32_t>(db.items.size()));
    };
    for (size_t i = 0; i <= buf.size(); ++i) {
        char c = i < buf.size() ? buf[i] : '\n';
        if (c >= '0' && c <= '9') {
            cur = (cur < 0 ? 0 : cur * 10) + (c - '0');
            line_has_content = true;
        } else {
            if (cur >= 0) {
                if (cur > INT32_MAX) throw std::runtime_error("item fora do intervalo int32");
                db.items.push_back(static_cast<int32_t>(cur));
                cur = -1;
            }
            if (c == '\n') {
                // A última linha sem '\n' só vira transação se tiver conteúdo.
                if (i < buf.size() || line_has_content) flush_tx();
                line_has_content = false;
            } else if (c != ' ' && c != '\t' && c != '\r') {
                throw std::runtime_error("caractere inválido em " + path);
            }
        }
    }
    if (db.items.size() > size_t(INT32_MAX)) throw std::runtime_error("base grande demais para offsets int32");
    return db;
}

DatasetStats compute_stats(const Csr& db) {
    DatasetStats s;
    s.n_transactions = db.n_transactions();
    s.total_items = static_cast<int64_t>(db.items.size());
    for (int32_t x : db.items) s.max_item_id = std::max(s.max_item_id, x);
    std::vector<char> seen(size_t(s.max_item_id + 1), 0);
    for (int32_t x : db.items) seen[x] = 1;
    for (char c : seen) s.n_distinct_items += c;
    for (int32_t t = 0; t < db.n_transactions(); ++t) s.max_len = std::max<int64_t>(s.max_len, db.len(t));
    return s;
}

int32_t minsup_count(double minsup, int64_t n_transactions) {
    double v = std::ceil(minsup * double(n_transactions) - 1e-9);
    return static_cast<int32_t>(std::max(1.0, v));
}

std::string dataset_name(const std::string& path) {
    size_t slash = path.find_last_of('/');
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
}

}  // namespace fpg
