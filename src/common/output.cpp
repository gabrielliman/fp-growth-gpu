#include "common/output.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace fpg {

std::vector<std::string> to_canonical_lines(const ItemsetList& sets, const std::vector<int32_t>& rank_to_id) {
    std::vector<std::string> lines(sets.size());
    std::vector<int32_t> tmp;
    char num[16];
    for (size_t s = 0; s < sets.size(); ++s) {
        tmp.assign(sets.items.begin() + sets.offsets[s], sets.items.begin() + sets.offsets[s + 1]);
        if (!rank_to_id.empty())
            for (int32_t& x : tmp) x = rank_to_id[x];
        std::sort(tmp.begin(), tmp.end());
        std::string& line = lines[s];
        for (size_t i = 0; i < tmp.size(); ++i) {
            int n = std::snprintf(num, sizeof num, i ? " %d" : "%d", tmp[i]);
            line.append(num, n);
        }
        int n = std::snprintf(num, sizeof num, " (%d)", sets.support[s]);
        line.append(num, n);
    }
    std::sort(lines.begin(), lines.end());
    return lines;
}

void write_lines(const std::string& path, const std::vector<std::string>& lines) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("não foi possível escrever " + path);
    std::vector<char> buf(1 << 20);
    std::setvbuf(f, buf.data(), _IOFBF, buf.size());
    for (const auto& l : lines) {
        std::fwrite(l.data(), 1, l.size(), f);
        std::fputc('\n', f);
    }
    std::fclose(f);
}

}  // namespace fpg
