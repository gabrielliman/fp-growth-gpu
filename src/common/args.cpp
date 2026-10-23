#include "common/args.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace fpg {

double Args::get_double(const std::string& k, double def) const {
    return has(k) ? std::strtod(get(k).c_str(), nullptr) : def;
}

long Args::get_long(const std::string& k, long def) const {
    return has(k) ? std::strtol(get(k).c_str(), nullptr, 10) : def;
}

Args parse_args(int argc, char** argv, const std::vector<std::string>& flags) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) {
            a.positional.push_back(s);
            continue;
        }
        s = s.substr(2);
        size_t eq = s.find('=');
        if (eq != std::string::npos) {
            a.opts[s.substr(0, eq)] = s.substr(eq + 1);
        } else if (std::find(flags.begin(), flags.end(), s) != flags.end()) {
            a.opts[s] = "1";
        } else {
            if (i + 1 >= argc) throw std::runtime_error("faltou valor para --" + s);
            a.opts[s] = argv[++i];
        }
    }
    return a;
}

}  // namespace fpg
