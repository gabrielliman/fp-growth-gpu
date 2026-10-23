#pragma once
// Parser mínimo de argumentos: posicionais e --chave valor / --chave=valor / --flag.

#include <map>
#include <string>
#include <vector>

namespace fpg {

struct Args {
    std::vector<std::string> positional;
    std::map<std::string, std::string> opts;

    bool has(const std::string& k) const { return opts.count(k) > 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = opts.find(k);
        return it == opts.end() ? def : it->second;
    }
    double get_double(const std::string& k, double def) const;
    long get_long(const std::string& k, long def) const;
};

// flags: nomes de opções sem valor (ex.: "debug").
Args parse_args(int argc, char** argv, const std::vector<std::string>& flags);

}  // namespace fpg
