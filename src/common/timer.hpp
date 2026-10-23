#pragma once
// Medição de tempo por etapa e impressão em CSV: impl,dataset,minsup,etapa,tempo_ms

#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace fpg {

class StageTimes {
public:
    StageTimes(std::string impl, std::string dataset, double minsup)
        : impl_(std::move(impl)), dataset_(std::move(dataset)), minsup_(minsup) {}

    void add(const std::string& stage, double ms) {
        for (auto& r : rows_)
            if (r.first == stage) { r.second += ms; return; }
        rows_.emplace_back(stage, ms);
    }
    double get(const std::string& stage) const {
        for (auto& r : rows_)
            if (r.first == stage) return r.second;
        return 0.0;
    }
    void print_csv(FILE* f = stdout) const {
        for (auto& r : rows_)
            std::fprintf(f, "%s,%s,%g,%s,%.4f\n", impl_.c_str(), dataset_.c_str(), minsup_, r.first.c_str(), r.second);
        std::fflush(f);
    }

private:
    std::string impl_, dataset_;
    double minsup_;
    std::vector<std::pair<std::string, double>> rows_;
};

class CpuTimer {
public:
    CpuTimer() : t0_(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }
    void reset() { t0_ = std::chrono::steady_clock::now(); }

private:
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace fpg
