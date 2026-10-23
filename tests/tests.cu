// Suíte de testes: compara cada etapa da GPU com o sequencial e o sequencial
// com as saídas calculadas à mão.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/fimi.hpp"
#include "common/output.hpp"
#include "baselines/vertical.cuh"
#include "gpu/gpu.cuh"
#include "seq/fpgrowth_seq.hpp"

using namespace fpg;

namespace {

int g_failures = 0, g_checks = 0;
// O Eclat em GPU tem um bug conhecido (ver TODO.md); seus testes só rodam com --with-eclat.
bool g_with_eclat = false;

void expect(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FALHOU: %s\n", what.c_str());
    }
}

Csr make_csr(const std::vector<std::vector<int32_t>>& txs) {
    Csr c;
    for (auto t : txs) {
        std::sort(t.begin(), t.end());
        t.erase(std::unique(t.begin(), t.end()), t.end());
        c.items.insert(c.items.end(), t.begin(), t.end());
        c.offsets.push_back(int32_t(c.items.size()));
    }
    return c;
}

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> v;
    for (std::string l; std::getline(f, l);) v.push_back(l);
    return v;
}

struct SeqResult {
    RankTable ranks;
    Csr pre;
    std::vector<std::string> tree, sets;
};

SeqResult run_seq(const Csr& db, int32_t minsup) {
    SeqResult r;
    r.ranks = build_ranks(count_support(db), minsup);
    r.pre = preprocess(db, r.ranks);
    FPTree t = build_tree(r.pre, r.ranks.n_frequent());
    r.tree = dump_tree(t);
    r.sets = to_canonical_lines(mine(t, minsup), r.ranks.rank_to_id);
    return r;
}

// Compara pré-processamento, árvore e (opcionalmente) mineração da GPU com o sequencial.
void check_gpu(const std::string& label, const Csr& db, int32_t minsup, bool with_mining) {
    SeqResult s = run_seq(db, minsup);
    int32_t max_id = -1;
    for (int32_t x : db.items) max_id = std::max(max_id, x);
    StageTimes times("test", label, 0);
    gpu::DeviceCsr d = gpu::upload(db, times);
    for (auto k : {gpu::CountKernel::V1, gpu::CountKernel::V2}) {
        gpu::Preprocessed pre = gpu::preprocess(d, max_id, minsup, k, times);
        expect(pre.rank_to_id == s.ranks.rank_to_id, label + ": ranks");
        expect(csr_equal(gpu::download(pre.db), s.pre), label + ": CSR pre-processado");
        if (k == gpu::CountKernel::V1) continue;

        gpu::BuildOptions opt;
        opt.cross_check = true;
        gpu::BuildStats st;
        gpu::DeviceTree tree = gpu::build_tree(pre.db, {}, pre.n_frequent, opt, times, &st);
        std::vector<int32_t> item, count, parent;
        gpu::download_tree(tree, item, count, parent);
        expect(dump_linear_tree(item, count, parent) == s.tree, label + ": FP-tree");
        expect(st.counts_match, label + ": contadores 3.6 == 3.7");

        if (with_mining) {
            // Baselines Apriori e Eclat em GPU.
            baselines::VerticalStats vs;
            Csr pre_host = gpu::download(pre.db);
            std::vector<int32_t> rsup(pre.n_frequent);
            thrust::copy(pre.rank_support.begin(), pre.rank_support.end(), rsup.begin());
            expect(to_canonical_lines(baselines::mine_apriori_gpu(pre_host, pre.n_frequent, rsup, minsup, times, vs),
                                      pre.rank_to_id) == s.sets, label + ": apriori gpu");
            if (g_with_eclat) {
                expect(to_canonical_lines(baselines::mine_eclat_gpu(pre_host, pre.n_frequent, rsup, minsup, times, vs),
                                          pre.rank_to_id) == s.sets, label + ": eclat gpu");
                // Pool no limite do pior caso: força lotes menores e expansão parcial.
                const int64_t nf = pre.n_frequent;
                expect(to_canonical_lines(baselines::mine_eclat_gpu(pre_host, pre.n_frequent, rsup, minsup, times, vs,
                                                                    nf * (nf + 1) / 2 + nf + 2),
                                          pre.rank_to_id) == s.sets, label + ": eclat gpu (pool minimo)");
            }
            for (auto mode : {gpu::MineMode::Static, gpu::MineMode::Dynamic}) {
                for (int grain : {0, 64, 1 << 30}) {
                    if (mode == gpu::MineMode::Static && grain != 0) continue;
                    gpu::MineOptions mo;
                    mo.mode = mode;
                    mo.grain = grain;
                    gpu::MineStats ms;
                    ItemsetList sets = gpu::mine(tree, minsup, mo, pre.db, times, ms);
                    std::string tag = mode == gpu::MineMode::Static ? "static" : "dynamic/" + std::to_string(grain);
                    expect(to_canonical_lines(sets, pre.rank_to_id) == s.sets, label + ": mineracao " + tag);
                }
            }
        }
    }
}

// 4.1: árvore com pesos == árvore das transações repetidas.
void check_weighted(const std::string& label, const Csr& pre, const std::vector<int32_t>& w, int32_t n_items) {
    Csr expanded;
    for (int32_t t = 0; t < pre.n_transactions(); ++t)
        for (int32_t r = 0; r < w[t]; ++r) {
            expanded.items.insert(expanded.items.end(), pre.items.begin() + pre.offsets[t],
                                  pre.items.begin() + pre.offsets[t + 1]);
            expanded.offsets.push_back(int32_t(expanded.items.size()));
        }
    auto expected = dump_tree(build_tree(expanded, n_items));
    StageTimes times("test", label, 0);
    gpu::DeviceCsr d = gpu::upload(pre, times);
    thrust::device_vector<int32_t> dw(w.begin(), w.end());
    gpu::BuildOptions opt;
    opt.cross_check = true;
    gpu::BuildStats st;
    gpu::DeviceTree tree = gpu::build_tree(d, dw, n_items, opt, times, &st);
    std::vector<int32_t> item, count, parent;
    gpu::download_tree(tree, item, count, parent);
    expect(dump_linear_tree(item, count, parent) == expected, label + ": arvore ponderada");
    expect(st.counts_match, label + ": contadores ponderados 3.6 == 3.7");
}

Csr random_db(std::mt19937& rng, int n_tx, int n_items, int max_len, double skew) {
    std::vector<std::vector<int32_t>> txs(n_tx);
    std::uniform_int_distribution<int> len(0, max_len);
    std::uniform_real_distribution<double> u(0, 1);
    for (auto& t : txs) {
        int l = len(rng);
        for (int i = 0; i < l; ++i) t.push_back(int32_t(std::pow(u(rng), skew) * n_items));
    }
    return make_csr(txs);
}

}  // namespace

int main(int argc, char** argv) {
    bool with_mining = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-mining") with_mining = false;
        if (a == "--with-eclat") g_with_eclat = true;
    }
    CUDA_CHECK(cudaFree(nullptr));
    const std::string dir = TESTS_DATA_DIR;

    std::printf("[1] sequencial x saida calculada a mao\n");
    struct Hand { const char* name; double minsup; };
    for (Hand h : {Hand{"han", 0.6}, Hand{"small", 0.4}, Hand{"edges", 0.25}, Hand{"identical", 0.5},
                   Hand{"singletons", 0.4}}) {
        Csr db = read_fimi(dir + "/" + h.name + ".dat");
        std::ostringstream exp;
        exp << dir << "/" << h.name << "_" << h.minsup << ".expected";
        SeqResult s = run_seq(db, minsup_count(h.minsup, db.n_transactions()));
        expect(s.sets == read_lines(exp.str()), std::string(h.name) + ": saida a mao");
    }

    std::printf("[2] casos de borda da Fase 3 (GPU x sequencial)\n");
    const std::vector<std::pair<std::string, std::vector<std::vector<int32_t>>>> edge_cases = {
        {"duplicadas", {{1, 2, 3}, {1, 2, 3}, {1, 2}, {1, 2}, {4}, {4}}},
        {"prefixo da seguinte", {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {2, 3}}},
        {"tamanho 1", {{1}, {2}, {3}, {1}, {2}, {1}}},
        {"todas iguais", {{5, 6, 7, 8}, {5, 6, 7, 8}, {5, 6, 7, 8}, {5, 6, 7, 8}}},
        {"uma transacao", {{3, 1, 2}}},
        {"com vazias", {{}, {1, 2}, {}, {1}, {2}, {}}},
        {"irmaos na raiz", {{1, 9}, {2, 9}, {3, 9}, {4, 9}, {5, 9}}},
    };
    for (auto& [name, txs] : edge_cases) {
        Csr db = make_csr(txs);
        for (int32_t minsup : {1, 2})
            check_gpu(name + " minsup=" + std::to_string(minsup), db, minsup, with_mining);
    }
    for (Hand h : {Hand{"han", 0.6}, Hand{"small", 0.4}, Hand{"edges", 0.25}, Hand{"identical", 0.5},
                   Hand{"singletons", 0.4}}) {
        Csr db = read_fimi(dir + "/" + h.name + ".dat");
        check_gpu(h.name, db, minsup_count(h.minsup, db.n_transactions()), with_mining);
    }

    std::printf("[3] bases aleatorias (GPU x sequencial)\n");
    std::mt19937 rng(12345);
    for (int it = 0; it < 30; ++it) {
        int n_tx = 1 + int(rng() % 2000), n_items = 2 + int(rng() % 60), max_len = 1 + int(rng() % 20);
        double skew = 1.0 + (rng() % 4);
        Csr db = random_db(rng, n_tx, n_items, max_len, skew);
        int32_t minsup = 1 + int32_t(rng() % std::max(1, n_tx / 10));
        check_gpu("aleatoria " + std::to_string(it), db, minsup, with_mining);
    }

    std::printf("[4] arvore com pesos (4.1)\n");
    for (int it = 0; it < 10; ++it) {
        Csr db = random_db(rng, 1 + int(rng() % 500), 30, 12, 2.0);
        auto ranks = build_ranks(count_support(db), 1);
        Csr pre = preprocess(db, ranks);
        std::vector<int32_t> w(pre.n_transactions());
        for (auto& x : w) x = 1 + int32_t(rng() % 5);
        check_weighted("pesos " + std::to_string(it), pre, w, ranks.n_frequent());
        std::vector<int32_t> ones(pre.n_transactions(), 1);
        check_weighted("pesos=1 " + std::to_string(it), pre, ones, ranks.n_frequent());
    }

    std::printf("%d verificacoes, %d falhas\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
