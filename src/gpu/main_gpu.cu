// fpg_gpu: FP-Growth com contagem, pré-processamento, construção e mineração na GPU.
// uso: fpg_gpu <arquivo.dat> --minsup <fração> [--mine=static|dynamic] [--count=v1|v2]
//              [--grain N] [--out saida.txt] [--dump-tree arvore.txt] [--debug]
//              [--count-bench N] [--device D] [--name dataset] [--no-mine]
//              [--block-stats blocos.csv] [--arena-kb K] [--pool-mb M] [--queue Q]

#include <cstdio>
#include <exception>

#include "common/args.hpp"
#include "common/fimi.hpp"
#include "common/timer.hpp"
#include "gpu/gpu.cuh"
#include "seq/fpgrowth_seq.hpp"

using namespace fpg;

namespace {

// 2.9: tempo médio dos kernels de contagem v1 e v2.
void count_bench(const gpu::DeviceCsr& db, int32_t max_id, int reps, const std::string& name) {
    for (auto k : {gpu::CountKernel::V1, gpu::CountKernel::V2}) {
        gpu::count_support(db, max_id, k);  // aquecimento
        gpu::GpuTimer t;
        for (int i = 0; i < reps; ++i) gpu::count_support(db, max_id, k);
        std::printf("countbench,%s,%s,%.4f\n", name.c_str(), k == gpu::CountKernel::V1 ? "v1" : "v2", t.ms() / reps);
    }
}

bool check(bool ok, const char* what) {
    std::fprintf(stderr, "[debug] %-40s %s\n", what, ok ? "OK" : "FALHOU");
    return ok;
}

}  // namespace

int main(int argc, char** argv) try {
    auto args = parse_args(argc, argv, {"debug", "no-mine"});
    std::string impl = "gpu_" + args.get("mine", "static");
    if (args.positional.empty()) {
        std::fprintf(stderr, "uso: fpg_gpu <arquivo.dat> --minsup <fração> [--mine=static|dynamic] ...\n");
        return 1;
    }
    const std::string path = args.positional[0];
    const std::string name = args.get("name", dataset_name(path));
    const double minsup = args.get_double("minsup", 0.01);
    const bool debug = args.has("debug");
    const auto count_kernel = args.get("count", "v2") == "v1" ? gpu::CountKernel::V1 : gpu::CountKernel::V2;

    const int device = static_cast<int>(args.get_long("device", 0));
    CUDA_CHECK(cudaSetDevice(device));
    char pci_bus[32] = {0};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus, sizeof pci_bus, device));
    CUDA_CHECK(cudaFree(nullptr));  // inicializa o contexto fora das medições
    gpu::MemTracker::init();

    CpuTimer total, t;
    Csr db = read_fimi(path);
    const int32_t minsup_abs = minsup_count(minsup, db.n_transactions());
    int32_t max_id = -1;
    for (int32_t x : db.items) max_id = std::max(max_id, x);
    StageTimes times(impl, name, minsup);
    times.add("leitura", t.ms());

    gpu::DeviceCsr d_db = gpu::upload(db, times);
    if (args.has("count-bench")) {
        count_bench(d_db, max_id, static_cast<int>(args.get_long("count-bench", 20)), name);
        return 0;
    }
    gpu::Preprocessed pre = gpu::preprocess(d_db, max_id, minsup_abs, count_kernel, times);

    int failures = 0;
    Csr pre_host, seq_pre;
    if (debug) {
        // 2.8: compara suportes, ranks e CSR pré-processado com o sequencial.
        auto support = count_support(db);
        auto ranks = build_ranks(support, minsup_abs);
        seq_pre = fpg::preprocess(db, ranks);
        pre_host = gpu::download(pre.db);
        failures += !check(std::equal(support.begin(), support.end(), pre.h_support.begin()), "suportes");
        failures += !check(ranks.rank_to_id == pre.rank_to_id, "tabela de ranks");
        failures += !check(csr_equal(seq_pre, pre_host), "CSR pre-processado");
    }

    // Fase 3: construção da FP-tree.
    gpu::BuildOptions bopt;
    bopt.cross_check = debug;
    gpu::BuildStats bstats;
    gpu::DeviceTree tree = gpu::build_tree(pre.db, {}, pre.n_frequent, bopt, times, &bstats);

    if (debug || args.has("dump-tree")) {
        std::vector<int32_t> item, count, parent;
        gpu::download_tree(tree, item, count, parent);
        auto lines = dump_linear_tree(item, count, parent);
        if (args.has("dump-tree")) write_lines(args.get("dump-tree"), lines);
        if (debug) {
            // 3.9: mesma árvore que a sequencial (mesmo formato textual e número de nós).
            FPTree seq_tree = fpg::build_tree(seq_pre, pre.n_frequent);
            failures += !check(seq_tree.n_nodes() == tree.n_nodes, "numero de nos");
            failures += !check(dump_tree(seq_tree) == lines, "FP-tree (formato textual)");
            failures += !check(bstats.counts_match, "contadores 3.6 == 3.7 (reduce_by_key)");
            std::vector<int32_t> isup(pre.n_frequent);
            thrust::copy(tree.item_support.begin(), tree.item_support.end(), isup.begin());
            std::vector<int32_t> hoff(pre.n_frequent + 1);
            thrust::copy(tree.header_offsets.begin(), tree.header_offsets.end(), hoff.begin());
            std::vector<int32_t> seq_nodes_per_item(pre.n_frequent + 1, 0);
            for (int32_t v = 1; v <= seq_tree.n_nodes(); ++v) ++seq_nodes_per_item[seq_tree.item_[v] + 1];
            for (int32_t r = 0; r < pre.n_frequent; ++r) seq_nodes_per_item[r + 1] += seq_nodes_per_item[r];
            failures += !check(isup == seq_tree.support_, "cabecalho: suporte por item");
            failures += !check(hoff == seq_nodes_per_item, "cabecalho: offsets");
        }
    }

    // Fases 4 e 5: mineração.
    gpu::MineOptions mopt;
    mopt.mode = args.get("mine", "static") == "dynamic" ? gpu::MineMode::Dynamic : gpu::MineMode::Static;
    mopt.grain = static_cast<int>(args.get_long("grain", 64));
    mopt.arena_bytes = size_t(args.get_long("arena-kb", 0)) << 10;
    mopt.pool_bytes = size_t(args.get_long("pool-mb", 0)) << 20;
    mopt.queue_capacity = static_cast<int>(args.get_long("queue", 1 << 16));
    mopt.block_stats_path = args.get("block-stats", "");
    gpu::MineStats mstats;
    ItemsetList sets;
    if (!args.has("no-mine")) sets = gpu::mine(tree, minsup_abs, mopt, pre.db, times, mstats);

    t.reset();
    std::vector<std::string> lines;
    if (args.has("out") || debug) lines = to_canonical_lines(sets, pre.rank_to_id);
    if (args.has("out")) write_lines(args.get("out"), lines);
    times.add("escrita", t.ms());
    times.add("total", total.ms());

    if (debug && !args.has("no-mine")) {
        // 4.10 / 5.8: mesma saída que o sequencial.
        FPTree seq_tree = fpg::build_tree(seq_pre, pre.n_frequent);
        auto seq_lines = to_canonical_lines(fpg::mine(seq_tree, minsup_abs), pre.rank_to_id);
        failures += !check(seq_lines == lines, "itemsets identicos ao sequencial");
    }

    std::fprintf(stderr,
                 "[gpu] %s minsup=%g (%d) itens_freq=%d transacoes=%d nos=%d mem_construcao_MB=%.1f itemsets=%zu "
                 "mine=%s grain=%d blocos=%d tarefas=%lld enfileiradas=%lld estouros=%lld desbalanceamento=%.2f "
                 "mem_mineracao_MB=%.0f mem_gpu_pico_MB=%.0f pci=%s\n",
                 name.c_str(), minsup, minsup_abs, pre.n_frequent, pre.db.n(), tree.n_nodes,
                 bstats.peak_bytes / 1048576.0, sets.size(), mopt.mode == gpu::MineMode::Dynamic ? "dynamic" : "static",
                 mopt.grain, mstats.n_blocks, (long long)mstats.n_tasks, (long long)mstats.n_enqueued,
                 (long long)mstats.n_overflow_tasks, mstats.max_over_mean_block_time, mstats.peak_bytes / 1048576.0,
                 gpu::MemTracker::peak() / 1048576.0, pci_bus);
    times.print_csv();
    return failures ? 2 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "erro: %s\n", e.what());
    return 1;
}
