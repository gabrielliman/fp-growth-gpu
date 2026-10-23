// fpg_apriori_gpu / fpg_eclat_gpu: baselines de mineração em GPU com bitsets
// verticais (ver vertical.cuh). Leitura, contagem, ranks e pré-processamento
// usam o mesmo código do sequencial; a saída é a mesma saída canônica.
// uso: fpg_{apriori,eclat}_gpu <arquivo.dat> --minsup <fração> [--out f] [--device D] [--pool-slots N]

#include <cstdio>
#include <exception>

#include "baselines/vertical.cuh"
#include "common/args.hpp"
#include "common/cuda_check.cuh"
#include "common/fimi.hpp"
#include "common/timer.hpp"
#include "seq/fpgrowth_seq.hpp"

#ifndef FPG_VERTICAL_ALGO
#error "defina FPG_VERTICAL_ALGO como 0 (apriori) ou 1 (eclat)"
#endif

using namespace fpg;

int main(int argc, char** argv) try {
    constexpr bool eclat = FPG_VERTICAL_ALGO == 1;
    const char* impl = eclat ? "gpu_eclat" : "gpu_apriori";
    auto args = parse_args(argc, argv, {});
    if (args.positional.empty()) {
        std::fprintf(stderr, "uso: %s <arquivo.dat> --minsup <fração> [--out f]\n", argv[0]);
        return 1;
    }
    const std::string path = args.positional[0];
    const std::string name = args.get("name", dataset_name(path));
    const double minsup = args.get_double("minsup", 0.01);
    const int device = static_cast<int>(args.get_long("device", 0));
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // contexto fora das medições
    char pci_bus[32] = {0};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus, sizeof pci_bus, device));

    CpuTimer total, t;
    Csr db = read_fimi(path);
    const int32_t minsup_abs = minsup_count(minsup, db.n_transactions());
    StageTimes times(impl, name, minsup);
    times.add("leitura", t.ms());

    t.reset();
    auto support = count_support(db);
    times.add("contagem", t.ms());
    t.reset();
    auto ranks = build_ranks(support, minsup_abs);
    Csr pre = preprocess(db, ranks);
    times.add("preprocessamento", t.ms());

    baselines::VerticalStats st;
    ItemsetList sets =
        eclat ? baselines::mine_eclat_gpu(pre, ranks.n_frequent(), ranks.rank_support, minsup_abs, times, st,
                                          args.get_long("pool-slots", 0))
              : baselines::mine_apriori_gpu(pre, ranks.n_frequent(), ranks.rank_support, minsup_abs, times, st);

    t.reset();
    if (args.has("out")) write_lines(args.get("out"), to_canonical_lines(sets, ranks.rank_to_id));
    times.add("escrita", t.ms());
    times.add("total", total.ms());

    std::fprintf(stderr,
                 "[gpu] %s %s minsup=%g (%d) itens_freq=%d transacoes=%d itemsets=%zu candidatos=%lld "
                 "lancamentos=%lld niveis=%d mem_gpu_MB=%.0f pci=%s\n",
                 impl, name.c_str(), minsup, minsup_abs, ranks.n_frequent(), pre.n_transactions(), sets.size(),
                 (long long)st.candidates, (long long)st.kernel_launches, st.levels, st.gpu_bytes / 1048576.0,
                 pci_bus);
    times.print_csv();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "erro: %s\n", e.what());
    return 1;
}
