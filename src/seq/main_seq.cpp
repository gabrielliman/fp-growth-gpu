// fpg_seq: FP-Growth sequencial.
// uso: fpg_seq <arquivo.dat> --minsup <fração> [--out saida.txt] [--dump-tree arvore.txt]
//              [--name dataset] [--stats] [--no-mine]

#include <cstdio>
#include <exception>

#include "common/args.hpp"
#include "common/fimi.hpp"
#include "common/timer.hpp"
#include "seq/fpgrowth_seq.hpp"

int main(int argc, char** argv) try {
    auto args = fpg::parse_args(argc, argv, {"stats", "no-mine"});
    if (args.positional.empty()) {
        std::fprintf(stderr, "uso: fpg_seq <arquivo.dat> --minsup <fração> [--out f] [--dump-tree f] [--stats]\n");
        return 1;
    }
    const std::string path = args.positional[0];
    const std::string name = args.get("name", fpg::dataset_name(path));

    fpg::CpuTimer total, t;
    auto db = fpg::read_fimi(path);
    double t_read = t.ms();
    if (args.has("stats")) {
        auto s = fpg::compute_stats(db);
        std::printf("%s,%lld,%lld,%.2f,%lld,%d\n", name.c_str(), (long long)s.n_transactions,
                    (long long)s.n_distinct_items, s.avg_len(), (long long)s.max_len, s.max_item_id);
        return 0;
    }
    const double minsup = args.get_double("minsup", 0.01);
    const int32_t minsup_abs = fpg::minsup_count(minsup, db.n_transactions());
    fpg::StageTimes times("seq", name, minsup);
    times.add("leitura", t_read);

    t.reset();
    auto support = fpg::count_support(db);
    times.add("contagem", t.ms());

    t.reset();
    auto ranks = fpg::build_ranks(support, minsup_abs);
    auto pre = fpg::preprocess(db, ranks);
    times.add("preprocessamento", t.ms());

    t.reset();
    auto tree = fpg::build_tree(pre, ranks.n_frequent());
    times.add("construcao", t.ms());

    t.reset();
    fpg::ItemsetList sets;
    if (!args.has("no-mine")) sets = fpg::mine(tree, minsup_abs);
    times.add("mineracao", t.ms());

    t.reset();
    if (args.has("out")) fpg::write_lines(args.get("out"), fpg::to_canonical_lines(sets, ranks.rank_to_id));
    times.add("escrita", t.ms());
    times.add("total", total.ms());

    if (args.has("dump-tree")) fpg::write_lines(args.get("dump-tree"), fpg::dump_tree(tree));

    std::fprintf(stderr, "[seq] %s minsup=%g (%d) itens_freq=%d transacoes=%d itens_pre=%zu nos=%d itemsets=%zu\n",
                 name.c_str(), minsup, minsup_abs, ranks.n_frequent(), pre.n_transactions(), pre.items.size(),
                 tree.n_nodes(), sets.size());
    times.print_csv();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "erro: %s\n", e.what());
    return 1;
}
