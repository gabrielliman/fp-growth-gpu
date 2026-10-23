#!/usr/bin/env python3
"""6.8: gráficos e tabelas a partir de results/{raw,runs,grain,build}.csv.

Gera results/fig_*.png e results/tables.md (mediana ± desvio padrão de cada medida).
"""
import glob
import os

import numpy as np
import pandas as pd

from gpuutil import MID, RESULTS_DIR as RES, ROOT  # noqa: F401
from plotstyle import COLORS, LABELS, SERIES, setup

DATASETS = ["chess", "mushroom", "accidents", "T10I4D100K", "retail", "kosarak"]
IMPLS = ["seq", "gpu_static", "gpu_dynamic"]
STAGES = ["leitura", "transferencia_h2d", "contagem", "preprocessamento", "construcao", "mineracao",
          "transferencia_d2h", "escrita"]
STAGE_LABELS = {"leitura": "leitura", "transferencia_h2d": "H2D", "contagem": "contagem",
                "preprocessamento": "pré-proc.", "construcao": "construção", "mineracao": "mineração",
                "transferencia_d2h": "D2H + montagem", "escrita": "escrita"}
SUBSTAGES = ["construcao_sort", "construcao_lcp", "construcao_scan", "construcao_nos", "construcao_sparse",
             "construcao_pais", "construcao_contadores", "construcao_cabecalho"]

plt = setup()
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


def plain(v, _pos=None):
    if v >= 1:
        return f"{v:,.0f}".replace(",", ".")
    return f"{v:g}".replace(".", ",")


def plain_y(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fmt_sup(s):
    t = f"{s:.6f}".rstrip("0").rstrip(".") if s < 0.001 else f"{s:g}"
    return t.replace(".", ",")


def xf(x, nd=1):
    """Razão com vírgula decimal."""
    return f"{x:.{nd}f}×".replace(".", ",")


def intf(n):
    return f"{int(n):,}".replace(",", ".")


def ms_fmt(m, sd=None):
    def one(x):
        if pd.isna(x):
            return "—"
        if x >= 100:
            return f"{x:,.0f}".replace(",", ".")
        if x >= 1:
            return f"{x:.1f}".replace(".", ",")
        return f"{x:.3f}".replace(".", ",")
    return one(m) if sd is None else f"{one(m)} ± {one(sd)}"


def load():
    raw = pd.read_csv(os.path.join(RES, "raw.csv"))
    runs = pd.read_csv(os.path.join(RES, "runs.csv"))
    return raw, runs


def stage_stats(raw):
    """Mediana e desvio por (impl, dataset, minsup, etapa), só execuções válidas."""
    v = raw[raw.valido == 1]
    g = v.groupby(["impl", "dataset", "minsup", "etapa"]).tempo_ms
    return g.median().unstack("etapa"), g.std().unstack("etapa")


def save(fig, name):
    path = os.path.join(RES, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("gravado", path)


def blues(n):
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    return ramp[-n:] if n <= len(ramp) else ramp


# (a) speedup da construção --------------------------------------------------
def fig_construction(med):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, without_sort in zip(axes, [False, True]):
        x = np.arange(len(DATASETS))
        sups_per = [sorted(med.loc["seq"].loc[d].index, reverse=True) for d in DATASETS]
        n = max(len(s) for s in sups_per)
        width = 0.8 / n
        cols = blues(n)
        for k in range(n):
            vals = []
            for d, sups in zip(DATASETS, sups_per):
                s = sups[k]
                seq = med.loc[("seq", d, s), "construcao"]
                gpu = med.loc[("gpu_static", d, s), "construcao"]
                if without_sort:
                    gpu -= med.loc[("gpu_static", d, s), "construcao_sort"]
                vals.append(seq / gpu)
            ax.bar(x - 0.4 + width * (k + 0.5), vals, width * 0.92, color=cols[k],
                   label=f"{k + 1}º suporte (maior → menor)")
        ax.axhline(1, color="#0b0b0b", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(x, DATASETS, rotation=25, ha="right")
        plain_y(ax)
        ax.set_title(("Sem a ordenação lexicográfica" if without_sort else "Construção completa"), loc="left")
        ax.set_ylabel("speedup sobre a construção sequencial (×)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("(a) Speedup da construção da FP-tree na GPU, por dataset e suporte (escala log)",
                 x=0.01, ha="left", fontsize=11)
    save(fig, "fig_a_construcao_speedup.png")


# (b) tempo total × suporte ----------------------------------------------------
def fig_total(med, stage="total", name="fig_b_total_vs_suporte.png", title="(b) Tempo total × suporte"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, d in zip(axes.flat, DATASETS):
        for impl in IMPLS:
            sub = med.loc[impl].loc[d][stage].sort_index()
            ax.plot(sub.index, sub.values, marker="o", markersize=5, color=COLORS[impl], label=LABELS[impl])
        ax.set_xscale("log")
        ax.set_yscale("log")
        sups = sorted(med.loc["seq"].loc[d].index)
        ax.set_xticks(sups, [fmt_sup(v) for v in sups], rotation=30, ha="right")
        ax.xaxis.set_minor_formatter(NullFormatter())
        plain_y(ax)
        ax.invert_xaxis()
        ax.set_title(d, loc="left")
        ax.set_xlabel("suporte mínimo (fração, decrescente →)")
        ax.set_ylabel("tempo (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " (mediana de 5, eixos log)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, name)


# (c) barras empilhadas por etapa --------------------------------------------
def fig_stages(med):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, d in zip(axes.flat, DATASETS):
        s = MID[d]
        bottom = np.zeros(len(IMPLS))
        for k, st in enumerate(STAGES):
            vals = np.array([med.loc[(i, d, s)].get(st, 0.0) if (i, d, s) in med.index else 0 for i in IMPLS])
            vals = np.nan_to_num(vals)
            ax.bar(range(len(IMPLS)), vals, 0.6, bottom=bottom, color=SERIES[k], edgecolor="#fcfcfb", linewidth=1,
                   label=STAGE_LABELS[st])
            bottom += vals
        ax.set_xticks(range(len(IMPLS)), ["seq", "estática", "dinâmica"])
        ax.set_title(f"{d} (minsup {fmt_sup(s)})", loc="left")
        ax.set_ylabel("tempo (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("(c) Tempo por etapa (mediana de 5)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_c_etapas.png")


# (d) distribuição do tempo por bloco ----------------------------------------
def fig_blocks():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    rows = []
    rng = np.random.default_rng(0)
    for ax, d in zip(axes.flat, DATASETS):
        for k, impl in enumerate(["gpu_static", "gpu_dynamic"]):
            path = os.path.join(RES, "blocks", f"{impl}_{d}.csv")
            if not os.path.exists(path):
                continue
            b = pd.read_csv(path)
            b = b[b.launch == 0]
            t = b.busy_ms.clip(lower=1e-3).values
            ratio = t.max() / t.mean()
            rows.append((d, impl, len(t), t.mean(), t.max(), ratio))
            y = k + rng.uniform(-0.25, 0.25, len(t))
            ax.scatter(t, y, s=8, color=COLORS[impl], alpha=0.6, linewidths=0)
            ax.plot([t.max()], [k], marker="|", markersize=18, color="#0b0b0b")
            ax.text(t.max(), k + 0.32, f"máx/média {ratio:.1f}", fontsize=8, ha="right", color="#52514e")
        ax.set_xscale("log")
        ax.set_yticks([0, 1], ["estática", "dinâmica"])
        ax.set_ylim(-0.6, 1.7)
        ax.set_title(f"{d} (minsup {fmt_sup(MID[d])})", loc="left")
        ax.set_xlabel("tempo ocupado por bloco (ms, log)")
    fig.suptitle("(d) Distribuição do tempo por bloco: estática (1 bloco por item) × dinâmica (256 blocos persistentes)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_d_blocos.png")
    return pd.DataFrame(rows, columns=["dataset", "impl", "blocos", "media_ms", "max_ms", "max_sobre_media"])


# (e) efeito do grain -----------------------------------------------------------
def fig_grain(med):
    path = os.path.join(RES, "grain.csv")
    if not os.path.exists(path):
        return None
    g = pd.read_csv(path)
    g = g[g.valido == 1]
    stats = g.groupby(["dataset", "minsup", "grain"]).mineracao_ms.agg(["median", "std"]).reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, d in zip(axes.flat, DATASETS):
        sub = stats[stats.dataset == d]
        x = sub.grain.replace(0, 1).values  # grain 0 desenhado em 1 no eixo log
        ax.errorbar(x, sub["median"], yerr=sub["std"], marker="o", markersize=5, color=COLORS["gpu_dynamic"],
                    label=LABELS["gpu_dynamic"], capsize=3)
        s = MID[d]
        if ("gpu_static", d, s) in med.index:
            ax.axhline(med.loc[("gpu_static", d, s), "mineracao"], color=COLORS["gpu_static"], linestyle="--",
                       linewidth=1.5, label=LABELS["gpu_static"])
        if ("seq", d, s) in med.index:
            ax.axhline(med.loc[("seq", d, s), "mineracao"], color=COLORS["seq"], linestyle=":", linewidth=1.5,
                       label=LABELS["seq"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 16, 64, 256, 1024, 4096], ["0", "16", "64", "256", "1024", "4096"])
        plain_y(ax)
        ax.set_title(f"{d} (minsup {fmt_sup(s)})", loc="left")
        ax.set_xlabel("grain (itens da base condicional)")
        ax.set_ylabel("tempo de mineração (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("(e) Efeito do --grain na mineração dinâmica (mediana ± desvio de 5, eixos log)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig_e_grain.png")
    return stats


# Tabelas ------------------------------------------------------------------------
def tables(raw, runs, med, sd, blocks, grain):
    out = []
    v = runs.groupby("impl").valido.agg(["sum", "count"])
    out.append("## Validade das medidas\n")
    out.append("| Implementação | válidas / total |\n| --- | ---: |")
    for impl in IMPLS:
        if impl in v.index:
            out.append(f"| {LABELS[impl]} | {int(v.loc[impl, 'sum'])} / {int(v.loc[impl, 'count'])} |")

    meta = runs[runs.valido == 1].groupby(["impl", "dataset", "minsup"]).median(numeric_only=True)

    out.append("\n## Construção da FP-tree (6.3), ms\n")
    out.append("| Dataset | minsup | nós | seq | GPU | GPU sem sort | speedup | speedup sem sort | sort (% GPU) |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        for s in sorted(med.loc["seq"].loc[d].index, reverse=True):
            a, b = ("seq", d, s), ("gpu_static", d, s)
            seq, gpu, srt = med.loc[a, "construcao"], med.loc[b, "construcao"], med.loc[b, "construcao_sort"]
            nos = int(meta.loc[b, "nos"]) if b in meta.index else 0
            out.append(f"| {d} | {fmt_sup(s)} | {intf(nos)} | {ms_fmt(seq, sd.loc[a, 'construcao'])} | "
                       f"{ms_fmt(gpu, sd.loc[b, 'construcao'])} | {ms_fmt(gpu - srt)} | {xf(seq / gpu)} | "
                       f"{xf(seq / (gpu - srt))} | {100 * srt / gpu:.0f} % |")

    out.append("\n### Subetapas da construção na GPU (mediana, ms)\n")
    out.append("| Dataset | minsup | " + " | ".join(s.replace("construcao_", "") for s in SUBSTAGES) + " |")
    out.append("| --- | --- | " + " | ".join("---:" for _ in SUBSTAGES) + " |")
    for d in DATASETS:
        for s in sorted(med.loc["seq"].loc[d].index, reverse=True):
            b = ("gpu_static", d, s)
            out.append(f"| {d} | {fmt_sup(s)} | " + " | ".join(ms_fmt(med.loc[b, k]) for k in SUBSTAGES) + " |")

    out.append("\n## Mineração (6.4), ms\n")
    out.append("| Dataset | minsup | itemsets | seq | GPU estática | GPU dinâmica | din./seq | din./est. |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        for s in sorted(med.loc["seq"].loc[d].index, reverse=True):
            r = {i: (med.loc[(i, d, s), "mineracao"], sd.loc[(i, d, s), "mineracao"]) for i in IMPLS}
            n = int(meta.loc[("seq", d, s), "itemsets"])
            out.append(f"| {d} | {fmt_sup(s)} | {intf(n)} | " + " | ".join(ms_fmt(*r[i]) for i in IMPLS) +
                       f" | {xf(r['seq'][0] / r['gpu_dynamic'][0])} | {xf(r['gpu_static'][0] / r['gpu_dynamic'][0])} |")

    out.append("\n## Tempo total ponta a ponta (6.6), ms\n")
    out.append("`total` inclui leitura e escrita canônica (formatação e ordenação das linhas, igual nas três "
               "implementações). `sem transf.` desconta H2D e D2H (o D2H inclui a montagem do CSR de itemsets no "
               "host). `núcleo` = contagem + pré-processamento + construção + mineração.\n")
    out.append("| Dataset | minsup | seq total | est. total | din. total | din. sem transf. | seq núcleo | din. núcleo "
               "| speedup núcleo |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    core = ["contagem", "preprocessamento", "construcao", "mineracao"]
    for d in DATASETS:
        for s in sorted(med.loc["seq"].loc[d].index, reverse=True):
            a, b, c = ("seq", d, s), ("gpu_static", d, s), ("gpu_dynamic", d, s)
            no_tr = med.loc[c, "total"] - med.loc[c, "transferencia_h2d"] - med.loc[c, "transferencia_d2h"]
            cs, cd = med.loc[a, core].sum(), med.loc[c, core].sum()
            out.append(f"| {d} | {fmt_sup(s)} | {ms_fmt(med.loc[a, 'total'], sd.loc[a, 'total'])} | "
                       f"{ms_fmt(med.loc[b, 'total'], sd.loc[b, 'total'])} | "
                       f"{ms_fmt(med.loc[c, 'total'], sd.loc[c, 'total'])} | {ms_fmt(no_tr)} | {ms_fmt(cs)} | "
                       f"{ms_fmt(cd)} | {xf(cs / cd)} |")

    out.append("\n## Memória e estouros (6.7)\n")
    out.append("Pico de memória da GPU usada pelo processo (inclui as arenas e o pool reservados; a reserva é "
               "fixa: até 35 % da memória livre para arenas e 15 % para o pool).\n")
    out.append("| Dataset | minsup | pico construção (MB) | pico est. (MB) | pico din. (MB) | estouros est. | "
               "estouros din. | tarefas enfileiradas (din.) |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        for s in sorted(med.loc["seq"].loc[d].index, reverse=True):
            b, c = ("gpu_static", d, s), ("gpu_dynamic", d, s)
            if b not in meta.index or c not in meta.index:
                continue
            out.append(f"| {d} | {fmt_sup(s)} | {meta.loc[b, 'mem_construcao_MB']:.0f} | "
                       f"{meta.loc[b, 'mem_gpu_pico_MB']:.0f} | {meta.loc[c, 'mem_gpu_pico_MB']:.0f} | "
                       f"{meta.loc[b, 'estouros']:.0f} | {meta.loc[c, 'estouros']:.0f} | "
                       f"{intf(meta.loc[c, 'enfileiradas'])} |")

    out.append("\n## Balanceamento (6.5): tempo do bloco mais lento / tempo médio\n")
    out.append("| Dataset | minsup | estática | dinâmica (grain 64) |\n| --- | --- | ---: | ---: |")
    for d in DATASETS:
        for s in sorted(med.loc["seq"].loc[d].index, reverse=True):
            b, c = ("gpu_static", d, s), ("gpu_dynamic", d, s)
            if b in meta.index and c in meta.index:
                out.append(f"| {d} | {fmt_sup(s)} | {xf(meta.loc[b, 'desbalanceamento'])} | "
                           f"{xf(meta.loc[c, 'desbalanceamento'], 2)} |")
    if blocks is not None and len(blocks):
        out.append("\nDistribuição detalhada (suporte intermediário, `results/blocks/`):\n")
        out.append("| Dataset | impl | blocos | média (ms) | máx (ms) | máx/média |\n| --- | --- | ---: | ---: | ---: | ---: |")
        for _, r in blocks.iterrows():
            out.append(f"| {r.dataset} | {LABELS[r.impl]} | {r.blocos} | {ms_fmt(r.media_ms)} | {ms_fmt(r.max_ms)} | "
                       f"{xf(r.max_sobre_media, 2)} |")

    if grain is not None:
        out.append("\n## Efeito do grain (6.4), mineração em ms (mediana ± desvio)\n")
        gs = sorted(grain.grain.unique())
        out.append("| Dataset | minsup | " + " | ".join(f"g={g}" for g in gs) + " |")
        out.append("| --- | --- | " + " | ".join("---:" for _ in gs) + " |")
        for d in DATASETS:
            sub = grain[grain.dataset == d].set_index("grain")
            if not len(sub):
                continue
            best = sub["median"].idxmin()
            cells = []
            for g in gs:
                c = ms_fmt(sub.loc[g, "median"], sub.loc[g, "std"]) if g in sub.index else "—"
                cells.append(f"**{c}**" if g == best else c)
            out.append(f"| {d} | {fmt_sup(MID[d])} | " + " | ".join(cells) + " |")

    path = os.path.join(RES, "tables.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("gravado", path)


def main():
    raw, runs = load()
    med, sd = stage_stats(raw)
    fig_construction(med)
    fig_total(med)
    fig_total(med, "mineracao", "fig_b2_mineracao_vs_suporte.png", "(b') Tempo de mineração × suporte")
    fig_stages(med)
    blocks = fig_blocks()
    grain = fig_grain(med)
    tables(raw, runs, med, sd, blocks, grain)


if __name__ == "__main__":
    main()
