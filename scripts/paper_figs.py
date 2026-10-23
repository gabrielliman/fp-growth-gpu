#!/usr/bin/env python3
"""Figuras do artigo (PDF vetorial nas larguras de coluna IEEE) em paper/figures/."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import MID, RESULTS_DIR, ROOT, THRESHOLD_SET  # noqa: E402
from plotstyle import COLORS, SERIES, setup  # noqa: E402

plt = setup()
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.4, "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
COL, FULL = 3.5, 7.16
RES = RESULTS_DIR
MICRO = os.path.join(ROOT, "results", "micro")
OUT = os.path.join(ROOT, "paper", "figures")
DATASETS = ["chess", "mushroom", "accidents", "T10I4D100K", "retail", "kosarak"]
SHORT = {"T10I4D100K": "T10I4D100K"}
LBL = {"seq": "Sequential CPU", "gpu_static": "GPU static", "gpu_dynamic": "GPU dynamic"}
CORE = ["contagem", "preprocessamento", "construcao", "mineracao"]


def plain(v, _pos=None):
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:g}"


def plain_y(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_minor_formatter(NullFormatter())


def sup(s):
    return f"{s:.6f}".rstrip("0").rstrip(".") if s < 0.001 else f"{s:g}"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("gravado", name)


def load():
    raw = pd.read_csv(os.path.join(RES, "raw.csv"))
    g = raw[raw.valido == 1].groupby(["impl", "dataset", "minsup", "etapa"]).tempo_ms
    return g.median().unstack("etapa"), g.std().unstack("etapa")


def fig_count():
    cb = pd.read_csv(os.path.join(MICRO, "countbench.csv")).pivot(index="dataset", columns="kernel",
                                                                          values="ms")
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    x = np.arange(len(DATASETS))
    ax.bar(x - 0.2, cb.loc[DATASETS, "v1"] * 1000, 0.38, color=SERIES[3], label="v1: global atomics")
    ax.bar(x + 0.2, cb.loc[DATASETS, "v2"] * 1000, 0.38, color=SERIES[0], label="v2: shared-memory histogram")
    for i, d in enumerate(DATASETS):
        ax.text(i, cb.loc[d, "v1"] * 1000 * 1.15, f"{cb.loc[d, 'v1'] / cb.loc[d, 'v2']:.1f}×", ha="center",
                fontsize=6, color="#52514e")
    ax.set_yscale("log")
    plain_y(ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 2)
    ax.set_xticks(x, DATASETS, rotation=20, ha="right")
    ax.set_ylabel("kernel time (µs)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    save(fig, "count.pdf")


def fig_construction(med):
    cfg = pd.read_csv(os.path.join(MICRO, f"config_stats_{THRESHOLD_SET}.csv"))
    micro = pd.read_csv(os.path.join(MICRO, "microbench.csv"), header=None,
                        names=["dataset", "ntx", "nitems", "metric", "value"])
    radix = micro[micro.metric == "radix64_pairs_ms"].set_index("dataset").value
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = np.arange(len(DATASETS))
    meas, nosort, ideal = [], [], []
    for d in DATASETS:
        s = cfg[cfg.dataset == d].minsup.min()
        seq = med.loc[("seq", d, s), "construcao"]
        gpu = med.loc[("gpu_static", d, s), "construcao"]
        srt = med.loc[("gpu_static", d, s), "construcao_sort"]
        meas.append(seq / gpu)
        ideal.append(seq / (gpu - srt + radix[d]))
        nosort.append(seq / (gpu - srt))
    ax.bar(x - 0.27, meas, 0.26, color=SERIES[0], label="measured")
    ax.bar(x, ideal, 0.26, color="#86b6ef", label="sort at radix-sort speed")
    ax.bar(x + 0.27, nosort, 0.26, color="#c3c2b7", label="without the sort")
    ax.axhline(1, color="#0b0b0b", linewidth=0.8)
    ax.set_yscale("log")
    plain_y(ax)
    ax.set_xticks(x, DATASETS, rotation=20, ha="right")
    ax.set_ylabel("speedup over sequential construction")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    save(fig, "construction_speedup.pdf")

    # Composição das subetapas (menor suporte de cada dataset).
    subs = [("construcao_sort", "lexicographic sort"), ("construcao_lcp", "LCP"), ("construcao_scan", "scan"),
            ("construcao_nos", "node write"), ("construcao_sparse", "sparse table"), ("construcao_pais", "parents"),
            ("construcao_contadores", "counts"), ("construcao_cabecalho", "header table")]
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    left = np.zeros(len(DATASETS))
    for k, (st, label) in enumerate(subs):
        vals = []
        for d in DATASETS:
            s = cfg[cfg.dataset == d].minsup.min()
            r = med.loc[("gpu_static", d, s)]
            vals.append(r[st] / sum(r[q] for q, _ in subs) * 100)
        ax.barh(range(len(DATASETS)), vals, left=left, color=SERIES[k], edgecolor="white", linewidth=0.6,
                label=label, height=0.7)
        left += np.array(vals)
    ax.set_yticks(range(len(DATASETS)), DATASETS)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of GPU construction time (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=4, frameon=False, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "construction_breakdown.pdf")


def fig_mining(med, sd):
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.3))
    for ax, d in zip(axes.flat, DATASETS):
        for impl in ["seq", "gpu_static", "gpu_dynamic"]:
            m = med.loc[impl].loc[d]["mineracao"].sort_index()
            e = sd.loc[impl].loc[d]["mineracao"].sort_index()
            ax.errorbar(m.index, m.values, yerr=e.values, marker="o", markersize=3, color=COLORS[impl],
                        label=LBL[impl], capsize=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        sups = sorted(med.loc["seq"].loc[d].index)
        # Em kosarak 0,0009 e 0,00085 ficam colados: rotula só um deles.
        labels = ["" if (d == "kosarak" and v == 0.0009) else sup(v) for v in sups]
        ax.set_xticks(sups, labels, rotation=25, ha="right")
        ax.xaxis.set_minor_formatter(NullFormatter())
        plain_y(ax)
        ax.invert_xaxis()
        ax.set_title(d, loc="left")
        ax.set_ylabel("mining time (ms)")
    for ax in axes[1]:
        ax.set_xlabel("minimum support (decreasing →)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0, 1, 0.96))
    save(fig, "mining_vs_support.pdf")


def fig_blocks():
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 2.9))
    rng = np.random.default_rng(0)
    for ax, d in zip(axes.flat, DATASETS):
        for k, impl in enumerate(["gpu_static", "gpu_dynamic"]):
            b = pd.read_csv(os.path.join(RES, "blocks", f"{impl}_{d}.csv"))
            b = b[b.launch == 0]
            t = b.busy_ms.clip(lower=1e-3).values
            y = k + rng.uniform(-0.25, 0.25, len(t))
            ax.scatter(t, y, s=3, color=COLORS[impl], alpha=0.55, linewidths=0, rasterized=True)
            ax.plot([t.mean()], [k], marker="|", markersize=11, color="#0b0b0b", mew=1)
            ax.text(t.max() * 1.25, k, f"{t.max() / t.mean():.1f}", fontsize=6, va="center", color="#0b0b0b")
        ax.set_xscale("log")
        ax.set_xlim(right=ax.get_xlim()[1] * 4)
        ax.set_yticks([0, 1], ["static", "dynamic"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(f"{d} (minsup {sup(MID[d])})", loc="left")
    for ax in axes[1]:
        ax.set_xlabel("busy time per block (ms, log)")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "blocks.pdf")


def fig_grain():
    g = pd.read_csv(os.path.join(RES, "grain.csv"))
    m = g[g.valido == 1].groupby(["dataset", "grain"]).mineracao_ms.median().unstack("grain")
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    xs = [1 if c == 0 else c for c in m.columns]
    for k, d in enumerate(DATASETS):
        ax.plot(xs, m.loc[d] / m.loc[d].min(), marker="o", markersize=3, color=SERIES[k], label=d)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs, [str(c) for c in m.columns])
    ax.xaxis.set_minor_formatter(NullFormatter())
    plain_y(ax)
    ax.set_xlabel("grain (items in the conditional base)")
    ax.set_ylabel("mining time / best")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "grain.pdf")


STAGES = [("leitura", "read"), ("transferencia_h2d", "H2D"), ("contagem", "counting"),
          ("preprocessamento", "preprocessing"), ("construcao", "construction"), ("mineracao", "mining"),
          ("transferencia_d2h", "D2H + assembly"), ("escrita", "canonical output")]


def fig_stages(med):
    fig, axes = plt.subplots(1, 6, figsize=(FULL, 1.9))
    impls = ["seq", "gpu_static", "gpu_dynamic"]
    for ax, d in zip(axes, DATASETS):
        s = MID[d]
        bottom = np.zeros(3)
        for k, (st, label) in enumerate(STAGES):
            vals = np.nan_to_num(np.array([med.loc[(i, d, s)].get(st, 0.0) for i in impls]))
            ax.bar(range(3), vals, 0.65, bottom=bottom, color=SERIES[k], edgecolor="white", linewidth=0.5,
                   label=label)
            bottom += vals
        ax.set_xticks(range(3), ["seq", "stat.", "dyn."])
        ax.set_title(f"{d}\n(minsup {sup(s)})", loc="left", fontsize=7)
        ax.yaxis.set_major_formatter(FuncFormatter(plain))
    axes[0].set_ylabel("time (ms)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=0.4, rect=(0, 0.05, 1, 1))
    save(fig, "stages.pdf")


def fig_amdahl(med):
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = np.arange(len(DATASETS))
    e2e, core, bound = [], [], []
    for d in DATASETS:
        s = MID[d]
        a, c = med.loc[("seq", d, s)], med.loc[("gpu_dynamic", d, s)]
        e2e.append(a["total"] / c["total"])
        core.append(a[CORE].sum() / c[CORE].sum())
        bound.append(a["total"] / (a["leitura"] + a["escrita"]))
    ax.bar(x - 0.27, e2e, 0.26, color=COLORS["gpu_dynamic"], label="end-to-end (measured)")
    ax.bar(x, bound, 0.26, color="#f3b49a", label="Amdahl bound (CPU read+output)")
    ax.bar(x + 0.27, core, 0.26, color=SERIES[0], label="algorithmic core (measured)")
    ax.axhline(1, color="#0b0b0b", linewidth=0.8)
    ax.set_yscale("log")
    plain_y(ax)
    ax.set_xticks(x, DATASETS, rotation=20, ha="right")
    ax.set_ylabel("speedup over sequential")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.set_ylim(0.5, 300)
    save(fig, "amdahl.pdf")


def main():
    med, sd = load()
    fig_count()
    fig_construction(med)
    fig_mining(med, sd)
    fig_blocks()
    fig_grain()
    fig_stages(med)
    fig_amdahl(med)


if __name__ == "__main__":
    main()
