#!/usr/bin/env python3
"""Figuras da apresentação de resultados parciais (só dados do sequencial).

uso: slides_figs.py <raw_comum.csv> <raw_baixo.csv> <pasta_saida>

Gera custo_fases.png: fração do tempo sequencial em cada fase, um par de
barras por dataset (suporte comum | suporte baixo), com o tempo total anotado.
"""
import os
import sys

import pandas as pd

from plotstyle import SERIES, setup

COMUM = {"chess": 0.6, "mushroom": 0.1, "accidents": 0.4, "T10I4D100K": 0.002, "retail": 0.002, "kosarak": 0.002}
BAIXO = {"chess": 0.35, "mushroom": 0.01, "accidents": 0.1, "T10I4D100K": 0.00003, "retail": 0.00003,
         "kosarak": 0.00085}
FASES = [("leitura", "leitura (CPU)"), ("contagem", "contagem"), ("preprocessamento", "pré-processamento"),
         ("construcao", "construção da FP-tree"), ("mineracao", "mineração"), ("escrita", "escrita (CPU)")]
# Leitura e escrita (fases que ficam na CPU) em cinzas; as do núcleo na paleta categórica.
CORES = {"leitura": "#b5b4ad", "contagem": SERIES[3], "preprocessamento": SERIES[2], "construcao": SERIES[0],
         "mineracao": SERIES[1], "escrita": "#6f6e69"}


def medianas(path, sel):
    r = pd.read_csv(path)
    r = r[(r.valido == 1) & (r.impl == "seq")]
    out = {}
    for d, s in sel.items():
        x = r[(r.dataset == d) & ((r.minsup - s).abs() < 1e-12)]
        out[d] = x.groupby("etapa").tempo_ms.median()
    return out


def fmt_tempo(ms):
    if ms >= 1000:
        return f"{ms / 1000:.1f} s".replace(".", ",")
    return f"{ms:.0f} ms"


def sup(s):
    t = f"{s:.6f}".rstrip("0").rstrip(".") if s < 0.001 else f"{s:g}"
    return t.replace(".", ",")


def main():
    raw_c, raw_b, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    plt = setup()
    plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "font.family": "DejaVu Sans"})
    mc, mb = medianas(raw_c, COMUM), medianas(raw_b, BAIXO)
    ds = list(COMUM)
    fig, ax = plt.subplots(figsize=(16.64, 6.2))
    w = 0.36
    for i, d in enumerate(ds):
        for j, (m, s) in enumerate([(mc[d], COMUM[d]), (mb[d], BAIXO[d])]):
            x = i + (j - 0.5) * (w + 0.06)
            tot = sum(m[f] for f, _ in FASES)
            base = 0.0
            for f, lab in FASES:
                v = m[f] / tot * 100
                ax.bar(x, v, w, bottom=base, color=CORES[f], edgecolor="#fcfcfb", linewidth=1.5,
                       label=lab if (i == 0 and j == 0) else None)
                base += v
            ax.text(x, 102, fmt_tempo(m["total"]), ha="center", va="bottom", fontsize=16, color="#0b0b0b")
            ax.text(x, -4, f"{'comum' if j == 0 else 'baixo'}\n{sup(s)}", ha="center", va="top", fontsize=14,
                    color="#52514e")
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([])
    for i, d in enumerate(ds):
        ax.text(i, -21, d, ha="center", va="top", fontsize=19, color="#0b0b0b", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.6, len(ds) - 0.4)
    ax.set_ylabel("% do tempo total (sequencial)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0.0, 1, 0.92))
    fig.subplots_adjust(bottom=0.24)
    path = os.path.join(out_dir, "custo_fases.png")
    fig.savefig(path, dpi=200, facecolor="#fbfbf8")
    print("gravado", path)


if __name__ == "__main__":
    main()
