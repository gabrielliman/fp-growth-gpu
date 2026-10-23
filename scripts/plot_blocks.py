#!/usr/bin/env python3
"""Histograma do tempo ocupado por bloco (instrumentação 4.9 / 5.7).

uso: plot_blocks.py saida.png rotulo1=arquivo1.csv [rotulo2=arquivo2.csv ...]
O rótulo "static" ou "dynamic" escolhe a cor da implementação.
"""
import sys

import numpy as np
import pandas as pd

from plotstyle import COLORS, LABELS, setup


def main():
    plt = setup()
    out = sys.argv[1]
    series = [a.split("=", 1) for a in sys.argv[2:]]
    fig, axes = plt.subplots(1, len(series), figsize=(5.2 * len(series), 3.6), squeeze=False)
    for ax, (label, path) in zip(axes[0], series):
        df = pd.read_csv(path)
        df = df[df.launch == 0]  # primeira execução: árvore global
        t = df.busy_ms.values
        key = "gpu_" + df["mode"].iloc[0]
        ax.hist(t, bins=40, color=COLORS[key], edgecolor="#fcfcfb", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("tempo ocupado do bloco (ms)")
        ax.set_ylabel("número de blocos")
        ratio = t.max() / t.mean() if t.mean() > 0 else float("nan")
        ax.set_title(f"{LABELS[key]} — {label}\n{len(t)} blocos, máx/média = {ratio:.1f}", loc="left")
        ax.axvline(t.mean(), color="#0b0b0b", linewidth=1, linestyle="--")
        ax.annotate("média", (t.mean(), ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                    fontsize=8, color="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("gravado", out)


if __name__ == "__main__":
    main()
