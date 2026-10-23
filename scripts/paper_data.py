#!/usr/bin/env python3
"""Números e tabelas do artigo (paper/generated/), calculados dos CSVs de results/.

Nada no texto do artigo é transcrito à mão: cada número citado é uma macro
\\V{chave} definida em paper/generated/numbers.tex. Também imprime um resumo
para conferência.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import CONFIGS, MID, RESULTS_DIR, ROOT, THRESHOLD_SET  # noqa: E402

RES = RESULTS_DIR
MICRO = os.path.join(ROOT, "results", "micro")
OUT = os.path.join(ROOT, "paper", "generated")
DATASETS = ["chess", "mushroom", "accidents", "T10I4D100K", "retail", "kosarak"]
DSNAME = {"T10I4D100K": "T10I4D100K"}
IMPLS = ["seq", "gpu_static", "gpu_dynamic"]
KIND = {"chess": "dense", "mushroom": "dense", "accidents": "dense, large", "T10I4D100K": "sparse, synthetic",
        "retail": "sparse, real", "kosarak": "sparse, large"}
CORE = ["contagem", "preprocessamento", "construcao", "mineracao"]
PEAK_BW = 1008.0      # GB/s, RTX 4090 (especificação)
SMEM_BINS = 12 * 1024  # contadores por faixa no histograma v2
N_SLOTS = 256

vals = {}


def put(key, v):
    vals[key] = v


def f1(x):
    return f"{x:.1f}"


def fx(x, nd=1):
    return f"{x:.{nd}f}$\\times$"


def ms(x):
    if pd.isna(x):
        return "--"
    if x >= 1000:
        return f"{x:,.0f}".replace(",", "\\,")
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def ipm(m, s):
    return f"{ms(m)}\\,$\\pm$\\,{ms(s)}"


def big(n):
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}\\,M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}\\,k"
    return f"{n:.0f}"


def sup(s):
    t = f"{s:.6f}".rstrip("0").rstrip(".") if s < 0.001 else f"{s:g}"
    return t


def key(*parts):
    return "".join((sup(p) if isinstance(p, float) else str(p)).replace(".", "p").replace("_", "") for p in parts)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = pd.read_csv(os.path.join(RES, "raw.csv"))
    runs = pd.read_csv(os.path.join(RES, "runs.csv"))
    grain = pd.read_csv(os.path.join(RES, "grain.csv"))
    cfg = pd.read_csv(os.path.join(MICRO, f"config_stats_{THRESHOLD_SET}.csv"))
    micro = pd.read_csv(os.path.join(MICRO, "microbench.csv"), header=None,
                        names=["dataset", "ntx", "nitems", "metric", "value"])
    pcie = pd.read_csv(os.path.join(MICRO, "pcie.csv"), header=None, names=["gpu", "metric", "value"])
    cb = pd.read_csv(os.path.join(MICRO, "countbench.csv"))

    assert (raw.valido == 1).all() and (runs.valido == 1).all(), "medidas inválidas"
    put("nruns", len(runs))
    put("ngrainruns", len(grain))
    v = raw[raw.valido == 1]
    g = v.groupby(["impl", "dataset", "minsup", "etapa"]).tempo_ms
    med, sd = g.median().unstack("etapa"), g.std().unstack("etapa")
    meta = runs.groupby(["impl", "dataset", "minsup"]).median(numeric_only=True)
    cfg = cfg.set_index(["dataset", "minsup"])
    configs = [(d, s) for d in DATASETS for s in sorted(cfg.loc[d].index, reverse=True)]

    # ------------------------------------------------------------------ datasets
    rows = []
    for d in DATASETS:
        c = cfg.loc[d].iloc[0]
        sups = ", ".join(sup(s) for s in sorted(cfg.loc[d].index, reverse=True))
        rows.append(f"{d} & {KIND[d].split(',')[0]} & {int(c.transacoes_orig):,} & {int(c.itens_distintos):,} & "
                    f"{c.tam_medio:.1f} & {int(c.tam_max):,} & {sups} \\\\".replace(",", "{,}", 0))
    with open(os.path.join(OUT, "tab_datasets.tex"), "w") as f:
        f.write("\n".join(r.replace(",", "{,}").replace("{,} ", ", ") for r in rows) + "\n")

    # ---------------------------------------------------- tempos por fase (faixas)
    stage_names = [("leitura", "Read (CPU)"), ("transferencia_h2d", "H2D transfer"), ("contagem", "Support counting"),
                   ("preprocessamento", "Preprocessing"), ("construcao", "FP-tree construction"),
                   ("mineracao", "Mining"), ("transferencia_d2h", "D2H + host assembly"),
                   ("escrita", "Canonical output (CPU)")]
    rows = []
    for st, label in stage_names:
        seqs, dyns, sps = [], [], []
        for d, s in configs:
            a = med.loc[("seq", d, s)].get(st, np.nan)
            b = med.loc[("gpu_dynamic", d, s)].get(st, np.nan)
            seqs.append(a)
            dyns.append(b)
            if st in CORE:
                sps.append(a / b)
        seqs, dyns = np.array(seqs, float), np.array(dyns, float)
        rng = lambda x: f"{ms(np.nanmin(x))}--{ms(np.nanmax(x))}" if not np.all(np.isnan(x)) else "--"
        sp = f"{min(sps):.1f}--{max(sps):.0f}$\\times$" if sps else "--"
        share = []
        for d, s in configs:
            t = med.loc[("gpu_dynamic", d, s), "total"]
            share.append(med.loc[("gpu_dynamic", d, s)].get(st, 0) / t * 100)
        rows.append(f"{label} & {rng(seqs) if st not in ('transferencia_h2d', 'transferencia_d2h') else '--'} & "
                    f"{rng(dyns)} & {sp} & {min(share):.0f}--{max(share):.0f}\\,\\% \\\\")
        if st in CORE:
            put(key("spmin", st), f"{min(sps):.1f}")
            put(key("spmax", st), f"{max(sps):.0f}")
    with open(os.path.join(OUT, "tab_phases.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ------------------------------------------------------------ contagem v1/v2
    cbp = cb.pivot(index="dataset", columns="kernel", values="ms")
    rows = []
    best = []
    for d in DATASETS:
        c = cfg.loc[d].iloc[0]
        nnz = c.itens_orig
        n_ids = int(c.itens_distintos)
        bands = math.ceil((n_ids + 1) / SMEM_BINS)
        v1, v2 = cbp.loc[d, "v1"], cbp.loc[d, "v2"]
        bw2 = nnz * 4 * bands / (v2 * 1e-3) / 1e9
        tmin = nnz * 4 / (PEAK_BW * 1e9) * 1e3
        best.append(v1 / v2)
        rows.append(f"{d} & {big(nnz)} & {bands} & {ms(v1 * 1000)} & {ms(v2 * 1000)} & {fx(v1 / v2)} & "
                    f"{ms(tmin * 1000)} & {bw2:.0f} \\\\")
        put(key("cnt", d, "sp"), f"{v1 / v2:.1f}")
        put(key("cnt", d, "tmin"), f"{tmin * 1000:.0f}")
        put(key("cnt", d, "vtwo"), f"{v2 * 1000:.0f}")
    larger = [d for d in DATASETS if cfg.loc[d].iloc[0].itens_orig > 5e5]
    ratios = []
    for d in larger:
        c = cfg.loc[d].iloc[0]
        ratios.append(cbp.loc[d, "v2"] / (c.itens_orig * 4 / (PEAK_BW * 1e9) * 1e3))
    put("cntratiomin", f"{min(ratios):.1f}")
    put("cntratiomax", f"{max(ratios):.1f}")
    put("cntfloor", f"{cbp.v2.min() * 1000:.0f}")
    put("nlarger", str(len(larger)))
    # Pré-processamento: ~6 passadas de 4 bytes por item (ler ids, escrever ranks,
    # flags, scan das flags, scatter) na banda prática.
    kos = cfg.loc["kosarak"].iloc[0]
    put("premins", f"{kos.itens_orig * 4 * 6 / (920e9) * 1e3:.1f}")
    put("cntspmin", f"{min(best):.1f}")
    put("cntspmax", f"{max(best):.0f}")
    with open(os.path.join(OUT, "tab_count.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------- construção
    radix = micro[micro.metric == "radix64_pairs_ms"].set_index("dataset").value
    d2d = micro[micro.metric == "d2d_GBps"].value.median()
    put("dtodbw", f"{d2d:.0f}")
    put("dtodpct", f"{100 * d2d / PEAK_BW:.0f}")
    rows, sub_rows = [], []
    sps, sps_ns, sortfrac = [], [], []
    for d, s in configs:
        a, b = ("seq", d, s), ("gpu_static", d, s)
        seq, gpu = med.loc[a, "construcao"], med.loc[b, "construcao"]
        srt = med.loc[b, "construcao_sort"]
        ideal = gpu - srt + radix[d]
        nodes = int(cfg.loc[(d, s), "nos"])
        sps.append(seq / gpu)
        sps_ns.append(seq / (gpu - srt))
        sortfrac.append(srt / gpu)
        put(key("cons", d, s, "sp"), f"{seq / gpu:.1f}")
        if s == max(cfg.loc[d].index) or s == min(cfg.loc[d].index):
            rows.append(f"{d} & {sup(s)} & {big(nodes)} & {ipm(seq, sd.loc[a, 'construcao'])} & "
                        f"{ipm(gpu, sd.loc[b, 'construcao'])} & {fx(seq / gpu)} & {100 * srt / gpu:.0f}\\,\\% & "
                        f"{fx(seq / ideal)} \\\\")
    big_sps = [x for (d, s), x in zip(configs, sps) if cfg.loc[(d, s), "nos"] > 4e5]
    small_sps = [x for (d, s), x in zip(configs, sps) if cfg.loc[(d, s), "nos"] < 1e5]
    put("consspbigmin", f"{min(big_sps):.0f}")
    put("consspbigmax", f"{max(big_sps):.0f}")
    put("consspsmallmin", f"{min(small_sps):.1f}")
    put("consspsmallmax", f"{max(small_sps):.1f}")
    big_ns = [x for (d, s), x in zip(configs, sps_ns) if cfg.loc[(d, s), "nos"] > 4e5]
    ideal_all = [med.loc[("seq", d, s), "construcao"] / (med.loc[("gpu_static", d, s), "construcao"]
                 - med.loc[("gpu_static", d, s), "construcao_sort"] + radix[d]) for d, s in configs]
    put("idealsortmax", f"{max(ideal_all):.0f}")
    put("consnsmin", f"{min(big_ns):.0f}")
    put("consnsmax", f"{max(big_ns):.0f}")
    put("sortfracmin", f"{100 * min(sortfrac):.0f}")
    put("sortfracmax", f"{100 * max(sortfrac):.0f}")
    with open(os.path.join(OUT, "tab_construction.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # Banda efetiva das subetapas (tráfego mínimo estimado / tempo) no maior caso.
    d, s = "kosarak", min(CONFIGS["kosarak"])  # maior árvore do conjunto
    c = cfg.loc[(d, s)]
    n, nnz, N = c.transacoes, c.itens_pre, c.nos
    L = math.ceil(math.log2(n)) + 1
    b = med.loc[("gpu_static", d, s)]
    traffic = {
        "construcao_lcp": (nnz * 4 * 2 + n * 16, "LCP"),
        "construcao_nos": (N * 16 + n * 12, "Node write"),
        "construcao_sparse": (L * n * 12, "Sparse table"),
        "construcao_pais": (N * 16, "Parents"),
        "construcao_contadores": (N * 16, "Counts"),
    }
    rows = []
    for st, (byt, label) in traffic.items():
        bw = byt / (b[st] * 1e-3) / 1e9
        rows.append(f"{label} & {byt / 1e6:.0f} & {ms(b[st])} & {bw:.0f} & {100 * bw / d2d:.0f}\\,\\% \\\\")
        put(key("bw", st), f"{bw:.0f}")
    mids = [float(vals[key("bw", k)]) for k in ("construcao_lcp", "construcao_pais", "construcao_contadores")]
    put("bwmidmin", f"{min(mids):.0f}")
    put("bwmidmax", f"{max(mids):.0f}")
    put("ptcmax", ms(max(b["construcao_pais"], b["construcao_contadores"])))
    with open(os.path.join(OUT, "tab_bandwidth.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    put("bwnodes", big(N))
    put("bwsup", sup(s))
    put("bwtx", big(n))

    # ------------------------------------------------------------------ mineração
    rows = []
    dyn_seq, dyn_st, st_slower = [], [], 0
    for d, s in configs:
        r = {i: (med.loc[(i, d, s), "mineracao"], sd.loc[(i, d, s), "mineracao"]) for i in IMPLS}
        n_sets = int(meta.loc[("seq", d, s), "itemsets"])
        a, b2 = r["seq"][0] / r["gpu_dynamic"][0], r["gpu_static"][0] / r["gpu_dynamic"][0]
        dyn_seq.append(a)
        dyn_st.append(b2)
        st_slower += r["gpu_static"][0] > r["seq"][0]
        p1 = med.loc[("gpu_dynamic", d, s), "mineracao_passada1"] / r["gpu_dynamic"][0]
        rows.append(f"{d} & {sup(s)} & {big(n_sets)} & {ipm(*r['seq'])} & {ipm(*r['gpu_static'])} & "
                    f"{ipm(*r['gpu_dynamic'])} & {fx(a)} & {fx(b2)} & "
                    f"{meta.loc[('gpu_static', d, s), 'desbalanceamento']:.1f} & "
                    f"{meta.loc[('gpu_dynamic', d, s), 'desbalanceamento']:.2f} \\\\")
        put(key("mine", d, s, "dynseq"), f"{a:.1f}")
        put(key("mine", d, s, "dynst"), f"{b2:.1f}")
    put("minedynseqmin", f"{min(dyn_seq):.1f}")
    put("minedynseqmax", f"{max(dyn_seq):.0f}")
    put("minedynstmin", f"{min(dyn_st):.1f}")
    put("minedynstmax", f"{max(dyn_st):.0f}")
    put("staticslower", str(st_slower))
    put("nconfigs", str(len(configs)))
    with open(os.path.join(OUT, "tab_mining.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # Desbalanceamento: faixas.
    st_imb = [meta.loc[("gpu_static", d, s), "desbalanceamento"] for d, s in configs]
    dy_imb = [meta.loc[("gpu_dynamic", d, s), "desbalanceamento"] for d, s in configs]
    put("imbstmin", f"{min(st_imb):.1f}")
    put("imbstmax", f"{max(st_imb):,.0f}".replace(",", "{,}"))
    put("imbdynlow", str(sum(x <= 2.35 for x in dy_imb)))
    put("imbdynmax", f"{max(dy_imb):.1f}")

    # Passada 1 (contagem) como fração da mineração.
    p1f = [med.loc[(i, d, s), "mineracao_passada1"] / med.loc[(i, d, s), "mineracao"]
           for i in ["gpu_static", "gpu_dynamic"] for d, s in configs]
    put("passonemin", f"{100 * min(p1f):.0f}")
    put("passonemax", f"{100 * max(p1f):.0f}")
    put("passonemed", f"{100 * np.median(p1f):.0f}")

    # Eficiência de balanceamento a partir da instrumentação por bloco.
    rows = []
    for d in DATASETS:
        line = [d, sup(MID[d])]
        for impl in ["gpu_static", "gpu_dynamic"]:
            bk = pd.read_csv(os.path.join(RES, "blocks", f"{impl}_{d}.csv"))
            bk = bk[bk.launch == 0]
            total_busy, mx = bk.busy_ms.sum(), bk.busy_ms.max()
            ideal = total_busy / N_SLOTS
            if impl == "gpu_static":
                # Posse fixa: o makespan não fica abaixo do maior bloco nem do trabalho / 256.
                lb = max(mx, ideal)
                line += [str(len(bk)), ms(mx), ms(ideal), f"{100 * ideal / lb:.0f}\\,\\%"]
                put(key("blk", d, "stmax"), ms(mx))
                put(key("blk", d, "steff"), f"{100 * ideal / lb:.0f}")
            else:
                line += [ms(mx), ms(ideal), f"{100 * ideal / mx:.0f}\\,\\%"]
                put(key("blk", d, "dyneff"), f"{100 * ideal / mx:.0f}")
        rows.append(" & ".join(line) + " \\\\")
    effs = {}
    for d in DATASETS:
        effs[d] = (float(vals[key("blk", d, "steff")]), float(vals[key("blk", d, "dyneff")]))
    put("steffmin", f"{min(e[0] for e in effs.values()):.0f}")
    put("steffmax", f"{max(e[0] for e in effs.values()):.0f}")
    mid = [effs[d][1] for d in ("mushroom", "retail", "T10I4D100K")]
    put("dyneffmidmin", f"{min(mid):.0f}")
    put("dyneffmidmax", f"{max(mid):.0f}")
    put("dynbalgainmin", f"{100 / max(mid):.2f}")
    put("dynbalgainmax", f"{100 / min(mid):.1f}")
    put("accbalgain", f"{100 / effs['accidents'][1]:.1f}")
    with open(os.path.join(OUT, "tab_balance.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ---------------------------------------------------------------------- grain
    gs = grain[grain.valido == 1].groupby(["dataset", "grain"]).mineracao_ms.median().unstack("grain")
    rows = []
    for d in DATASETS:
        best = gs.loc[d].idxmin()
        cells = [(f"\\textbf{{{ms(gs.loc[d, gg])}}}" if gg == best else ms(gs.loc[d, gg])) for gg in gs.columns]
        rows.append(f"{d} & " + " & ".join(cells) + " \\\\")
        put(key("grain", d, "best"), str(best))
        put(key("grain", d, "ratiohigh"), f"{gs.loc[d, 4096] / gs.loc[d].min():.0f}")
    with open(os.path.join(OUT, "tab_grain.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    put("grainkosarakfour", ms(gs.loc["kosarak", 4096]))
    put("grainkosaraksixtyfour", ms(gs.loc["kosarak", 64]))
    put("grainchesszero", ms(gs.loc["chess", 0]))
    put("grainchesssixtyfour", ms(gs.loc["chess", 64]))

    # ------------------------------------------------- ponta a ponta e Amdahl
    rows = []
    core_sp, tot_sp = [], []
    for d, s in configs:
        a, c2 = med.loc[("seq", d, s)], med.loc[("gpu_dynamic", d, s)]
        cs, cd = a[CORE].sum(), c2[CORE].sum()
        cpu_fixed = a["leitura"] + a["escrita"]
        amdahl = a["total"] / cpu_fixed
        frac = cpu_fixed / a["total"]
        gpu_cpu_share = (c2["leitura"] + c2["escrita"]) / c2["total"]
        core_sp.append(cs / cd)
        tot_sp.append(a["total"] / c2["total"])
        rows.append(f"{d} & {sup(s)} & {ms(a['total'])} & {ms(c2['total'])} & {fx(a['total'] / c2['total'], 2)} & "
                    f"{fx(cs / cd)} & {100 * frac:.0f}\\,\\% & {fx(amdahl, 2)} & {100 * gpu_cpu_share:.0f}\\,\\% \\\\")
        put(key("e2e", d, s, "tot"), f"{a['total'] / c2['total']:.1f}")
        put(key("e2e", d, s, "amdahl"), f"{amdahl:.2f}")
    put("coremin", f"{min(core_sp):.1f}")
    put("coremax", f"{max(core_sp):.0f}")
    put("totmin", f"{min(tot_sp):.2f}")
    put("totmax", f"{max(tot_sp):.1f}")
    with open(os.path.join(OUT, "tab_e2e.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    # Teto de Amdahl e fração CPU-only nos densos pequenos (chess, mushroom).
    dense = [(d, s) for d, s in configs if d in ("chess", "mushroom")]
    am, fs, fg = [], [], []
    for d, s in dense:
        a, c2 = med.loc[("seq", d, s)], med.loc[("gpu_dynamic", d, s)]
        am.append(a["total"] / (a["leitura"] + a["escrita"]))
        fs.append((a["leitura"] + a["escrita"]) / a["total"] * 100)
        fg.append((c2["leitura"] + c2["escrita"]) / c2["total"] * 100)
    put("amdahldensemin", f"{min(am):.2f}")
    put("amdahldensemax", f"{max(am):.2f}")
    put("cpusharedensemin", f"{min(fs):.0f}")
    put("cpusharedensemax", f"{max(fs):.0f}")
    put("gpucpusharedensemin", f"{min(fg):.0f}")
    put("gpucpusharedensemax", f"{max(fg):.0f}")
    d2h_share = [med.loc[("gpu_dynamic", d, s), "transferencia_d2h"] / med.loc[("gpu_dynamic", d, s), "total"] * 100
                 for d, s in configs]
    put("dtohsharemax", f"{max(d2h_share):.0f}")
    put("readseqmax", ms(max(med.loc[("seq", d, s), "leitura"] for d, s in configs)))

    # Escrita canônica: custo por itemset.
    per_set = []
    for d, s in configs:
        nset = meta.loc[("seq", d, s), "itemsets"]
        if nset > 1e6:
            per_set.append(med.loc[("seq", d, s), "escrita"] * 1e6 / nset)
    put("writensmin", f"{min(per_set):.0f}")
    put("writensmax", f"{max(per_set):.0f}")
    m01 = med.loc[("gpu_dynamic", "mushroom", min(CONFIGS["mushroom"]))]
    put("mushwrite", f"{m01['escrita'] / 1000:.0f}")
    put("mushtotal", f"{m01['total'] / 1000:.0f}")

    # PCIe.
    pc = pcie.set_index(["gpu", "metric"]).value
    put("pcieA", f"{pc[('gpu0', 'h2d_pinned_GBps')]:.1f}")
    put("pcieApage", f"{pc[('gpu0', 'h2d_pageable_GBps')]:.1f}")
    put("pcieB", f"{pc[('gpu1', 'h2d_pinned_GBps')]:.1f}")
    put("pcieBpage", f"{pc[('gpu1', 'h2d_pageable_GBps')]:.1f}")
    # Todas as execuções da GPU devem ter rodado na mesma placa (slot x16).
    gpu_runs = runs[runs.impl != "seq"]
    buses = sorted(gpu_runs.pci.dropna().astype(str).unique())
    assert len(buses) == 1 and gpu_runs.pci.notna().all(), f"execuções em mais de uma GPU: {buses}"
    put("gpubus", buses[0].replace("_", "\\_"))
    h = raw[(raw.etapa == "transferencia_h2d") & (raw.dataset == "kosarak") & (raw.impl != "seq")].tempo_ms
    put("htodkosmed", ms(h.median()))
    put("htodkosmin", ms(h.min()))
    put("htodkosmax", ms(h.max()))
    kb = cfg.loc["kosarak"].iloc[0]
    kos_bytes = (kb.itens_orig + kb.transacoes_orig + 1) * 4
    put("htodkosmb", f"{kos_bytes / 1e6:.0f}")
    put("htodkosgbps", f"{kos_bytes / (h.median() * 1e-3) / 1e9:.1f}")
    put("htodkosxfourms", ms(kos_bytes / (pc[('gpu1', 'h2d_pageable_GBps')] * 1e9) * 1e3))

    # Memória, estouros e fila.
    mm = runs.groupby(["impl", "dataset", "minsup"]).median(numeric_only=True)
    put("memconsmax", f"{mm.loc['gpu_static'].mem_construcao_MB.max():.0f}")
    put("mempeakmin", f"{mm.mem_gpu_pico_MB.min() / 1024:.1f}")
    put("mempeakmax", f"{mm.mem_gpu_pico_MB.max() / 1024:.1f}")
    put("ovfmax", f"{mm.estouros.max():.0f}")
    put("ovfconfigs", str(int((mm.loc['gpu_dynamic'].estouros > 0).sum())))
    enq = mm.loc["gpu_dynamic"].enfileiradas
    put("enqmin", big(enq.min()))
    put("enqmax", big(enq.max()))
    put("peakbw", f"{PEAK_BW:.0f}")
    # Contagem dentro do pipeline (inclui alocação) x kernel isolado.
    pc = med.xs("contagem", axis=1)
    put("cntstagemin", ms(pc.loc["gpu_dynamic"].min()))
    put("cntstagemax", ms(pc.loc["gpu_dynamic"].max()))
    pre = med.xs("preprocessamento", axis=1)
    put("prestagemax", ms(pre.loc["gpu_dynamic"].max()))
    put("prestageseqmax", ms(pre.loc["seq"].max()))

    # Contagens de validação e outros fatos fixos do projeto.
    put("ntests", "535")
    put("nstress", "420")
    put("nborgelt", "12")

    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("% Gerado por scripts/paper_data.py. Não editar.\n")
        for k, v in sorted(vals.items()):
            f.write(f"\\defval{{{k}}}{{{v}}}\n")
    print(f"{len(vals)} valores gravados em {OUT}")
    for k in sorted(vals):
        print(f"  {k} = {vals[k]}")


if __name__ == "__main__":
    main()
