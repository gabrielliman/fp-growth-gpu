#!/usr/bin/env python3
"""Gera o relatório detalhado por fase a partir dos CSVs do conjunto de
thresholds escolhido (FPG_SET). Usado para a versão com suportes padrão
(reports/relatorio_fases_suporte_padrao.md); as descrições dos mecanismos são
as mesmas de reports/relatorio_fases.md, e todos os números e leituras vêm dos
dados.

uso: FPG_SET=common python3 scripts/relatorio_fases.py <saida.md>
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import CONFIGS, MID, RESULTS_DIR, ROOT, THRESHOLD_SET  # noqa: E402

DATASETS = ["chess", "mushroom", "accidents", "T10I4D100K", "retail", "kosarak"]
KIND = {"chess": "denso, pequeno", "mushroom": "denso, pequeno", "accidents": "denso, grande",
        "T10I4D100K": "esparso, sintético", "retail": "esparso, real", "kosarak": "esparso, grande"}
CORE = ["contagem", "preprocessamento", "construcao", "mineracao"]
STAGES = ["leitura", "transferencia_h2d", "contagem", "preprocessamento", "construcao", "mineracao",
          "transferencia_d2h", "escrita"]
MICRO = os.path.join(ROOT, "results", "micro")


# ---------------------------------------------------------------- formatação
def num(x, nd=None):
    """Número no formato brasileiro, com casas decimais adaptativas."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    ax = abs(x)
    if nd is None:
        nd = 0 if ax >= 100 else 1 if ax >= 10 else 2 if ax >= 1 else 3 if ax >= 0.01 else 4
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def sp(x):
    if x >= 10:
        return f"{num(x, 0)}×"
    if x < 0.1:
        return f"{num(x, 2)}×"
    return f"{num(x, 1)}×"


def sup(s):
    t = f"{s:.6f}".rstrip("0").rstrip(".") if s < 0.001 else f"{s:g}"
    return t.replace(".", ",")


def big(n):
    n = float(n)
    if n >= 1e6:
        return f"{num(n / 1e6, 2)} M"
    if n >= 1e3:
        return f"{num(n / 1e3, 1)} k"
    return num(n, 0)


def pct(x):
    return f"{x:.0f} %"


def geo(xs):
    xs = np.array(xs, float)
    return float(np.exp(np.log(xs).mean()))


def rng(xs):
    xs = list(xs)
    return min(xs), float(np.median(xs)), max(xs), geo(xs)


# ------------------------------------------------------------------- dados
def load():
    raw = pd.read_csv(os.path.join(RESULTS_DIR, "raw.csv"))
    runs = pd.read_csv(os.path.join(RESULTS_DIR, "runs.csv"))
    assert (raw.valido == 1).all() and (runs.valido == 1).all()
    g = raw.groupby(["impl", "dataset", "minsup", "etapa"]).tempo_ms
    med = g.median().unstack("etapa")
    meta = runs.groupby(["impl", "dataset", "minsup"]).median(numeric_only=True)
    cfg = pd.read_csv(os.path.join(MICRO, f"config_stats_{THRESHOLD_SET}.csv")).set_index(["dataset", "minsup"])
    grain = pd.read_csv(os.path.join(RESULTS_DIR, "grain.csv"))
    cb = pd.read_csv(os.path.join(MICRO, "countbench.csv")).pivot(index="dataset", columns="kernel", values="ms")
    micro = pd.read_csv(os.path.join(MICRO, "microbench.csv"), header=None,
                        names=["dataset", "ntx", "nitems", "metric", "value"])
    pcie = pd.read_csv(os.path.join(MICRO, "pcie.csv"), header=None, names=["gpu", "metric", "value"])
    gpu_runs = runs[runs.impl != "seq"]
    buses = sorted(gpu_runs.pci.dropna().astype(str).unique())
    return raw, runs, med, meta, cfg, grain, cb, micro, pcie, buses


def blocks(impl, d):
    b = pd.read_csv(os.path.join(RESULTS_DIR, "blocks", f"{impl}_{d}.csv"))
    return b[b.launch == 0]


def main(out_path):
    raw, runs, med, meta, cfg, grain, cb, micro, pcie, buses = load()
    configs = [(d, s) for d in DATASETS for s in sorted(CONFIGS[d], reverse=True)]
    M = lambda impl, d, s, st: float(med.loc[(impl, d, s)].get(st, np.nan))  # noqa: E731
    lo = {d: min(CONFIGS[d]) for d in DATASETS}
    hi = {d: max(CONFIGS[d]) for d in DATASETS}
    nrep = int(runs.groupby(["impl", "dataset", "minsup"]).size().median())
    n_runs = len(runs)
    n_grain = len(grain)

    def core(impl, d, s):
        return sum(M(impl, d, s, c) for c in CORE)

    def hidden(impl, d, s):
        return M(impl, d, s, "total") - sum(np.nan_to_num(M(impl, d, s, st)) for st in STAGES)

    # Speedups por fase (dinâmica final sobre o sequencial).
    phase_sp = {st: [M("seq", d, s, st) / M("gpu_dynamic", d, s, st) for d, s in configs] for st in CORE}
    core_sp = [core("seq", d, s) / core("gpu_dynamic", d, s) for d, s in configs]
    coretr_sp = [core("seq", d, s) / (core("gpu_dynamic", d, s) + M("gpu_dynamic", d, s, "transferencia_h2d")
                                      + M("gpu_dynamic", d, s, "transferencia_d2h")) for d, s in configs]
    tot_sp = [M("seq", d, s, "total") / M("gpu_dynamic", d, s, "total") for d, s in configs]
    amdahl = [M("seq", d, s, "total") / (M("seq", d, s, "leitura") + M("seq", d, s, "escrita")) for d, s in configs]
    sums = {i: (sum(core(i, d, s) for d, s in configs) / 1000, sum(M(i, d, s, "total") for d, s in configs) / 1000)
            for i in ["seq", "gpu_static", "gpu_dynamic"]}
    hid = {i: [hidden(i, d, s) for d, s in configs] for i in ["gpu_static", "gpu_dynamic"]}

    mine_dyn_slower = [(d, s) for d, s in configs if M("gpu_dynamic", d, s, "mineracao") > M("seq", d, s, "mineracao")]
    mine_st_slower = [(d, s) for d, s in configs if M("gpu_static", d, s, "mineracao") > M("seq", d, s, "mineracao")]
    dyn_vs_st = [M("gpu_static", d, s, "mineracao") / M("gpu_dynamic", d, s, "mineracao") for d, s in configs]
    dyn_worse_st = [(d, s) for (d, s), x in zip(configs, dyn_vs_st) if x < 1]
    tot_slower = [(d, s) for (d, s), x in zip(configs, tot_sp) if x < 1]
    cons_sp = {(d, s): M("seq", d, s, "construcao") / M("gpu_static", d, s, "construcao") for d, s in configs}
    cons_lose = [(d, s) for (d, s), x in cons_sp.items() if x < 1]
    big_tree = [(d, s) for d, s in configs if cfg.loc[(d, s), "nos"] >= 1e5]
    small_tree = [(d, s) for d, s in configs if cfg.loc[(d, s), "nos"] < 1e5]
    sparse = [(d, s) for d, s in configs if d in ("T10I4D100K", "retail", "kosarak")]

    def lst(cs):
        return ", ".join(f"{d} {sup(s)}" for d, s in cs) if cs else "nenhuma"

    L = []
    w = L.append

    # =====================================================================
    w("# FP-Growth em GPU: relatório detalhado por fase — suportes padrão\n")
    w("Esta é uma cópia de `reports/relatorio_fases.md` refeita com os **suportes "
      "mínimos mais usados com estes datasets na literatura FIMI** (os da tabela do "
      "plano original), em vez dos suportes reduzidos escolhidos para que o sequencial "
      "levasse de 0,17 s a 80 s. As descrições das implementações são as mesmas; "
      "todos os números, tabelas e leituras foram recalculados a partir de `results/` "
      "por `scripts/relatorio_fases.py`. A comparação com o relatório original mostra "
      "como as conclusões dependem do tamanho do problema.\n")
    w("Para cada fase, o relatório mostra:\n")
    w("- como ela é implementada originalmente (versão sequencial em CPU) e quanto tempo leva;")
    w("- as ideias que testamos para acelerá-la, na ordem em que foram tentadas;")
    w("- o desempenho de cada ideia, os detalhes de implementação e por que cada mecanismo foi escolhido.\n")
    w("No fim, compara a versão final com a original, com o speedup de cada fase e o speedup total.\n")
    w(f"Todos os números vêm dos CSVs em `results/`, medidos numa única RTX 4090: a do slot "
      f"PCIe x16, barramento `{buses[0]}`. Valores em milissegundos são medianas de {nrep} "
      f"execuções, depois de 1 de aquecimento descartada.\n")
    w("---\n")

    w("## Sumário\n")
    for i, t in enumerate(["Resumo dos resultados", "Ambiente, dados e método",
                           "O algoritmo e onde o tempo está na versão original", "Fase 1 — Leitura",
                           "Fase 2 — Contagem de suporte", "Fase 3 — Pré-processamento",
                           "Transferência CPU → GPU", "Fase 4 — Construção da FP-tree", "Fase 5 — Mineração",
                           "Fase 6 — Retorno e escrita da saída", "Custo fixo fora das etapas medidas",
                           "Comparação final: tudo junto", "O que falta e o ganho que ainda é possível",
                           "Comparação com o relatório de suportes reduzidos", "Reprodução"], 1):
        w(f"{i}. {t}")
    w("\n---\n")

    # ------------------------------------------------------------ 1. resumo
    w("## 1. Resumo dos resultados\n")
    w("Na tabela, **speedup = tempo do sequencial / tempo da versão final em GPU** "
      "(mineração dinâmica, `grain` 64). Faixa e média geométrica sobre as "
      f"{len(configs)} configurações (6 datasets × 4 suportes).\n")
    w("| Fase | Ideia final | Speedup mín. | Speedup mediano | Speedup máx. | Média geométrica |")
    w("| --- | --- | ---: | ---: | ---: | ---: |")
    ideas = {"contagem": "histograma por bloco em memória compartilhada",
             "preprocessamento": "scans + ordenação segmentada",
             "construcao": "ordenação + LCP + scan + sparse table",
             "mineracao": "fila global dinâmica + blocos persistentes"}
    names = {"contagem": "Contagem de suporte", "preprocessamento": "Pré-processamento",
             "construcao": "Construção da FP-tree", "mineracao": "Mineração"}
    for st in CORE:
        a, b, c, g = rng(phase_sp[st])
        w(f"| {names[st]} | {ideas[st]} | {sp(a)} | {sp(b)} | {sp(c)} | {sp(g)} |")
    a, b, c, g = rng(core_sp)
    w(f"| **Núcleo** (as 4 fases acima) | — | **{sp(a)}** | **{sp(b)}** | **{sp(c)}** | **{sp(g)}** |")
    a, b, c, g = rng(coretr_sp)
    w(f"| Núcleo + transferências | — | {sp(a)} | {sp(b)} | {sp(c)} | {sp(g)} |")
    a, b, c, g = rng(tot_sp)
    w(f"| **Ponta a ponta** (com leitura e escrita) | — | **{sp(a)}** | **{sp(b)}** | **{sp(c)}** | **{sp(g)}** |\n")
    w(f"Somando as {len(configs)} configurações: o núcleo do algoritmo cai de **{num(sums['seq'][0], 2)} s para "
      f"{num(sums['gpu_dynamic'][0], 2)} s ({sp(sums['seq'][0] / sums['gpu_dynamic'][0])})**, e o tempo total cai de "
      f"**{num(sums['seq'][1], 2)} s para {num(sums['gpu_dynamic'][1], 2)} s "
      f"({sp(sums['seq'][1] / sums['gpu_dynamic'][1])})**.\n")
    w("Leituras principais:\n")
    w(f"- **Com os suportes padrão, a maior parte das configurações é pequena.** "
      f"A mineração sequencial leva de {num(min(M('seq', d, s, 'mineracao') for d, s in configs))} ms a "
      f"{num(max(M('seq', d, s, 'mineracao') for d, s in configs))} ms e o tempo total sequencial de "
      f"{num(min(M('seq', d, s, 'total') for d, s in configs))} ms a "
      f"{num(max(M('seq', d, s, 'total') for d, s in configs))} ms. Nessa escala, custos fixos da GPU "
      "(lançamentos, alocações, duas passadas) pesam tanto quanto o trabalho.")
    w(f"- **Construção:** ganha {sp(min(cons_sp[c] for c in big_tree))}–{sp(max(cons_sp[c] for c in big_tree))} "
      f"nas árvores com 100 mil nós ou mais ({len(big_tree)} configurações) e perde "
      f"({sp(min(cons_sp[c] for c in cons_lose))}–{sp(max(cons_sp[c] for c in cons_lose))}) nas "
      f"{len(cons_lose)} configurações de chess e mushroom, cujas árvores têm até "
      f"{big(max(cfg.loc[c, 'nos'] for c in cons_lose))} nós.")
    w(f"- **Mineração:** a dinâmica é mais rápida que a CPU em {len(configs) - len(mine_dyn_slower)} das "
      f"{len(configs)} configurações. Perde nas {len(mine_dyn_slower)} em que a mineração sequencial inteira "
      f"leva no máximo {num(max(M('seq', d, s, 'mineracao') for d, s in mine_dyn_slower))} ms "
      f"({lst(mine_dyn_slower)}). A fila dinâmica só faz diferença onde há carga grande e desigual: "
      f"até {sp(max(dyn_vs_st))} sobre a estática; nas configurações leves fica empatada ou um pouco atrás "
      f"(em {len(dyn_worse_st)} delas).")
    prep_all = [M("gpu_dynamic", d, s, "mineracao_preparacao") for d, s in configs]
    w("- **Custo fixo corrigido:** na primeira rodada, a preparação da mineração (reserva e zeragem de 6–9 GB "
      "de arenas e pool) ficava fora do cronômetro de qualquer etapa e custava dezenas de ms por execução. "
      "A memória deixou de ser zerada e a preparação passou a ser medida dentro da mineração: agora custa "
      f"{num(min(prep_all), 1)}–{num(max(prep_all), 1)} ms (seção 11).")
    a, b, c, g = rng(tot_sp)
    w(f"- **Ponta a ponta:** {sp(a)}–{sp(c)}. A GPU fica mais lenta que a CPU em {len(tot_slower)} configurações "
      f"({lst(tot_slower)}), todas com tempo sequencial total de no máximo "
      f"{num(max(M('seq', d, s, 'total') for d, s in tot_slower), 0)} ms.\n")
    w("---\n")

    # ---------------------------------------------------------- 2. ambiente
    w("## 2. Ambiente, dados e método\n")
    pc = pcie.set_index(["gpu", "metric"]).value
    d2d = micro[micro.metric == "d2d_GBps"].value.median()
    w(f"**Hardware.** NVIDIA RTX 4090: 128 SMs, 24 GB e banda de pico de 1008 GB/s; medimos "
      f"{num(d2d, 0)} GB/s numa cópia D2D. Ela fica num slot PCIe Gen4 x16 "
      f"({num(pc[('gpu0', 'h2d_pinned_GBps')], 1)} GB/s com memória *pinned*, "
      f"{num(pc[('gpu0', 'h2d_pageable_GBps')], 1)} GB/s *pageable*). CPU AMD Ryzen 9 7900 (12 núcleos), "
      "60 GB de RAM, CUDA 12.4, g++ 13, `-O3`.\n")
    w(f"A máquina tem uma segunda RTX 4090 num link x4 ({num(pc[('gpu1', 'h2d_pinned_GBps')], 1)} GB/s). Ela não "
      "foi usada. Cada execução registra o barramento PCI, e os scripts falham se alguma tiver rodado em outra placa.\n")
    w("**Datasets.** Seis bases do repositório FIMI, obtidas de um espelho da Wayback Machine porque o site "
      "original está fora do ar.\n")
    w("| Dataset | Tipo | Transações | Itens distintos | Tam. médio | Tam. máx. | Suportes avaliados |")
    w("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for d in DATASETS:
        c = cfg.loc[d].iloc[0]
        w(f"| {d} | {KIND[d]} | {num(c.transacoes_orig, 0)} | {num(c.itens_distintos, 0)} | {num(c.tam_medio, 1)} | "
          f"{num(c.tam_max, 0)} | {' · '.join(sup(s) for s in sorted(CONFIGS[d], reverse=True))} |")
    w("\nOs suportes são os mais usados com estes datasets na literatura FIMI (a tabela do plano original). "
      "O conjunto com suportes reduzidos está descrito em `reports/relatorio_fases.md` (números anteriores à "
      "correção da seção 11); suas medições serão refeitas em `results/low_support/` (TODO 1.1).\n")
    w("**O que cada configuração produz** (maior suporte → menor suporte):\n")
    w("| Dataset | Itens frequentes | Nós da FP-tree | Itemsets frequentes |")
    w("| --- | ---: | ---: | ---: |")
    for d in DATASETS:
        a, b = cfg.loc[(d, hi[d])], cfg.loc[(d, lo[d])]
        w(f"| {d} | {num(a.itens_freq, 0)} → {num(b.itens_freq, 0)} | {big(a.nos)} → {big(b.nos)} | "
          f"{big(meta.loc[('seq', d, hi[d]), 'itemsets'])} → {big(meta.loc[('seq', d, lo[d]), 'itemsets'])} |")
    w("\nComparado ao conjunto reduzido, as saídas são de 10× a 1.000× menores (no máximo "
      f"{big(max(meta.loc[('seq', d, s), 'itemsets'] for d, s in configs))} itemsets) e várias árvores são "
      "pequenas: o regime em que custos fixos da GPU importam.\n")
    w("**Corretude.** A saída é canônica (itens em ordem crescente, linhas ordenadas), então comparar duas "
      "saídas é um `cmp`.\n")
    w("- O sequencial é idêntico ao `fpgrowth` 6.21 de Christian Borgelt em 12 casos e às saídas calculadas "
      "à mão em 5 datasets minúsculos.")
    w(f"- Toda execução do benchmark teve o hash da saída comparado com o do sequencial: {n_runs} execuções "
      f"principais mais {n_grain} de `grain`, **nenhuma inválida**.")
    w("- A suíte `tests` tem 535 verificações; a versão dinâmica rodou 420 vezes seguidas sem travar.\n")
    w("---\n")

    # ------------------------------------------------- 3. algoritmo e tempos
    w("## 3. O algoritmo e onde o tempo está na versão original\n")
    w("O FP-Growth tem quatro passos algorítmicos, que aqui ficam entre uma leitura e uma escrita:\n")
    w("```")
    w(" leitura ─► contagem ─► pré-processamento ─► construção da FP-tree ─► mineração ─► escrita")
    w("  (CPU)       (1)              (2)                    (3)                 (4)        (CPU)")
    w("              └──────────────── núcleo do algoritmo (vai para a GPU) ────────┘")
    w("```\n")
    w("1. **Contagem:** suporte de cada item.")
    w("2. **Pré-processamento:** descarta itens infrequentes, renumera os frequentes por suporte decrescente "
      "(o *rank*) e ordena cada transação.")
    w("3. **Construção:** insere as transações numa árvore de prefixos com contadores (a FP-tree) e numa tabela "
      "de cabeçalho que liga todos os nós de cada item.")
    w("4. **Mineração:** para cada item `y`, do menos para o mais frequente, emite `S ∪ {y}`, extrai a *base "
      "condicional*, constrói a árvore condicional e repete a mineração nela; árvores que são um caminho único "
      "têm todas as combinações emitidas diretamente.\n")
    w("As transações ficam em formato **CSR**: um vetor `items[]` com os ids de todas as transações em sequência "
      "e um vetor `offsets[]` de tamanho n+1.\n")
    w("**Tempo da versão original por fase** (sequencial, suporte intermediário, ms e % do total):\n")
    w("| Dataset (minsup) | Leitura | Contagem | Pré-proc. | Construção | Mineração | Escrita | Total |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    def seqrow(d, s):
        t = M("seq", d, s, "total")
        cells = [f"{num(M('seq', d, s, st))} ({pct(100 * M('seq', d, s, st) / t)})"
                 for st in ["leitura", "contagem", "preprocessamento", "construcao", "mineracao", "escrita"]]
        return " | ".join(cells) + f" | {num(t)}"
    for d in DATASETS:
        w(f"| {d} ({sup(MID[d])}) | {seqrow(d, MID[d])} |")
    share = {st: [100 * M("seq", d, s, st) / M("seq", d, s, "total") for d, s in configs]
             for st in ["leitura", "construcao", "mineracao", "escrita"]}
    w("\nNo conjunto inteiro (apêndice A):\n")
    w(f"- **Mineração:** {pct(min(share['mineracao']))}–{pct(max(share['mineracao']))} do total.")
    w(f"- **Construção:** {pct(min(share['construcao']))}–{pct(max(share['construcao']))}; pesa mais nos esparsos "
      "(T10I4D100K chega perto de metade do tempo).")
    w(f"- **Leitura:** {pct(min(share['leitura']))}–{pct(max(share['leitura']))}; com saídas pequenas, a leitura "
      "passa a ser uma fatia grande do tempo nos datasets grandes (accidents em suporte alto, kosarak).")
    w(f"- **Escrita:** {pct(min(share['escrita']))}–{pct(max(share['escrita']))}; só domina chess e mushroom nos "
      "suportes mais baixos.\n")
    w("---\n")

    # ------------------------------------------------------------ 4. leitura
    w("## 4. Fase 1 — Leitura (CPU)\n")
    w("**Implementação original.** O arquivo inteiro é lido de uma vez (`fread`) e convertido em CSR por um único "
      "laço que acumula dígitos e fecha uma transação a cada `\\n`. Itens repetidos dentro de uma transação são "
      "removidos. Código em `src/common/fimi.cpp`.\n")
    rd = {d: np.median([M("seq", d, s, "leitura") for s in CONFIGS[d]]) for d in DATASETS}
    w(f"**Tempo:** {num(rd['chess'])}–{num(rd['mushroom'])} ms em chess e mushroom, {num(rd['T10I4D100K'])}–"
      f"{num(rd['retail'])} ms em T10I4D100K e retail, {num(rd['accidents'], 0)} ms em accidents e "
      f"{num(rd['kosarak'], 0)} ms em kosarak.\n")
    w("**Ideias de melhoria:** nenhuma implementada; a leitura é idêntica nas três versões. Com os suportes "
      f"padrão ela pesa bem mais que no conjunto reduzido: até {pct(max(share['leitura']))} do tempo sequencial "
      "e uma fração ainda maior do tempo da GPU (seção 12). Opções futuras: parsing paralelo com os 12 núcleos "
      "(ganho ideal ~12×) ou parsing na GPU com um scan sobre os separadores.\n")
    w("---\n")

    # ---------------------------------------------------------- 5. contagem
    w("## 5. Fase 2 — Contagem de suporte\n")
    w("### 5.1 Implementação original\n")
    w("Um vetor de contadores do tamanho do maior id e um laço sobre todos os itens: `++sup[x]`. É o melhor "
      "caso possível para a CPU: acesso sequencial e o vetor cabe no cache.\n")
    cs = {d: np.median([M("seq", d, s, "contagem") for s in CONFIGS[d]]) for d in DATASETS}
    w(f"**Tempo original:** {num(min(cs.values()))} ms (chess) a {num(max(cs.values()))} ms (accidents). "
      "Independe do suporte e é sempre menos de 1 % do total.\n")
    w("### 5.2 Ideia 1 — `atomicAdd` global (kernel v1)\n")
    w("Uma thread por ocorrência executa `atomicAdd(&support[items[p]], 1)` na memória global. É a tradução "
      "direta do laço sequencial; os atômicos são resolvidos no L2, um por endereço de cada vez.\n")
    w("### 5.3 Ideia 2 — Histograma privado por bloco em memória compartilhada (kernel v2)\n")
    w("Cada bloco de 256 threads mantém um histograma de 12.288 contadores (48 KB) em memória compartilhada e, "
      "no fim, soma no vetor global um `atomicAdd` por contador não nulo. Se há mais ids que contadores, "
      "`gridDim.y` divide os ids em faixas e cada bloco relê a base (retail 2 faixas, kosarak 4). No máximo 4 "
      "blocos por SM.\n")
    w("**Desempenho** (kernel isolado, média de 50 lançamentos; independe do suporte):\n")
    w("| Dataset | Ocorrências | Sequencial (ms) | v1 (ms) | v2 (ms) | v2 vs. v1 | v2 vs. seq |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        c = cfg.loc[d].iloc[0]
        w(f"| {d} | {big(c.itens_orig)} | {num(cs[d])} | {num(cb.loc[d, 'v1'])} | {num(cb.loc[d, 'v2'])} | "
          f"{sp(cb.loc[d, 'v1'] / cb.loc[d, 'v2'])} | {sp(cs[d] / cb.loc[d, 'v2'])} |")
    w("\nO maior ganho está onde a contenção era o problema (accidents). Estes números são os mesmos do "
      "relatório original, porque a contagem não depende do suporte.\n")
    w("### 5.4 Resultado da fase dentro do pipeline\n")
    a, b, c, g = rng(phase_sp["contagem"])
    stage = [M("gpu_dynamic", d, s, "contagem") for d, s in configs]
    w(f"Dentro do programa, a fase de contagem na GPU mede {num(min(stage))}–{num(max(stage))} ms, muito mais "
      f"que o kernel, e o speedup da fase vai de {sp(a)} a {sp(c)} (média geométrica {sp(g)}). O custo extra é "
      "fixo (alocação do vetor e, provavelmente, carregamento preguiçoso do módulo CUDA no primeiro "
      "lançamento; **não medido isoladamente**). A fase vale menos de 1 % do tempo.\n")
    w("---\n")

    # ---------------------------------------------------- 6. pré-processamento
    w("## 6. Fase 3 — Pré-processamento\n")
    w("### 6.1 Implementação original\n")
    w("Tabela de ranks (suporte decrescente, id crescente), filtragem e troca de id por rank, `std::sort` por "
      "transação e descarte das transações vazias.\n")
    w("### 6.2 Ideia — tudo como primitivas paralelas\n")
    w("`copy_if` + `thrust::sort` para os ranks; um kernel com uma thread por ocorrência para filtrar e "
      "remapear; `exclusive_scan` das flags; `cub::DeviceSegmentedReduce::Sum` para os novos tamanhos; "
      "*scatter*; `copy_if` + `inclusive_scan` para descartar as vazias; `cub::DeviceSegmentedSort::SortKeys` "
      "para ordenar cada transação. O trabalho é dividido por ocorrência, não por transação.\n")
    w("**Desempenho** (maior / menor suporte):\n")
    w("| Dataset | minsup | Sequencial (ms) | GPU (ms) | Speedup |")
    w("| --- | --- | ---: | ---: | ---: |")
    for d in DATASETS:
        a1, a2 = M("seq", d, hi[d], "preprocessamento"), M("seq", d, lo[d], "preprocessamento")
        b1, b2 = M("gpu_dynamic", d, hi[d], "preprocessamento"), M("gpu_dynamic", d, lo[d], "preprocessamento")
        w(f"| {d} | {sup(hi[d])} / {sup(lo[d])} | {num(a1)} / {num(a2)} | {num(b1)} / {num(b2)} | "
          f"{sp(a1 / b1)} / {sp(a2 / b2)} |")
    a, b, c, g = rng(phase_sp["preprocessamento"])
    w(f"\nFaixa: {sp(a)}–{sp(c)}. Com suporte alto sobram poucos itens frequentes e a versão sequencial fica "
      "barata, então a GPU (dominada por ~10 chamadas de biblioteca com alocação e lançamento) ganha menos ou "
      "empata. O tráfego obrigatório em kosarak é de ~0,2 ms; a GPU leva ~2,5 ms.\n")
    w("---\n")

    # --------------------------------------------------------------- 7. H2D
    w("## 7. Transferência CPU → GPU (H2D)\n")
    w("Dois `cudaMemcpy` (items e offsets) a partir de memória *pageable*.\n")
    w("| Dataset | Bytes | Tempo (ms) | Banda efetiva |")
    w("| --- | ---: | ---: | ---: |")
    for d in DATASETS:
        c = cfg.loc[d].iloc[0]
        byt = (c.itens_orig + c.transacoes_orig + 1) * 4
        t = np.median([M("gpu_dynamic", d, s, "transferencia_h2d") for s in CONFIGS[d]])
        w(f"| {d} | {num(byt / 1e6, 1)} MB | {num(t)} | {num(byt / t / 1e6, 1)} GB/s |")
    w("\nNos datasets pequenos o tempo é dominado pela alocação dos vetores na GPU, não pela cópia. Melhoria "
      "possível, não implementada: memória *pinned* e cópia sobreposta à contagem.\n")
    w("---\n")

    # --------------------------------------------------------- 8. construção
    w("## 8. Fase 4 — Construção da FP-tree\n")
    w("### 8.1 Implementação original\n")
    w("FP-tree clássica com nós em vetores e índices no lugar de ponteiros; busca de filho por tabela hash "
      "(pai, item); lista ligada por item na tabela de cabeçalho.\n")
    w("### 8.2–8.4 Ideias implementadas\n")
    w("1. **Nós como prefixos distintos:** ordenação lexicográfica das transações (`thrust::sort` com "
      "comparador), LCP com a transação anterior, `exclusive_scan` de `len − LCP` para os ids dos nós "
      "(em pré-ordem), escrita de item, profundidade e transação criadora.")
    w("2. **Pais e contadores sem atômicos:** *previous smaller value* e *next smaller or equal* sobre o LCP, "
      "respondidos por busca binária numa sparse table de mínimos.")
    w("3. **Tabela de cabeçalho por ordenação:** `stable_sort_by_key` por item, `lower_bound` e `reduce_by_key`.\n")
    w("(Detalhes, alternativas descartadas e o exemplo de Han, Pei e Yin: seção 8 de `relatorio_fases.md`.)\n")
    w("### 8.5 Desempenho da construção\n")
    w("Mediana, ms (maior e menor suporte de cada dataset; todas as configurações no apêndice B):\n")
    head = ("| Dataset | minsup | Nós | Seq. | GPU | Speedup | sort | LCP | scan | nós | sparse | pais | cont. | "
            "cabeç. | % sort | Speedup sem o sort |")
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    subs = ["construcao_sort", "construcao_lcp", "construcao_scan", "construcao_nos", "construcao_sparse",
            "construcao_pais", "construcao_contadores", "construcao_cabecalho"]

    def consrow(d, s):
        sq, gp, srt = M("seq", d, s, "construcao"), M("gpu_static", d, s, "construcao"), \
            M("gpu_static", d, s, "construcao_sort")
        return (f"| {d} | {sup(s)} | {big(cfg.loc[(d, s), 'nos'])} | {num(sq)} | {num(gp)} | {sp(sq / gp)} | "
                + " | ".join(num(M("gpu_static", d, s, x)) for x in subs)
                + f" | {pct(100 * srt / gp)} | {sp(sq / (gp - srt))} |")
    w(head)
    w(sep)
    for d in DATASETS:
        for s in [hi[d], lo[d]]:
            w(consrow(d, s))
    sortfrac = [100 * M("gpu_static", d, s, "construcao_sort") / M("gpu_static", d, s, "construcao")
                for d, s in configs]
    nosort = [M("seq", d, s, "construcao") / (M("gpu_static", d, s, "construcao")
                                              - M("gpu_static", d, s, "construcao_sort")) for d, s in configs]
    w("\n**Leitura.**\n")
    w(f"- **Árvores com 100 mil nós ou mais** ({len(big_tree)} configurações): "
      f"{sp(min(cons_sp[c] for c in big_tree))}–{sp(max(cons_sp[c] for c in big_tree))} mais rápida que a "
      "inserção sequencial, crescendo com o tamanho da árvore (kosarak vai de "
      f"{sp(cons_sp[('kosarak', hi['kosarak'])])} com {big(cfg.loc[('kosarak', hi['kosarak']), 'nos'])} nós a "
      f"{sp(cons_sp[('kosarak', lo['kosarak'])])} com {big(cfg.loc[('kosarak', lo['kosarak']), 'nos'])}).")
    w(f"- **Árvores pequenas** (abaixo de 100 mil nós, {len(small_tree)} configurações): "
      f"{sp(min(cons_sp[c] for c in small_tree))}–{sp(max(cons_sp[c] for c in small_tree))}. A GPU perde em "
      f"{len(cons_lose)} delas ({lst(cons_lose)}): a construção sequencial leva "
      f"{num(min(M('seq', d, s, 'construcao') for d, s in cons_lose))}–"
      f"{num(max(M('seq', d, s, 'construcao') for d, s in cons_lose))} ms, menos que o custo fixo da ordenação "
      "na GPU (~2–3 ms).")
    w(f"- **A ordenação lexicográfica domina**: {pct(min(sortfrac))}–{pct(max(sortfrac))} do tempo da fase. Sem "
      f"ela, o resto seria {sp(min(nosort))}–{sp(max(nosort))} mais rápido que o sequencial.")
    w(f"- **Pais e contadores** custam no máximo "
      f"{num(max(max(M('gpu_static', d, s, 'construcao_pais'), M('gpu_static', d, s, 'construcao_contadores')) for d, s in configs))} ms "
      f"mesmo com {big(max(cfg.nos))} nós.\n")
    radix = micro[micro.metric == "radix64_pairs_ms"].set_index("dataset").value
    ideal = {(d, s): M("seq", d, s, "construcao") / (M("gpu_static", d, s, "construcao")
                                                     - M("gpu_static", d, s, "construcao_sort") + radix[d])
             for d, s in configs}
    best = max(ideal, key=ideal.get)
    still = [c for c in cons_lose if ideal[c] <= 1]
    w(f"**Ganho ainda possível com uma ordenação melhor.** Com a ordenação no tempo de um radix sort de 64 bits "
      f"do mesmo número de chaves (medido), a construção chegaria a até {sp(ideal[best])} sobre o sequencial "
      f"({best[0]} {sup(best[1])}), e a GPU passaria a ganhar em {len(cons_lose) - len(still)} das "
      f"{len(cons_lose)} configurações em que hoje perde"
      + (f"; a exceção é {lst(still)} ("
         + ", ".join(f"de {sp(cons_sp[c])} para {sp(ideal[c])}" for c in still) + ")" if still else "")
      + ".\n")
    w("---\n")

    # ---------------------------------------------------------- 9. mineração
    w("## 9. Fase 5 — Mineração\n")
    w("### 9.1 Implementação original\n")
    w("Recursão clássica com atalho de caminho único; as árvores condicionais mantêm a ordem global de ranks. "
      f"**Tempo original:** {num(min(M('seq', d, s, 'mineracao') for d, s in configs))} ms a "
      f"{num(max(M('seq', d, s, 'mineracao') for d, s in configs))} ms.\n")
    w("### 9.2 Ideia 1 — Mineração estática: um bloco de threads por item\n")
    w("Um bloco de 256 threads por item frequente resolve toda a recursão do item com pilha explícita, arena de "
      "duas pontas, construção condicional em bloco (as mesmas funções da construção global), merge sort de "
      "bloco, histograma em shared, atalho de caminho único analítico, saída em duas passadas e tratamento de "
      "estouro por projeção na GPU. (Tabela de mecanismos e motivos: seção 9.2 de `relatorio_fases.md`.)\n")
    w("**Desempenho:**\n")
    w("| Dataset | minsup | Itemsets | Seq. (ms) | Estática (ms) | Estática vs. seq. | Bloco mais lento / média |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        for s in [hi[d], lo[d]]:
            a, b = M("seq", d, s, "mineracao"), M("gpu_static", d, s, "mineracao")
            w(f"| {d} | {sup(s)} | {big(meta.loc[('seq', d, s), 'itemsets'])} | {num(a)} | {num(b)} | "
              f"{'**' if b > a else ''}{sp(a / b)}{'**' if b > a else ''} | "
              f"{num(meta.loc[('gpu_static', d, s), 'desbalanceamento'], 1)} |")
    w(f"\nA estática é **mais lenta que a CPU em {len(mine_st_slower)} das {len(configs)} configurações** "
      f"({lst(mine_st_slower)}), todas de datasets densos, onde há poucos itens (poucos blocos) e, nos "
      "suportes altos, pouquíssimo trabalho. Nos esparsos ela ganha de "
      f"{sp(min(M('seq', d, s, 'mineracao') / M('gpu_static', d, s, 'mineracao') for d, s in sparse))} a "
      f"{sp(max(M('seq', d, s, 'mineracao') / M('gpu_static', d, s, 'mineracao') for d, s in sparse))}, porque há "
      "centenas de itens com carga parecida; o pior caso é kosarak "
      f"{sup(lo['kosarak'])}, em que um único item domina o kernel (razão bloco mais lento / média de "
      f"{num(meta.loc[('gpu_static', 'kosarak', lo['kosarak']), 'desbalanceamento'], 0)}).\n")
    w("Distribuição do tempo por bloco (suporte intermediário):\n")
    w("| Dataset | minsup | Blocos (itens) | Bloco mais lento (ms) | Média (ms) | Eficiência* |")
    w("| --- | --- | ---: | ---: | ---: | ---: |")
    eff = {}
    for d in DATASETS:
        b = blocks("gpu_static", d)
        W = b.busy_ms.sum()
        e = (W / 256) / max(b.busy_ms.max(), W / 256)
        eff[(d, "s")] = e
        w(f"| {d} | {sup(MID[d])} | {num(len(b), 0)} | {num(b.busy_ms.max())} | {num(b.busy_ms.mean())} | "
          f"{pct(100 * e)} |")
    w("\n\\* trabalho total dividido por 256 blocos residentes, sobre o tempo do bloco mais lento.\n")
    w("### 9.3 Ideia 2 — Mineração dinâmica: fila global de tarefas com blocos persistentes\n")
    w("256 blocos persistentes consomem uma fila global MPMC sem travas (número de sequência por posição); "
      "árvores condicionais com base ≥ `grain` são copiadas para um pool global e publicadas como tarefas; "
      "terminação por contador de pendentes; fila ou pool cheio → processamento local. (Tabela de mecanismos: "
      "seção 9.3 de `relatorio_fases.md`.)\n")
    w("**Desempenho** (`grain` = 64):\n")
    w("| Dataset | minsup | Seq. (ms) | Estática (ms) | Dinâmica (ms) | Din. vs. seq. | Din. vs. est. | "
      "Mais lento / média: est. → din. | Tarefas na fila |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for d in DATASETS:
        for s in [hi[d], lo[d]]:
            a, b, c = M("seq", d, s, "mineracao"), M("gpu_static", d, s, "mineracao"), \
                M("gpu_dynamic", d, s, "mineracao")
            w(f"| {d} | {sup(s)} | {num(a)} | {num(b)} | {num(c)} | {sp(a / c)} | {sp(b / c)} | "
              f"{num(meta.loc[('gpu_static', d, s), 'desbalanceamento'], 1)} → "
              f"{num(meta.loc[('gpu_dynamic', d, s), 'desbalanceamento'], 2)} | "
              f"{num(meta.loc[('gpu_dynamic', d, s), 'enfileiradas'], 0)} |")
    a, b, c, g = rng(phase_sp["mineracao"])
    w(f"\nResumo: a dinâmica é mais rápida que a CPU em {len(configs) - len(mine_dyn_slower)} das "
      f"{len(configs)} configurações ({sp(a)}–{sp(c)}, média geométrica {sp(g)}) e "
      f"{sp(min(dyn_vs_st))}–{sp(max(dyn_vs_st))} em relação à estática.\n")
    w("**Por regime:**\n")
    w(f"- **Configurações minúsculas** ({lst(mine_dyn_slower)}): a mineração sequencial leva "
      f"{num(min(M('seq', d, s, 'mineracao') for d, s in mine_dyn_slower))}–"
      f"{num(max(M('seq', d, s, 'mineracao') for d, s in mine_dyn_slower))} ms e a dinâmica "
      f"{num(min(M('gpu_dynamic', d, s, 'mineracao') for d, s in mine_dyn_slower))}–"
      f"{num(max(M('gpu_dynamic', d, s, 'mineracao') for d, s in mine_dyn_slower))} ms: são duas passadas, cada "
      "uma com lançamento, sincronização e cópia de contadores, e árvores condicionais construídas por um bloco "
      "com dezenas de barreiras.")
    w(f"- **Carga leve** (esparsos em suporte alto, accidents): dinâmica e estática ficam próximas; a dinâmica "
      f"perde por pouco em {len(dyn_worse_st)} configurações ({lst(dyn_worse_st)}), por "
      f"{sp(min(dyn_vs_st))}–{sp(max(x for x in dyn_vs_st if x < 1))}. Sem tarefas grandes, a fila só "
      "acrescenta o custo de cópia e de sincronização.")
    w(f"- **Carga grande e desigual** (chess ≤ 0,6, mushroom 0,05, kosarak ≤ 0,002): é onde a fila compensa — "
      f"chess {sup(lo['chess'])} {sp(M('gpu_static', 'chess', lo['chess'], 'mineracao') / M('gpu_dynamic', 'chess', lo['chess'], 'mineracao'))}, "
      f"kosarak {sup(lo['kosarak'])} {sp(M('gpu_static', 'kosarak', lo['kosarak'], 'mineracao') / M('gpu_dynamic', 'kosarak', lo['kosarak'], 'mineracao'))} "
      "sobre a estática.")
    w("- **Razão máx/média da dinâmica alta nos suportes altos** (até "
      f"{num(max(meta.loc[('gpu_dynamic', d, s), 'desbalanceamento'] for d, s in configs), 0)}): não indica "
      "desbalanceamento, e sim que há menos tarefas do que os 256 blocos; a maioria fica ociosa e a métrica "
      "perde o sentido.\n")
    w("**Eficiência de balanceamento** (suporte intermediário):\n")
    w("| Dataset | minsup | Estática | Dinâmica |")
    w("| --- | --- | ---: | ---: |")
    for d in DATASETS:
        b = blocks("gpu_dynamic", d)
        W = b.busy_ms.sum()
        w(f"| {d} | {sup(MID[d])} | {pct(100 * eff[(d, 's')])} | {pct(100 * (W / 256) / b.busy_ms.max())} |")
    w("\n### 9.4 Ideia 3 — Ajuste da granularidade (`grain`)\n")
    gm = grain[grain.valido == 1].groupby(["dataset", "grain"]).mineracao_ms.median().unstack("grain")
    w("Mineração dinâmica em ms, suporte intermediário; o melhor em negrito:\n")
    w("| Dataset | minsup | Estática | " + " | ".join(f"g = {c}" for c in gm.columns)
      + " | Melhor vs. estática | 64 vs. melhor |")
    w("| --- | --- | ---: | " + " | ".join("---:" for _ in gm.columns) + " | ---: | ---: |")
    for d in DATASETS:
        bst = gm.loc[d].idxmin()
        cells = [f"**{num(gm.loc[d, c])}**" if c == bst else num(gm.loc[d, c]) for c in gm.columns]
        stt = M("gpu_static", d, MID[d], "mineracao")
        w(f"| {d} | {sup(MID[d])} | {num(stt)} | " + " | ".join(cells)
          + f" | {sp(stt / gm.loc[d].min())} | {sp(gm.loc[d, 64] / gm.loc[d].min())} |")
    worst = max(DATASETS, key=lambda d: gm.loc[d, 64] / gm.loc[d].min())
    w(f"\nO melhor grain fica entre {min(gm.loc[d].idxmin() for d in DATASETS)} e "
      f"{max(gm.loc[d].idxmin() for d in DATASETS)}. O padrão 64 fica a até "
      f"{sp(gm.loc[worst, 64] / gm.loc[worst].min())} do melhor ({worst} {sup(MID[worst])}; a causa não foi "
      "investigada); nos outros datasets fica a no máximo "
      f"{sp(max(gm.loc[d, 64] / gm.loc[d].min() for d in DATASETS if d != worst))}. Grains ≥ 1024 fazem a "
      "dinâmica voltar ao tempo da estática onde há carga desigual (chess, mushroom, kosarak).\n")
    w("### 9.5 Limites e o que ainda falta na mineração\n")
    p1 = [M(i, d, s, "mineracao_passada1") / M(i, d, s, "mineracao") for i in ["gpu_static", "gpu_dynamic"]
          for d, s in configs]
    w(f"- **Duas passadas:** a passada 1 é {pct(100 * min(p1))}–{pct(100 * max(p1))} do tempo de mineração. "
      "Nas configurações minúsculas o custo fixo de cada passada é o que impede a GPU de ganhar; uma passada "
      "única com buffers otimistas daria até 2×.")
    w("- **Tarefa indivisível** (accidents): as primeiras árvores condicionais são construídas por um bloco só.")
    w(f"- **Preparação:** reservar e liberar 6–9 GB de arenas e pool ainda custa "
      f"{num(min(prep_all), 1)}–{num(max(prep_all), 1)} ms por execução (seção 11); dimensionar pela "
      "necessidade reduziria esse custo nas configurações pequenas.\n")
    w("---\n")

    # ---------------------------------------------------------- 10. escrita
    w("## 10. Fase 6 — Retorno (D2H) e escrita da saída\n")
    w("Escrita idêntica nas três versões (tradução rank → id, ordenação dos itens, formatação e ordenação das "
      "linhas, em uma thread). D2H + montagem do CSR de itemsets no host, medidos juntos.\n")
    w("| Dataset | minsup | Itemsets | D2H + montagem, GPU (ms) | Escrita, seq. (ms) | Escrita, GPU (ms) |")
    w("| --- | --- | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        s = lo[d]
        w(f"| {d} | {sup(s)} | {big(meta.loc[('seq', d, s), 'itemsets'])} | "
          f"{num(M('gpu_dynamic', d, s, 'transferencia_d2h'))} | {num(M('seq', d, s, 'escrita'))} | "
          f"{num(M('gpu_dynamic', d, s, 'escrita'))} |")
    w("\nCom os suportes padrão a saída é pequena (até "
      f"{big(max(meta.loc[('seq', d, s), 'itemsets'] for d, s in configs))} itemsets), e a escrita só domina "
      f"chess e mushroom nos menores suportes (mushroom {sup(lo['mushroom'])}: "
      f"{num(M('seq', 'mushroom', lo['mushroom'], 'escrita'), 0)} de "
      f"{num(M('seq', 'mushroom', lo['mushroom'], 'total'), 0)} ms no sequencial).\n")
    w("---\n")

    # ------------------------------------------------- 11. custo fixo oculto
    w("## 11. Custo fixo fora das etapas medidas (corrigido)\n")
    pre_path = os.path.join(ROOT, "results", "pre_fix", f"raw_{THRESHOLD_SET}.csv")
    w("Na primeira rodada com os suportes padrão apareceu um custo que o relatório original não mostrava: "
      "na GPU, a soma das etapas medidas era bem menor que o tempo total.\n")
    w("**De onde vinha.** A preparação da mineração reservava as arenas dos blocos (até 35 % da memória livre, "
      "6 GB) e, na dinâmica, o pool (até 15 %, 3 GB) com `thrust::device_vector::resize`, que **zera** toda a "
      "memória, e esse trecho ficava fora do cronômetro da etapa \"mineração\", que media só os kernels. A "
      "memória não precisa ser zerada: a arena é sempre escrita antes de ser lida.\n")
    w("**Correção.** Arenas e pool passaram a ser reservados com `cudaMalloc` sem inicialização, e a reserva e "
      "a liberação entram na etapa de mineração, como a subetapa `mineracao_preparacao`. Todos os números deste "
      "relatório já são da versão corrigida.\n")
    if os.path.exists(pre_path):
        old = pd.read_csv(pre_path)
        om = old.groupby(["impl", "dataset", "minsup", "etapa"]).tempo_ms.median().unstack("etapa")

        def OM(impl, d, s, st):
            return float(om.loc[(impl, d, s)].get(st, np.nan))

        def ohid(impl, d, s):
            return OM(impl, d, s, "total") - sum(np.nan_to_num(OM(impl, d, s, st)) for st in STAGES)
        w("| Dataset | minsup | Total dinâmica antes (ms) | Fora das etapas antes (ms) | Total dinâmica depois (ms) | "
          "Preparação, agora medida (ms) | Fora das etapas depois (ms) |")
        w("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for d in DATASETS:
            for s in [hi[d], lo[d]]:
                w(f"| {d} | {sup(s)} | {num(OM('gpu_dynamic', d, s, 'total'))} | {num(ohid('gpu_dynamic', d, s))} | "
                  f"{num(M('gpu_dynamic', d, s, 'total'))} | {num(M('gpu_dynamic', d, s, 'mineracao_preparacao'))} | "
                  f"{num(hidden('gpu_dynamic', d, s))} |")
        oh = [ohid("gpu_dynamic", d, s) for d, s in configs]
        prep = [M("gpu_dynamic", d, s, "mineracao_preparacao") for d, s in configs]
        gain = [OM("gpu_dynamic", d, s, "total") - M("gpu_dynamic", d, s, "total") for d, s in configs]
        old_tot_slower = [(d, s) for d, s in configs if OM("gpu_dynamic", d, s, "total") > OM("seq", d, s, "total")]
        w(f"\nAntes, {num(min(oh), 0)}–{num(max(oh), 0)} ms por execução da dinâmica ficavam fora de qualquer etapa. "
          f"Agora a preparação custa {num(min(prep), 1)}–{num(max(prep), 1)} ms (só a reserva e a liberação da "
          f"memória) e está dentro da mineração; o que sobra fora das etapas é "
          f"{num(min(hid['gpu_dynamic']), 1)}–{num(max(hid['gpu_dynamic']), 1)} ms. O tempo total da dinâmica caiu "
          f"até {num(max(gain), 0)} ms por execução (mediana {num(float(np.median(gain)), 0)} ms; nas execuções "
          "longas a diferença fica dentro do ruído), e o número de configurações em que a GPU perde "
          f"ponta a ponta foi de {len(old_tot_slower)} para {len(tot_slower)}.\n")
    else:
        w("(Medições anteriores à correção não encontradas em `results/pre_fix/`.)\n")
    w("---\n")

    # ------------------------------------------------- 12. comparação final
    w("## 12. Comparação final: tudo junto\n")
    w("**Versão final:** leitura (CPU) → H2D → contagem v2 → pré-processamento por primitivas → construção por "
      "ordenação + LCP + sparse table → mineração dinâmica (`grain` 64) → D2H → escrita (CPU).\n")
    w("### 12.1 Speedup de cada fase e total, por configuração\n")
    w("Speedup = tempo do sequencial / tempo da versão final. Núcleo = contagem + pré-processamento + "
      "construção + mineração. \"Amdahl\" é o teto do speedup ponta a ponta se o núcleo levasse tempo zero: "
      "`T_seq / (T_leitura + T_escrita)`.\n")
    w("| Dataset | minsup | Contagem | Pré-proc. | Construção | Mineração | Núcleo seq. → GPU (ms) | **Núcleo** | "
      "Núcleo + transf. | Total seq. → GPU (ms) | **Total** | Amdahl |")
    w("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |")
    for k, (d, s) in enumerate(configs):
        w(f"| {d} | {sup(s)} | " + " | ".join(sp(phase_sp[st][k]) for st in CORE)
          + f" | {num(core('seq', d, s))} → {num(core('gpu_dynamic', d, s))} | **{sp(core_sp[k])}** | "
          f"{sp(coretr_sp[k])} | {num(M('seq', d, s, 'total'))} → {num(M('gpu_dynamic', d, s, 'total'))} | "
          f"**{sp(tot_sp[k])}** | {sp(amdahl[k])} |")
    w("\n### 12.2 Agregado\n")
    w("| Medida | Mín. | Mediana | Máx. | Média geométrica |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for label, xs in [("Contagem", phase_sp["contagem"]), ("Pré-processamento", phase_sp["preprocessamento"]),
                      ("Construção", phase_sp["construcao"]), ("Mineração", phase_sp["mineracao"]),
                      ("**Núcleo**", core_sp), ("Núcleo + transferências", coretr_sp),
                      ("**Total ponta a ponta**", tot_sp)]:
        a, b, c, g = rng(xs)
        w(f"| {label} | {sp(a)} | {sp(b)} | {sp(c)} | {sp(g)} |")
    a, b, c, g = rng(amdahl)
    w(f"| Teto de Amdahl (núcleo = 0) | {sp(a)} | {sp(b)} | {sp(c)} | — |\n")
    w(f"Soma das {len(configs)} configurações:\n")
    w("| Versão | Núcleo (s) | Total (s) |")
    w("| --- | ---: | ---: |")
    for i, label in [("seq", "Sequencial"), ("gpu_static", "GPU estática"), ("gpu_dynamic", "**GPU dinâmica (final)**")]:
        w(f"| {label} | {num(sums[i][0], 3)} | {num(sums[i][1], 3)} |")
    w(f"| Speedup final | **{sp(sums['seq'][0] / sums['gpu_dynamic'][0])}** | "
      f"**{sp(sums['seq'][1] / sums['gpu_dynamic'][1])}** |\n")
    nz = sum(hid["gpu_dynamic"]) / 1000
    w(f"Do total somado da GPU dinâmica, {num(nz, 2)} s ({pct(100 * nz / sums['gpu_dynamic'][1])}) ficam fora "
      "das etapas medidas.\n")
    w("### 12.3 Onde o tempo está na versão final (suporte intermediário)\n")
    w("| Dataset | Leitura | H2D | Núcleo (GPU) | D2H + montagem | Escrita | Fora das etapas | Total |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in DATASETS:
        s = MID[d]
        t = M("gpu_dynamic", d, s, "total")
        parts = [M("gpu_dynamic", d, s, "leitura"), M("gpu_dynamic", d, s, "transferencia_h2d"),
                 core("gpu_dynamic", d, s), M("gpu_dynamic", d, s, "transferencia_d2h"),
                 M("gpu_dynamic", d, s, "escrita"), hidden("gpu_dynamic", d, s)]
        w(f"| {d} {sup(s)} | " + " | ".join(pct(100 * p / t) for p in parts) + f" | {num(t)} ms |")
    w("\nO resto do tempo, fora do núcleo do algoritmo, é leitura e escrita na CPU e transferências.\n")
    w("---\n")

    # -------------------------------------------------------- 13. o que falta
    w("## 13. O que falta e o ganho que ainda é possível\n")
    w("Em ordem de impacto esperado no tempo total, **com os suportes padrão**:\n")
    w("| # | Melhoria | Fase | Ganho esperado |")
    w("| --- | --- | --- | --- |")
    prep_all = [M("gpu_dynamic", d, s, "mineracao_preparacao") for d, s in configs]
    w(f"| 1 | Dimensionar arenas e pool pela necessidade (hoje 6–9 GB fixos) | mineração | a preparação ainda "
      f"custa {num(min(prep_all), 1)}–{num(max(prep_all), 1)} ms por execução, só em `cudaMalloc`/`cudaFree` |")
    w("| 2 | Passada única na mineração | mineração | até 2× na mineração; decisivo nas configurações minúsculas |")
    w("| 3 | Parsing paralelo ou na GPU | leitura | leitura é a maior fase restante em kosarak e accidents |")
    w("| 4 | Ordenação lexicográfica por radix híbrido + alocador com cache | construção | elimina a derrota nas "
      "árvores pequenas |")
    w("| 5 | Escrita paralela ou na GPU | escrita | decisiva só em chess e mushroom nos menores suportes |")
    w("| 6 | Memória *pinned* + sobreposição | H2D | ~1–3 ms |\n")
    w("---\n")

    # ------------------------------------------------- 14. comparação conjuntos
    w("## 14. Comparação com o relatório de suportes reduzidos\n")
    low_path = os.path.join(ROOT, "results", "low_support", "raw.csv")
    if not os.path.exists(low_path):
        w("**Pendente:** as medições do conjunto reduzido com a versão corrigida ainda não foram refeitas "
          "(TODO 1.1). Os números de `reports/relatorio_fases.md` são anteriores à correção da seção 11.\n")
    else:
        lr = pd.read_csv(low_path)
        lm = lr.groupby(["impl", "dataset", "minsup", "etapa"]).tempo_ms.median().unstack("etapa")
        lidx = lm.loc["seq"].index

        def lsp(st):
            return [lm.loc[("seq",) + i, st] / lm.loc[("gpu_dynamic",) + i, st] for i in lidx]

        def lcore(impl, i):
            return sum(lm.loc[(impl,) + i, c] for c in CORE)
        l_core = [lcore("seq", i) / lcore("gpu_dynamic", i) for i in lidx]
        l_tot = [lm.loc[("seq",) + i, "total"] / lm.loc[("gpu_dynamic",) + i, "total"] for i in lidx]
        w("| Medida (média geométrica, dinâmica sobre o sequencial) | Suportes padrão | Suportes reduzidos |")
        w("| --- | ---: | ---: |")
        for st in CORE:
            w(f"| {names[st]} | {sp(geo(phase_sp[st]))} | {sp(geo(lsp(st)))} |")
        w(f"| Núcleo | {sp(geo(core_sp))} | {sp(geo(l_core))} |")
        w(f"| Ponta a ponta | {sp(geo(tot_sp))} | {sp(geo(l_tot))} |")
        w(f"| Configurações em que a mineração dinâmica perde da CPU | {len(mine_dyn_slower)} | "
          f"{sum(lm.loc[('gpu_dynamic',) + i, 'mineracao'] > lm.loc[('seq',) + i, 'mineracao'] for i in lidx)} |")
        w(f"| Configurações em que a GPU perde ponta a ponta | {len(tot_slower)} | {sum(x < 1 for x in l_tot)} |\n")
    w("A diferença entre os dois relatórios é o tamanho do problema. Com suportes reduzidos, quase todas as "
      "configurações têm trabalho de sobra e a GPU vence em praticamente tudo. Com os suportes padrão, várias "
      "configurações são pequenas demais: a GPU ainda vence nos esparsos e onde há muitos itemsets, mas perde "
      "onde o problema inteiro leva poucos milissegundos, por causa de custos fixos (lançamentos, alocações, "
      "duas passadas). Um deles, a preparação da mineração, ficava fora das etapas medidas até a correção "
      "da seção 11; a correção vale para os dois conjuntos.\n")
    w("---\n")

    # ------------------------------------------------------------ 15. reprod.
    w("## 15. Reprodução\n")
    w("```bash")
    w("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
    w("./scripts/download_data.sh")
    w("FPG_SET=common PY=/caminho/python3 ./scripts/run_all.sh")
    w("FPG_SET=common python3 scripts/relatorio_fases.py reports/relatorio_fases_suporte_padrao.md")
    w("```\n")
    w("| Arquivo | Conteúdo |")
    w("| --- | --- |")
    w("| `results/raw.csv` | tempo de cada etapa de cada execução (suportes padrão) |")
    w("| `results/runs.csv` | itemsets, nós, estouros, tarefas, desbalanceamento, memória e barramento PCI |")
    w("| `results/grain.csv` | experimento de granularidade |")
    w("| `results/blocks/*.csv` | tempo, tarefas e itemsets de cada bloco |")
    w("| `results/micro/` | limites de hardware, kernels de contagem isolados, estatísticas por configuração |")
    w("| `results/low_support/` | o mesmo para os suportes reduzidos (`FPG_SET=low`; a refazer, TODO 1.1) |")
    w("| `results/pre_fix/` | medições anteriores à correção da preparação da mineração |\n")
    w("---\n")

    # ------------------------------------------------------------ apêndices
    w("## Apêndice A — Tempo sequencial por fase (todas as configurações)\n")
    w("ms (% do total).\n")
    w("| Dataset | minsup | Leitura | Contagem | Pré-proc. | Construção | Mineração | Escrita | Total |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d, s in configs:
        w(f"| {d} | {sup(s)} | {seqrow(d, s)} |")
    w("\n## Apêndice B — Construção (todas as configurações)\n")
    w(head)
    w(sep)
    for d, s in configs:
        w(consrow(d, s))
    w("\n## Apêndice C — Mineração (todas as configurações)\n")
    w("ms; \"P1\" = fração do tempo na passada de contagem.\n")
    w("| Dataset | minsup | Itemsets | Seq. | Estática | Est. vs. seq. | Dinâmica | Din. vs. seq. | Din. vs. est. | "
      "P1 est. | P1 din. | Mais lento / média est. | Mais lento / média din. | Na fila |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d, s in configs:
        a, b, c = M("seq", d, s, "mineracao"), M("gpu_static", d, s, "mineracao"), M("gpu_dynamic", d, s, "mineracao")
        w(f"| {d} | {sup(s)} | {big(meta.loc[('seq', d, s), 'itemsets'])} | {num(a)} | {num(b)} | {sp(a / b)} | "
          f"{num(c)} | {sp(a / c)} | {sp(b / c)} | "
          f"{pct(100 * M('gpu_static', d, s, 'mineracao_passada1') / b)} | "
          f"{pct(100 * M('gpu_dynamic', d, s, 'mineracao_passada1') / c)} | "
          f"{num(meta.loc[('gpu_static', d, s), 'desbalanceamento'], 1)} | "
          f"{num(meta.loc[('gpu_dynamic', d, s), 'desbalanceamento'], 2)} | "
          f"{num(meta.loc[('gpu_dynamic', d, s), 'enfileiradas'], 0)} |")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("gravado", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reports", "relatorio_fases_suporte_padrao.md"))
