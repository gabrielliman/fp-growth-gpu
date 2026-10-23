# Relatório — Fase 6: experimentos e análise

**Status:** concluída. Critérios de aceite atendidos: todas as medidas válidas,
reproduzíveis por `scripts/run_all.sh`, gráficos com eixos rotulados e unidades.

Relatório completo com gráficos e tabelas: `results/RESULTS.md` na época. Esse arquivo foi
substituído depois por `results/tables.md` e pelos relatórios gerados em `reports/`
(ver [relatorio_fases_suporte_padrao.md](relatorio_fases_suporte_padrao.md)).

## Execução

- 360 medidas principais + 180 de grain + instrumentação por bloco, todas com a
  saída conferida contra o sequencial (hash SHA-256): **0 inválidas**.
- Processos limitados a 20 GB de RAM por cgroup. Saídas temporárias no disco,
  não no tmpfs. Durante a rodada outras pessoas usaram as GPUs; o script
  detectou e esperou, sem interferir.

## Respostas

**1. Construção por primitivas × sequencial.** 11–37× mais rápida nas árvores
grandes (0,5–3 M nós). A ordenação lexicográfica domina (43–86 %); sem ela, o
resto é 34–170× mais rápido. Nas árvores pequenas (chess, mushroom) a GPU perde
pelo custo fixo (~2–3 ms).

**2. Dinâmica × estática.** A dinâmica vence nas 24 configurações (1,1–32×) e
minera 1,8–43× mais rápido que o sequencial. Ganha mais quando há poucos itens
(chess) ou quando a carga se concentra em poucos itens (kosarak, retail); ganha
pouco quando a carga já é uniforme (T10I4D100K) ou quando as primeiras árvores
condicionais são gigantes e indivisíveis (accidents).

| Medida | Estática | Dinâmica |
| --- | --- | --- |
| máx/média do tempo por bloco | 3,4 – 1.644 | 1,0 – 2,3 (19/24); até 10,1 em accidents |
| Mineração vs. sequencial | 0,3× – 18× | 1,8× – 43× |

**Grain:** ótimo entre 16 e 64 itens; ≥ 1024 volta a ser estática.

**Ponta a ponta:** núcleo (contagem → mineração) 3,4–39× mais rápido; no total o
ganho é limitado pela escrita canônica na CPU (61 dos 80 s em mushroom 0,01).

## Gráficos

| | |
| --- | --- |
| (a) speedup da construção | `results/fig_a_construcao_speedup.png` |
| (b) tempo total × suporte | `results/fig_b_total_vs_suporte.png` (+ `fig_b2` só mineração) |
| (c) tempo por etapa | `results/fig_c_etapas.png` |
| (d) tempo por bloco | `results/fig_d_blocos.png` |
| (e) efeito do grain | `results/fig_e_grain.png` |
