# Relatório — Fase 3: construção da FP-tree por primitivas paralelas

**Status:** concluída. Critérios de aceite atendidos.

## Como funciona

A FP-tree é construída sem nenhuma inserção concorrente. Os nós são os prefixos
distintos das transações ordenadas:

1. **Sort** — `thrust::sort` dos índices das transações com comparador lexicográfico.
2. **LCP** — 1 thread por transação calcula o prefixo comum com a anterior.
3. **Scan** — `new_i = len_i − LCP_i`; `exclusive_scan` dá o id do primeiro nó novo.
4. **Nós** — item, profundidade e transação criadora de cada nó.
5. **Sparse table** sobre o LCP (log n passos paralelos).
6. **Pais** — previous smaller value por busca binária na sparse table.
7. **Contadores** — next smaller or equal: `count = W[m] − W[k]`, sem atômicos (já ponderado → 4.1).
8. **Cabeçalho** — `stable_sort_by_key` por item + `lower_bound` + `reduce_by_key`.

## Corretude

| Verificação | Resultado |
| --- | --- |
| Árvore idêntica à sequencial (texto canônico + nº de nós) | 19/19 configurações, 6 datasets, 3–4 suportes cada |
| Contadores 3.6 (sem atômicos) = 3.7 (`reduce_by_key`) | 19/19 |
| Cabeçalho (suporte e offsets por item) | 19/19 |
| Alvo `tests` (bordas + 30 bases aleatórias + 20 ponderadas) | 339 verificações, 0 falhas |

Maior árvore validada: kosarak 0,0001, **4.581.576 nós**.

## Desempenho (mediana de 5 execuções)

| Dataset | minsup | nós | seq (ms) | GPU (ms) | speedup | % sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 15,7 k | 1,05 | 2,74 | 0,4× | 80 % |
| mushroom | 0,05 | 22,3 k | 2,27 | 3,73 | 0,6× | 87 % |
| retail | 0,0001 | 653 k | 45,7 | 2,56 | 17,8× | 48 % |
| T10I4D100K | 0,0001 | 715 k | 50,5 | 3,34 | 15,1× | 60 % |
| accidents | 0,1 | 2,87 M | 394,5 | 11,13 | **35,4×** | 78 % |
| kosarak | 0,0001 | 4,58 M | 549,3 | 10,17 | **54,0×** | 63 % |

- **A ordenação lexicográfica domina**, como previsto no plano. O cabeçalho
  (outra ordenação) vem em segundo.
- Pais e contadores pela sparse table custam < 0,2 ms mesmo com 4,6 M nós.
- O speedup cresce com o tamanho da árvore. Nos datasets minúsculos a GPU perde
  por custo fixo de ~2,5 ms.
- Memória de pico da construção: até 228 MB (kosarak 0,0001).

## Incidente e medida de proteção

Na primeira rodada do benchmark, o **sequencial** (pipeline completo, com
mineração) em kosarak 0,0001 esgotou a RAM e foi morto. A memória da máquina
voltou ao normal logo em seguida (8 GB usados, igual ao início). Desde então
todo processo pesado roda num cgroup com `MemoryMax=20G`
(`systemd-run --user --scope`): se estourar, só o nosso processo morre, e o OOM
killer do sistema não chega a escolher processos de outros usuários. O
benchmark de construção agora usa `--no-mine`.
