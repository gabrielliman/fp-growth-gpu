# Relatório — Fase 1: baseline sequencial

**Status:** concluída. Critérios de aceite atendidos.

## Resumo

`fpg_seq` implementa o FP-Growth clássico em C++17: contagem, ranks por
suporte decrescente (desempate por id crescente), pré-processamento, FP-tree com
nós indexados e tabela de cabeçalho com lista ligada por item, mineração
recursiva com atalho de caminho único, saída canônica, CSV de tempos e dump
textual da árvore (`--dump-tree`).

## Corretude

| Teste | Resultado |
| --- | --- |
| 5 datasets minúsculos (saída calculada à mão) | 5/5 idênticos |
| Referência: `fpgrowth` 6.21 de C. Borgelt | 12/12 idênticos |

Casos contra a referência: chess (0,9 · 0,8 · 0,7 · 0,6), mushroom
(0,3 · 0,2 · 0,1 · 0,05), retail (0,01 · 0,005), T10I4D100K (0,01 · 0,005).
O maior caso tem 3.755.511 itemsets (mushroom 0,05). Os números conferem com a
literatura (ex.: chess 0,6 → 254.944 itemsets).

Única discrepância encontrada: o Borgelt emite o conjunto vazio; o script de
normalização o descarta.

## Onde o tempo se concentra

| Dataset | minsup | construção (ms) | mineração (ms) | escrita (ms) | total (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| chess | 0,5 | 1,3 | 280,6 | 664,6 | 948,9 |
| mushroom | 0,05 | 2,3 | 243,4 | 2.034,6 | 2.283,2 |
| accidents | 0,3 | 98,5 | 144,7 | 55,7 | 462,2 |
| T10I4D100K | 0,001 | 49,6 | 68,8 | 7,0 | 142,4 |
| retail | 0,001 | 23,8 | 31,6 | 1,7 | 73,7 |
| kosarak | 0,001 | 300,6 | 1.114,4 | 446,6 | 2.084,1 |

(Tabela completa em `NOTES.md`.)

- **Densos (chess, mushroom):** árvore pequena (≤ 22 k nós), custo na mineração e na escrita.
- **Esparsos/grandes (kosarak, T10I4D100K, accidents):** árvore grande (até 2,9 M nós);
  a construção é 15–35 % do total. É onde a construção paralela da Fase 3 tem mais a ganhar.

## Pontos de atenção para as próximas fases

- O sequencial é rápido: a maioria das configurações sugeridas para a Fase 6 roda em < 1 s.
  Os suportes terão de ser reduzidos para chegar à faixa de 1 s – 10 min.
- A escrita canônica (formatação + ordenação de strings) é compartilhada entre CPU e GPU e
  pesa muito nos densos; os experimentos devem reportar tempos com e sem ela.
