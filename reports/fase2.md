# Relatório — Fase 2: contagem e pré-processamento na GPU

**Status:** concluída. Critérios de aceite atendidos.

## Resumo

Toda a preparação dos dados roda na GPU: contagem de suporte, tabela de ranks,
filtragem, remapeamento, compactação e ordenação de cada transação. O resultado
fica em CSR na memória da GPU, pronto para a construção da árvore.

| Passo | Primitiva |
| --- | --- |
| Contagem v1 | `atomicAdd` global por ocorrência |
| Contagem v2 | histograma por bloco em shared (faixas de 12 K ids) + `atomicAdd` por bloco |
| Ranks | `copy_if` + `thrust::sort` (suporte desc, id asc) + `scatter` |
| Filtragem/remapeamento | kernel, 1 thread por item |
| Compactação | `exclusive_scan` + `DeviceSegmentedReduce::Sum` + `copy_if` + `inclusive_scan` |
| Ordenação por transação | `cub::DeviceSegmentedSort::SortKeys` |

## Corretude

Modo `--debug` compara com o sequencial: **suportes, ranks e CSR idênticos** em
todos os 6 datasets (9 configurações, incluindo retail com todos os 16.470
itens frequentes e kosarak com 4 faixas de histograma) e nos 5 minúsculos,
com v1 e com v2.

## Contagem v1 × v2

| Dataset | v1 (ms) | v2 (ms) | ganho |
| --- | ---: | ---: | ---: |
| chess | 0,027 | 0,013 | 2,1× |
| mushroom | 0,045 | 0,013 | 3,4× |
| retail | 0,094 | 0,024 | 3,9× |
| T10I4D100K | 0,181 | 0,017 | 10,9× |
| kosarak | 1,155 | 0,183 | 6,3× |
| accidents | 1,906 | 0,036 | **53×** |

A v1 sofre com contenção quando muitas ocorrências caem em poucos ids
(accidents). Em retail e kosarak, com muitos ids, a v2 precisa de várias faixas
(2 e 4) e relê a base por faixa, mas ainda ganha de 4 a 6×.

## Tempos de ponta (ms)

| Dataset | H2D | contagem | pré-proc. GPU | pré-proc. CPU (Fase 1) |
| --- | ---: | ---: | ---: | ---: |
| chess 0,5 | 0,97 | 0,02 | 0,68 | 1,0 |
| kosarak 0,001 | 2,96 | 0,20 | 2,86 | 65,4 |
| accidents 0,3 | 3,27 | 0,06 | 2,24 | 74,8 |

Nos datasets grandes o pré-processamento fica 23–33× mais rápido. A transferência
H2D já custa tanto quanto todo o pré-processamento.
