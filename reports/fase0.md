# Relatório — Fase 0: ambiente e datasets

**Status:** concluída. Critérios de aceite atendidos.

## O que foi feito

| Tarefa | Resultado |
| --- | --- |
| 0.1 Estrutura e CMake | Três alvos (`fpg_seq`, `fpg_gpu`, `tests`) compilam com `-O3`, arquitetura `native` (sm_89) e `-Wall -Wextra`, **sem warnings**. |
| 0.2 Ambiente | 2× RTX 4090, Ryzen 9 7900, CUDA 12.4, g++ 13 (registrado em `NOTES.md`). |
| 0.3 Download | `scripts/download_data.sh` baixa os 6 datasets do espelho Wayback (o site FIMI original está fora do ar). |
| 0.4 Datasets minúsculos | 5 arquivos em `tests/data/` com saída esperada feita à mão, incluindo o exemplo de Han, Pei e Yin (18 itemsets com minsup 3). |
| 0.5 Leitor FIMI → CSR | `src/common/fimi.{hpp,cpp}`: `items[]` int32 e `offsets[]` (n+1). |
| 0.6 Saída canônica e `CUDA_CHECK` | `src/common/output.*`, `src/common/cuda_check.cuh`. |
| 0.7 Estatísticas | Tabela em `NOTES.md`. |

## Estatísticas lidas × publicadas

| Dataset | Transações | Itens | Tam. médio | Confere? |
| --- | ---: | ---: | ---: | :---: |
| chess | 3.196 | 75 | 37,0 | sim |
| mushroom | 8.124 | 119 | 23,0 | sim |
| retail | 88.162 | 16.470 | 10,3 | sim |
| T10I4D100K | 100.000 | 870 | 10,1 | sim |
| kosarak | 990.002 | 41.270 | 8,1 | sim |
| accidents | 340.183 | 468 | 33,8 | sim |

## Problemas encontrados e contornos

- **Sem cmake no sistema** → instalado via pip num venv local (`../.venv`), sem mexer no sistema.
- **nvcc 12.4 não aceita g++ 15** (padrão do sistema) → o CMake seleciona `g++-13` automaticamente.
- **Site FIMI fora do ar** → espelho da Wayback Machine; `kosarak` e `accidents` via `.gz` porque o `.dat` vem truncado.

## Uso compartilhado da máquina

As duas GPUs estavam ociosas no início. Nenhum processo de outro usuário foi tocado.
