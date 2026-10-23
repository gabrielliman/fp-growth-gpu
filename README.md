# FP-Growth paralelo em GPU

Implementação do FP-Growth em CUDA em que **todas as fases do núcleo rodam na
GPU**: contagem de suporte, pré-processamento, construção da FP-tree e
mineração. Uma versão sequencial em C++ serve de baseline e de oráculo de
corretude.

- **Construção da FP-tree sem inserções concorrentes.** Depois de ordenar as
  transações lexicograficamente, os nós da árvore são exatamente os prefixos
  distintos. Ids, pais e contadores saem de um vetor LCP, de um *prefix sum*
  e de consultas de mínimo numa *sparse table*.
- **Mineração em duas variantes.** *Estática* (um bloco de threads por item)
  e *dinâmica* (fila global de tarefas sem travas, consumida por blocos
  persistentes). As árvores condicionais são construídas por bloco com o mesmo
  código da construção global.
- **Baselines em GPU para comparação** (em andamento): Apriori no estilo
  GPApriori e Eclat no estilo Frontier Expansion, com bitsets verticais.

Toda saída da GPU é comparada byte a byte com a do sequencial, que por sua
vez é idêntica à referência de Christian Borgelt.

## Começando

```bash
python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
export PATH=$PWD/../.venv/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./scripts/download_data.sh
./build/tests
./build/fpg_seq data/chess.dat --minsup 0.6 --out seq.txt
./build/fpg_gpu data/chess.dat --minsup 0.6 --mine=dynamic --out gpu.txt
cmp seq.txt gpu.txt && echo "saídas idênticas"
```

Requisitos: GPU NVIDIA, CUDA 12 (testado com 12.4), g++ ≤ 13 para o nvcc 12.4,
CMake ≥ 3.24 e Python 3 com pandas e matplotlib.

**O guia completo, com cada parte separadamente, está em [TUTORIAL.md](TUTORIAL.md).**

## Estado atual

| Parte | Estado |
| --- | --- |
| Sequencial, GPU estática, GPU dinâmica | completos e validados (suíte com 584 verificações; benchmarks com hash de saída) |
| Benchmarks com os suportes padrão (`results/`) | completos, com a correção da mineração |
| Benchmarks com os suportes reduzidos (`results/low_support/`) | a refazer (TODO 1.1) |
| Apriori em GPU | implementado e validado; fora do benchmark (TODO 1.5) |
| Eclat em GPU | implementado, **com bug** (TODO 1.4) |
| Artigo IEEE (`paper/`) | escrito sobre os suportes reduzidos e anterior à correção; a reescrever (TODO 1.6) |

O que falta, com contexto e comandos: [TODO.md](TODO.md).

## Resultados (suportes padrão)

Resumo de [reports/relatorio_fases_suporte_padrao.md](reports/relatorio_fases_suporte_padrao.md)
(RTX 4090, 6 datasets × 4 suportes, mediana de 5 execuções):

- **Construção:** ganha nas árvores com 100 mil nós ou mais e perde nas
  árvores pequenas de chess e mushroom, por custo fixo (~2–3 ms) da ordenação.
- **Mineração dinâmica:** perde da CPU onde a mineração inteira leva poucos ms
  e chega a dezenas de vezes mais rápida onde há carga grande e desigual.
- **Ponta a ponta:** limitado pela leitura e pela escrita, que continuam na CPU.

Os números exatos estão no relatório e em `results/tables.md`.

## Documentação

| Arquivo | Para quê |
| --- | --- |
| [TUTORIAL.md](TUTORIAL.md) | instalar, compilar, rodar cada programa, testar, medir, gerar gráficos e relatórios |
| [TODO.md](TODO.md) | pendências priorizadas |
| [NOTES.md](NOTES.md) | diário técnico: decisões, problemas encontrados e correções |
| [docs/plano_original.md](docs/plano_original.md) | o plano em fases que guiou a implementação |
| [reports/](reports/) | relatórios por fase (o atual e os históricos) |

## Estrutura

```
src/common/    leitura FIMI → CSR, saída canônica, timers, argumentos, CUDA_CHECK
src/seq/       FP-Growth sequencial
src/gpu/       preprocess.cu, build_tree.cu + tree_core.cuh, mine.cu, main_gpu.cu
src/baselines/ Apriori e Eclat em GPU
tests/         suíte de testes, microbenchmark e bases minúsculas
scripts/       download, validação, benchmarks, gráficos, relatórios, artigo
results/       CSVs, gráficos e tabelas dos benchmarks
reports/       relatórios
paper/         artigo IEEE em LaTeX
docs/          plano original
```
