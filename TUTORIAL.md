# Tutorial: como rodar tudo (e cada parte separadamente)

Este guia leva de um clone limpo até os benchmarks, gráficos, relatórios e o
artigo. Cada seção pode ser seguida isoladamente, desde que o build (seção 3) e
os dados (seção 2) já existam.

Para entender o projeto antes de rodar: [README.md](README.md) (visão geral),
[docs/plano_original.md](docs/plano_original.md) (o plano em fases),
[reports/relatorio_fases_suporte_padrao.md](reports/relatorio_fases_suporte_padrao.md)
(resultados atuais) e [TODO.md](TODO.md) (o que falta).

## Sumário

0. [Mapa do repositório](#0-mapa-do-repositório)
1. [Pré-requisitos e instalação](#1-pré-requisitos-e-instalação)
2. [Dados](#2-dados)
3. [Compilação](#3-compilação)
4. [Executar cada programa](#4-executar-cada-programa)
5. [Testes e validação](#5-testes-e-validação)
6. [Máquina compartilhada: GPU fixa e limite de memória](#6-máquina-compartilhada-gpu-fixa-e-limite-de-memória)
7. [Benchmarks](#7-benchmarks)
8. [Gráficos, tabelas e relatórios](#8-gráficos-tabelas-e-relatórios)
9. [Artigo (LaTeX)](#9-artigo-latex)
10. [Tudo de uma vez: `run_all.sh`](#10-tudo-de-uma-vez-run_allsh)
11. [Onde mexer no código](#11-onde-mexer-no-código)
12. [Problemas comuns](#12-problemas-comuns)

---

## 0. Mapa do repositório

```
CMakeLists.txt        alvos: fpg_seq, fpg_gpu, fpg_apriori_gpu, fpg_eclat_gpu, tests, microbench
src/common/           leitura FIMI → CSR, saída canônica, timers, argumentos, CUDA_CHECK
src/seq/              FP-Growth sequencial (baseline e oráculo de corretude)
src/gpu/              FP-Growth em GPU: preprocess.cu, build_tree.cu + tree_core.cuh, mine.cu, main_gpu.cu
src/baselines/        Apriori e Eclat em GPU com bitsets verticais (comparação; Eclat com bug, ver TODO)
tests/                tests.cu (suíte), microbench.cu (limites do hardware), data/ (bases minúsculas)
scripts/              download, validação, benchmarks, gráficos, relatórios, artigo
results/              CSVs e gráficos dos suportes padrão; micro/ (hardware); pre_fix/ (antes da correção)
reports/              relatórios por fase (históricos e atuais)
paper/                artigo IEEE em LaTeX (ver seção 9 e TODO 1.6)
docs/                 plano original do projeto
NOTES.md              diário técnico com todas as decisões
TODO.md               pendências
```

`data/` (datasets) e `build/` não são versionados.

## 1. Pré-requisitos e instalação

### 1.1 Obrigatórios

| Item | Versão usada | Observação |
| --- | --- | --- |
| GPU NVIDIA | RTX 4090 (sm_89) | qualquer GPU com CUDA 12 serve; a arquitetura é detectada |
| CUDA Toolkit | 12.4 | traz Thrust e CUB; nenhuma outra biblioteca de GPU é usada |
| Compilador C++17 aceito pelo nvcc | g++ 13 | **nvcc 12.4 não aceita g++ ≥ 14**; o CMake escolhe `g++-13` sozinho se existir |
| CMake | ≥ 3.24 | se o sistema não tiver, instale no venv (abaixo) |
| Python 3 | 3.13 | com pandas, matplotlib e numpy |
| `systemd-run --user` | — | usado para limitar a memória de cada execução (seção 6) |

### 1.2 Ambiente Python (inclui o cmake)

Sem mexer no sistema:

```bash
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements.txt
export PATH=$PWD/../.venv/bin:$PATH      # deixa o cmake do venv no PATH
```

Nos exemplos abaixo, `PY` é esse interpretador: `PY=$PWD/../.venv/bin/python3`.

### 1.3 Opcionais

- **Referência de Christian Borgelt** (valida o sequencial):
  ```bash
  mkdir -p ../ref && cd ../ref
  curl -sL https://borgelt.net/src/fpgrowth.tar.gz | tar xz
  make -C fpgrowth/src          # gera ../ref/fpgrowth/src/fpgrowth
  cd -
  ```
- **tectonic** (compila o artigo), num ambiente conda local:
  ```bash
  conda create -y -p ../.tex-env -c conda-forge tectonic poppler
  # tectonic: ../.tex-env/bin/tectonic ; pdftoppm (para revisar o PDF): ../.tex-env/bin/pdftoppm
  ```

## 2. Dados

```bash
./scripts/download_data.sh
```

Baixa chess, mushroom, retail, T10I4D100K, kosarak e accidents para `data/`.
O site FIMI original está fora do ar; o script usa o espelho da Wayback
Machine (kosarak e accidents vêm como `.gz`, porque o `.dat` do espelho vem
truncado). Para conferir as estatísticas:

```bash
for f in data/*.dat; do ./build/fpg_seq $f --stats; done
# dataset,transações,itens distintos,tamanho médio,tamanho máximo,maior id
```

## 3. Compilação

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Alvos gerados em `build/`:

| Binário | O que é |
| --- | --- |
| `fpg_seq` | FP-Growth sequencial em C++ |
| `fpg_gpu` | FP-Growth com contagem, pré-processamento, construção e mineração na GPU |
| `fpg_apriori_gpu` | Apriori em GPU (estilo GPApriori) |
| `fpg_eclat_gpu` | Eclat em GPU (estilo Frontier Expansion); **ainda com bug** |
| `tests` | suíte de testes de corretude |
| `microbench` | banda D2D, banda H2D e radix sort de referência |

Para outro compilador host: `-DCMAKE_CUDA_HOST_COMPILER=/caminho/g++`. O build
usa `-O3` e `-Wall -Wextra` e deve sair sem warnings.

## 4. Executar cada programa

Todos recebem um arquivo FIMI (uma transação por linha, itens inteiros) e o
suporte mínimo como **fração**; a contagem absoluta é `ceil(minsup · n)`.

### 4.1 Formato de saída e de tempos

- `--out arquivo`: saída **canônica**, uma linha por itemset, `i1 i2 … (suporte)`,
  itens em ordem crescente de id e linhas ordenadas byte a byte. Duas saídas
  se comparam com `cmp`.
- **stdout**: uma linha CSV por etapa, `impl,dataset,minsup,etapa,tempo_ms`. As
  etapas são `leitura`, `transferencia_h2d`, `contagem`, `preprocessamento`,
  `construcao` (e subetapas `construcao_sort`, `_lcp`, `_scan`, `_nos`,
  `_sparse`, `_pais`, `_contadores`, `_cabecalho`), `mineracao` (e
  `mineracao_preparacao`, `_passada1`, `_passada2`), `transferencia_d2h`,
  `escrita` e `total`.
- **stderr**: resumo `chave=valor` (itens frequentes, nós da árvore, itemsets,
  tarefas, estouros, desbalanceamento, memória, barramento PCI).

### 4.2 Sequencial

```bash
./build/fpg_seq data/chess.dat --minsup 0.6 --out seq.txt
./build/fpg_seq data/chess.dat --minsup 0.6 --dump-tree arvore.txt   # FP-tree em texto
./build/fpg_seq data/kosarak.dat --minsup 0.001 --no-mine            # só até a construção
```

### 4.3 GPU

```bash
./build/fpg_gpu data/chess.dat --minsup 0.6 --mine=dynamic --out gpu.txt
cmp seq.txt gpu.txt && echo idênticas
```

| Opção | Efeito |
| --- | --- |
| `--mine=static\|dynamic` | mineração estática (um bloco por item) ou dinâmica (fila global); padrão `static` |
| `--grain N` | na dinâmica, árvores condicionais com base ≥ N itens vão para a fila (padrão 64) |
| `--count=v1\|v2` | contagem com atômicos globais (v1) ou histograma em memória compartilhada (v2, padrão) |
| `--debug` | compara suportes, CSR, árvore e itemsets com o sequencial e imprime OK/FALHOU |
| `--dump-tree f` | FP-tree global no mesmo formato do sequencial |
| `--no-mine` | para após a construção |
| `--block-stats f.csv` | tempo, tarefas e itemsets de cada bloco (instrumentação com `clock64`) |
| `--count-bench N` | mede os kernels de contagem v1 e v2 (média de N lançamentos) e sai |
| `--arena-kb K` | arena por bloco em KB; valores pequenos forçam o caminho de estouro |
| `--pool-mb M`, `--queue Q` | pool e capacidade da fila da dinâmica (testes de degradação) |
| `--device D` | índice CUDA (prefira fixar a GPU pela seção 6) |

Exemplo de verificação completa de uma configuração:

```bash
./build/fpg_gpu data/kosarak.dat --minsup 0.002 --mine=dynamic --debug
# [debug] suportes OK, tabela de ranks OK, CSR OK, nós OK, FP-tree OK, contadores OK,
#         cabeçalho OK (2x), itemsets idênticos ao sequencial OK
```

### 4.4 Apriori e Eclat em GPU

```bash
./build/fpg_apriori_gpu data/chess.dat --minsup 0.8 --out apriori.txt
./build/fpg_eclat_gpu   data/chess.dat --minsup 0.8 --out eclat.txt [--pool-slots N]
```

Mesmo formato de saída e de CSV. O Eclat ainda dá resultados errados em alguns
casos (TODO 1.4); não use os números dele.

## 5. Testes e validação

### 5.1 Suíte de testes

```bash
./build/tests                 # 584 verificações: bases à mão, casos de borda, 30 aleatórias, árvores ponderadas
./build/tests --no-mining     # só pré-processamento e árvore (mais rápido)
./build/tests --with-eclat    # inclui o Eclat em GPU (falha hoje; TODO 1.4)
```

### 5.2 GPU × sequencial em datasets reais

```bash
./scripts/validate_gpu.sh "--mine=dynamic" chess:0.6 kosarak:0.002 accidents:0.4
./scripts/validate_gpu.sh "--mine=static --arena-kb 512" kosarak:0.002   # força estouros de arena
./scripts/validate_gpu.sh "--mine=dynamic --grain 0 --queue 4" chess:0.6 # fila cheia
```

Cada caso roda `fpg_gpu --debug` e imprime `OK`/`FAIL` com itemsets, tarefas,
tarefas enfileiradas e estouros.

### 5.3 Sequencial × Borgelt

```bash
./scripts/check_reference.sh ../ref/fpgrowth/src/fpgrowth
```

Compara 12 configurações (chess, mushroom, retail, T10I4D100K). Usa
`scripts/canon.py` para normalizar a saída do Borgelt.

### 5.4 Estresse da mineração dinâmica

```bash
./scripts/stress_dynamic.sh > results/stress_dynamic.txt   # 7 configs × 3 grains × 20 repetições
REPS=3 ./scripts/stress_dynamic.sh                         # versão curta
```

Cada execução tem timeout (detecta deadlock) e a saída é comparada com a do
sequencial.

## 6. Máquina compartilhada: GPU fixa e limite de memória

A máquina original é compartilhada e tem duas RTX 4090 em links PCIe
diferentes (x16 e x4). Três regras valem para todos os scripts:

1. **Uma GPU só.** Todas as medições usam a placa do barramento
   `FPG_GPU_BUS` (padrão `00000000:01:00.0`, o slot x16). `scripts/gpu_env.sh`
   (shell) e `scripts/gpuutil.py` (Python) a passam ao CUDA pelo UUID. Em outra
   máquina, veja o barramento com
   `nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.width.current --format=csv`
   e exporte `FPG_GPU_BUS`. Para rodar à mão na mesma GPU:
   `source scripts/gpu_env.sh` antes dos comandos.
2. **Esperar, nunca interromper.** Se houver processo de outro usuário na
   GPU fixa, os benchmarks esperam 60 s e tentam de novo. Nenhum script mata
   processos.
3. **Limite de memória por execução.** Cada processo pesado roda em
   `systemd-run --user --scope -p MemoryMax=20G` (ajustável por
   `FPG_MEM_LIMIT`). Se estourar, só o nosso processo morre; o OOM killer
   global não chega a escolher processos de outros usuários.

Saídas temporárias dos benchmarks ficam em `results/tmp/` (disco), não em
`/tmp`, que nesta máquina é tmpfs e consome RAM.

## 7. Benchmarks

### 7.1 Conjuntos de thresholds

Definidos em `scripts/gpuutil.py` e escolhidos por `FPG_SET`:

| `FPG_SET` | Suportes | Saídas em |
| --- | --- | --- |
| `common` (padrão) | os mais usados na literatura FIMI: chess 0,8–0,5; mushroom 0,3–0,05; accidents 0,8–0,3; T10I4D100K, retail, kosarak 0,01–0,001 | `results/` |
| `low` | reduzidos, para o sequencial levar de 0,17 s a 80 s | `results/low_support/` |

`MID` (no mesmo arquivo) é o suporte intermediário usado nos experimentos de
grain e de blocos.

### 7.2 Cada experimento separadamente

Todos aceitam `FPG_SET` e `REPS` (medidas por configuração, padrão 5, mais 1
de aquecimento descartada). Toda execução tem o hash da saída comparado com o
do sequencial; uma diferença marca a medida como inválida (`valido=0`).

| Comando | O que mede | Gera |
| --- | --- | --- |
| `$PY scripts/bench_build.py` | construção da árvore GPU × sequencial, com subetapas (`--no-mine`) | `build.csv` |
| `$PY scripts/bench.py main` | as 3 implementações (`seq`, `gpu_static`, `gpu_dynamic`) em todas as configurações | `raw.csv`, `runs.csv` |
| `$PY scripts/bench.py grain` | dinâmica com grain 0, 16, 64, 256, 1024, 4096 no suporte `MID` | `grain.csv` |
| `$PY scripts/bench.py blocks` | instrumentação por bloco, estática e dinâmica, no `MID` | `blocks/*.csv` |
| `./scripts/config_stats.sh` | itens frequentes, transações, itens pré-processados e nós por configuração | `results/micro/config_stats_<set>.csv` |
| `./build/microbench <n_tx> <n_itens>` | banda D2D, H2D (paginável e pinned) e radix sort de 32/64 bits | stdout (ver `run_all.sh`) |
| `./build/fpg_gpu data/X.dat --count-bench 50` | kernels de contagem v1 e v2 isolados | stdout |

Os quatro primeiros comandos juntos levaram 8 min com `FPG_SET=common` na RTX
4090. Com `FPG_SET=low`, a rodada anterior levou ~2 h (mushroom 0,01 escreve
90 M itemsets por execução).

`bench.py main` é retomável: configurações (impl, dataset, suporte) já
presentes em `runs.csv` são puladas. Para refazer do zero, apague os CSVs.

### 7.3 Esquema dos CSVs

| Arquivo | Colunas |
| --- | --- |
| `raw.csv` | `impl,dataset,minsup,rep,etapa,tempo_ms,valido` (uma linha por etapa e execução) |
| `runs.csv` | `impl,dataset,minsup,rep,valido,wall_s,pci,itemsets,nos,estouros,enfileiradas,tarefas,desbalanceamento,mem_construcao_MB,mem_mineracao_MB,mem_gpu_pico_MB,blocos` |
| `grain.csv` | `dataset,minsup,grain,rep,valido,mineracao_ms,total_ms,enfileiradas,desbalanceamento` |
| `build.csv` | `dataset,minsup,rep,nos,mem_MB,seq_construcao_ms,` + subetapas da construção |
| `blocks/<impl>_<dataset>.csv` | `mode,launch,block,slot,tasks,itemsets,cycles,busy_cycles,ms,busy_ms` |

## 8. Gráficos, tabelas e relatórios

```bash
(cd scripts && FPG_SET=common $PY plots.py)
#   results/fig_a_construcao_speedup.png   speedup da construção
#   results/fig_b_total_vs_suporte.png     tempo total × suporte (+ fig_b2, só mineração)
#   results/fig_c_etapas.png               tempo por etapa
#   results/fig_d_blocos.png               tempo por bloco, estática × dinâmica
#   results/fig_e_grain.png                efeito do grain
#   results/tables.md                      todas as tabelas, mediana ± desvio

FPG_SET=common $PY scripts/relatorio_fases.py reports/relatorio_fases_suporte_padrao.md
FPG_SET=low    $PY scripts/relatorio_fases.py reports/relatorio_fases_suporte_reduzido.md
```

`relatorio_fases.py` gera o relatório detalhado por fase inteiramente a partir
dos CSVs, incluindo o antes/depois da correção da mineração (usa
`results/pre_fix/`).

Outros scripts de figura:

- `scripts/slides_figs.py <raw_comum.csv> <raw_baixo.csv> <pasta>`: figura da
  apresentação de resultados parciais (só dados do sequencial). Ex.:
  `cd scripts && $PY slides_figs.py ../results/raw.csv ../results/pre_fix/raw_low.csv /tmp/fig`.
- `scripts/plot_blocks.py saida.png rotulo=arquivo.csv ...`: histograma de
  tempo por bloco usado nos relatórios das Fases 4 e 5.

Relatórios em `reports/`:

| Arquivo | Conteúdo |
| --- | --- |
| `relatorio_fases_suporte_padrao.md` | **atual**: relatório por fase com os suportes padrão (gerado) |
| `relatorio_fases.md` | relatório por fase com os suportes reduzidos, escrito à mão, números de antes da correção da mineração |
| `fase0.md` … `fase6.md` | estado de cada fase quando foi concluída (histórico) |

## 9. Artigo (LaTeX)

```bash
PY=$PY TECTONIC=../.tex-env/bin/tectonic ./paper/build.sh   # gera paper/main.pdf
```

O script roda `scripts/paper_data.py` (números e tabelas em
`paper/generated/`, como macros `\V{chave}`), `scripts/paper_figs.py`
(figuras em PDF em `paper/figures/`) e compila `paper/main.tex` com tectonic,
que baixa a classe IEEEtran na primeira vez.

**Estado:** o texto foi escrito sobre o conjunto reduzido. `build.sh` força
`FPG_SET=low` e precisa de `results/low_support/` preenchido (TODO 1.1). O
`paper/main.pdf` versionado é da versão anterior à correção da mineração. A
reescrita com os suportes padrão é o TODO 1.6.

Para revisar páginas como imagem:
`../.tex-env/bin/pdftoppm -r 80 -png paper/main.pdf /tmp/pagina`.

## 10. Tudo de uma vez: `run_all.sh`

```bash
PY=$PY BORGELT=../ref/fpgrowth/src/fpgrowth ./scripts/run_all.sh
```

Etapas, na ordem: `build data tests reference validate stress bench micro report paper`.
Para rodar só algumas:

```bash
STEPS="build tests validate" ./scripts/run_all.sh                 # só corretude
STEPS="bench micro report" PY=$PY ./scripts/run_all.sh           # benchmarks + relatório (common)
STEPS="bench report" FPG_SET=low PY=$PY ./scripts/run_all.sh     # conjunto reduzido (TODO 1.1)
```

Com `FPG_SET=common`, a etapa `bench` leva ~8 min; o estresse (420 execuções) é
a etapa mais longa.

## 11. Onde mexer no código

| Quero mudar… | Arquivo |
| --- | --- |
| leitura do FIMI, formato CSR | `src/common/fimi.{hpp,cpp}` |
| formato da saída canônica | `src/common/output.{hpp,cpp}` |
| nomes/medição das etapas | `src/common/timer.hpp` e os `main_*` |
| algoritmo sequencial (oráculo) | `src/seq/fpgrowth_seq.cpp` |
| contagem e pré-processamento na GPU | `src/gpu/preprocess.cu` |
| construção da FP-tree (grade) | `src/gpu/build_tree.cu` |
| funções de construção compartilhadas por grade e bloco | `src/gpu/tree_core.cuh` |
| mineração estática e dinâmica, arena, fila, duas passadas, estouro | `src/gpu/mine.cu` |
| interfaces e opções da GPU | `src/gpu/gpu.cuh` |
| linha de comando do `fpg_gpu` | `src/gpu/main_gpu.cu` |
| Apriori/Eclat em GPU | `src/baselines/vertical.cu` |
| testes | `tests/tests.cu` |
| configurações dos benchmarks, GPU fixa, limite de memória | `scripts/gpuutil.py` |

Regra do projeto: toda saída da GPU tem que ser idêntica à do sequencial.
Uma diferença é sempre bug; não ajuste o teste para passar.

## 12. Problemas comuns

| Sintoma | Causa e solução |
| --- | --- |
| `nvcc` reclama da versão do g++ | nvcc 12.4 só aceita g++ ≤ 13; instale `g++-13` ou passe `-DCMAKE_CUDA_HOST_COMPILER` |
| `cmake: command not found` | instale no venv (`pip install cmake`) e ponha `../.venv/bin` no PATH |
| `GPU 00000000:01:00.0 não encontrada` | outra máquina: exporte `FPG_GPU_BUS` com o barramento da sua GPU |
| execução morta com código 137 | estourou o limite de memória do cgroup (saída grande demais); aumente `FPG_MEM_LIMIT` com cuidado numa máquina compartilhada |
| benchmark parado em `[espera] GPU … em uso por outros` | outro usuário está usando a GPU fixa; o script continua sozinho quando ela liberar |
| `ERR_NVGPUCTRPERM` no `ncu` | contadores de desempenho exigem permissão de administrador; não contorne |
| `paper/build.sh` falha com `undefined value …` | o artigo precisa do conjunto reduzido (`results/low_support/`); ver seção 9 |
| `./build/tests --with-eclat` falha | esperado hoje (TODO 1.4) |
