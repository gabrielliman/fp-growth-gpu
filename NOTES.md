# NOTES — FP-Growth paralelo em GPU

Diário técnico do projeto. Cada fase registra o que foi feito, as decisões e os
problemas em aberto. Relatórios detalhados por fase estão em `reports/`.

## Ambiente (Fase 0)

| Componente | Versão |
| --- | --- |
| GPU | 2× NVIDIA GeForce RTX 4090 (24 GB, sm_89, 128 SMs); usamos uma por execução |
| Driver | 595.91.07 (suporta CUDA 13.2) |
| CUDA Toolkit | 12.4 (nvcc V12.4.131), Thrust/CUB do próprio toolkit |
| CPU | AMD Ryzen 9 7900, 12 núcleos / 24 threads |
| RAM | 60 GB |
| Compilador host | g++ 13 (g++ 15 é o padrão do sistema, mas nvcc 12.4 não o aceita) |
| CMake | 4.4.4 (instalado via pip em `../.venv`, o sistema não tem cmake) |
| Python | 3.13 (venv `../.venv` com pandas e matplotlib) |
| SO | Linux 7.0.0-28-generic |

A máquina é compartilhada. Antes de rodar algo pesado na GPU, os scripts
verificam `nvidia-smi`; nenhum processo de outro usuário é interrompido.

**GPU fixa (decisão posterior à Fase 6):** as duas RTX 4090 estão em links PCIe
diferentes, `01:00.0` em Gen4 x16 (26,5 GB/s pinned) e `08:00.0` em x4
(6,4 GB/s). A primeira rodada de benchmarks escolhia a GPU mais livre, e por
isso os tempos de transferência variavam entre execuções (H2D de kosarak de
2,5 a 9 ms). Agora todas as medições usam só a placa `00000000:01:00.0` (x16),
passada ao CUDA pelo UUID (`scripts/gpu_env.sh`, `scripts/gpuutil.py`). Se ela
estiver ocupada, os scripts esperam. O `fpg_gpu` imprime `pci=` em cada
execução, e `scripts/paper_data.py` falha se alguma execução tiver rodado em
outra placa. Todos os resultados de `results/` foram refeitos nessa GPU.

## Datasets (Fase 0)

O site original `fimi.uantwerpen.be/data` está fora do ar (404). O script
`scripts/download_data.sh` usa o espelho da Wayback Machine. Para `kosarak` e
`accidents` o `.dat` do espelho vem truncado em 1 MiB, então baixamos o `.dat.gz`.

| Dataset | Transações | Itens distintos | Tam. médio | Tam. máximo | Maior id |
| --- | ---: | ---: | ---: | ---: | ---: |
| chess | 3.196 | 75 | 37,00 | 37 | 75 |
| mushroom | 8.124 | 119 | 23,00 | 23 | 119 |
| retail | 88.162 | 16.470 | 10,31 | 76 | 16.469 |
| T10I4D100K | 100.000 | 870 | 10,10 | 29 | 999 |
| kosarak | 990.002 | 41.270 | 8,10 | 2.497 | 41.270 |
| accidents | 340.183 | 468 | 33,81 | 51 | 468 |

Todos os números batem com as estatísticas publicadas do repositório FIMI
(ex.: chess 3.196 transações de 37 itens; accidents 340.183 transações).
Observação: chess tem 75 ids distintos no total, mas cada transação tem
exatamente 37 itens.

## Decisões gerais

- Formato canônico de saída: `i1 i2 ... ik (suporte)` com um espaço antes do
  parêntese, itens em ordem crescente de id original, arquivo ordenado como
  `LC_ALL=C sort`.
- `minsup` → contagem: `ceil(minsup * n - 1e-9)`, com n incluindo transações
  vazias. O epsilon evita que `0.01 * 100000` vire 1001 por arredondamento.
- Itens repetidos dentro de uma transação são removidos na leitura.
- Offsets em `int32` (o maior dataset tem ~8 M itens).
- CSV de tempos em stdout: `impl,dataset,minsup,etapa,tempo_ms`; logs em stderr.

## Fase 0 — concluída

Feito: estrutura do repositório, `CMakeLists.txt` (alvos `fpg_seq`, `fpg_gpu`,
`tests`; `-O3`; `CMAKE_CUDA_ARCHITECTURES native`; `-Wall -Wextra`), download
dos dados, leitor FIMI → CSR, escritor canônico, `CUDA_CHECK`, cinco datasets
minúsculos com saída esperada calculada à mão (`tests/data/`: exemplo de Han,
Pei e Yin; `small`; `edges` com duplicadas, prefixo e tamanho 1; `identical`;
`singletons`).

Em aberto: nada.

## Fase 1 — concluída

Feito: `src/seq/` com contagem (1.1), tabela de ranks com desempate por id
crescente (1.2), pré-processamento (1.3), FP-tree clássica (1.4), mineração
recursiva com atalho de caminho único (1.5), saída canônica (1.6), timers (1.7)
e dump textual da árvore (`--dump-tree`, 1.9).

Decisões:
- Nós guardados em vetores com índices (equivalente a ponteiros). Filhos numa
  lista de irmãos, mas a busca de filho usa uma tabela hash (pai, item) aberta,
  para não varrer listas longas na raiz (kosarak tem 41 k itens).
- As árvores condicionais mantêm a ordem global de ranks (não reordenam por
  frequência condicional), igual ao que a GPU fará na Fase 4.
- Transações que ficam vazias após a filtragem são descartadas.
- `mine_extensions` minera a partir de um sufixo usando a base projetada; servirá
  para as tarefas de estouro de memória da GPU (Fase 4.8).
- `scripts/canon.py` normaliza a saída do Borgelt e descarta o conjunto vazio,
  que ele emite e que não faz parte da saída deste projeto.

Validação (1.8): 5 datasets minúsculos idênticos à saída feita à mão; 12 casos
idênticos ao `fpgrowth` 6.21 de Christian Borgelt (`scripts/check_reference.sh`):
chess 0,9/0,8/0,7/0,6; mushroom 0,3/0,2/0,1/0,05; retail 0,01/0,005;
T10I4D100K 0,01/0,005.

Tempo por etapa do sequencial (ms, uma execução):

| Dataset | minsup | leitura | contagem | pré-proc. | construção | mineração | escrita | total | nós | itemsets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,7 | 1,3 | 0,1 | 0,5 | 0,6 | 11,5 | 23,1 | 37,0 | 2.168 | 48.731 |
| chess | 0,5 | 1,3 | 0,1 | 1,0 | 1,3 | 280,6 | 664,6 | 948,9 | 15.653 | 1.272.932 |
| mushroom | 0,1 | 1,4 | 0,1 | 1,0 | 1,7 | 46,7 | 273,8 | 324,7 | 14.322 | 574.431 |
| mushroom | 0,05 | 1,3 | 0,1 | 1,5 | 2,3 | 243,4 | 2.034,6 | 2.283,2 | 22.292 | 3.755.511 |
| accidents | 0,6 | 84,0 | 4,5 | 52,6 | 32,2 | 3,2 | 0,8 | 177,3 | 41.554 | 2.074 |
| accidents | 0,3 | 84,0 | 4,5 | 74,8 | 98,5 | 144,7 | 55,7 | 462,2 | 528.018 | 149.545 |
| T10I4D100K | 0,01 | 9,1 | 0,4 | 7,8 | 28,2 | 21,9 | 0,5 | 67,9 | 515.122 | 385 |
| T10I4D100K | 0,001 | 8,6 | 0,4 | 8,0 | 49,6 | 68,8 | 7,0 | 142,4 | 711.002 | 27.532 |
| retail | 0,01 | 9,9 | 0,4 | 3,3 | 2,4 | 0,8 | 0,4 | 17,2 | 31.035 | 159 |
| retail | 0,001 | 9,5 | 0,4 | 6,8 | 23,8 | 31,6 | 1,7 | 73,7 | 433.381 | 7.589 |
| kosarak | 0,01 | 155,5 | 4,1 | 35,4 | 33,4 | 18,7 | 0,2 | 247,2 | 294.539 | 383 |
| kosarak | 0,001 | 152,9 | 4,2 | 65,4 | 300,6 | 1.114,4 | 446,6 | 2.084,1 | 2.898.483 | 759.400 |

Onde está o tempo: nos densos (chess, mushroom) a mineração e a escrita
(formatar e ordenar milhões de linhas) dominam; a construção é desprezível.
Nos esparsos e grandes (kosarak, T10I4D100K, retail, accidents) a leitura, o
pré-processamento e a construção somam metade ou mais do tempo em suportes
altos, e a mineração passa a dominar só em suportes baixos.

Em aberto: os suportes sugeridos para a Fase 6 deixam o sequencial abaixo de 1 s
em quase todos os casos; será preciso descer os suportes (ver Fase 6).

## Fase 2 — concluída

Feito: `src/gpu/preprocess.cu` com transferência H2D (2.1), contagem v1 com
`atomicAdd` global (2.2), contagem v2 com histograma por bloco em memória
compartilhada (2.3), tabela de ranks com `thrust::sort` e o mesmo desempate
da CPU (2.4), kernel de filtragem/remapeamento (2.5), compactação com
`exclusive_scan` + `cub::DeviceSegmentedReduce::Sum` + `copy_if` dos tamanhos
não nulos (2.6), `cub::DeviceSegmentedSort::SortKeys` por transação (2.7) e
modo `--debug` que compara com o sequencial (2.8).

Decisões:
- v2 usa histogramas de 12 K contadores (48 KB de shared). Quando os ids não
  cabem, `gridDim.y` particiona os ids em faixas e cada bloco só conta a sua
  faixa (retail usa 2 faixas, kosarak 4). Cada faixa relê a base inteira, o que
  ainda é muito mais barato que a contenção da v1.
- v2 lança no máximo 4 blocos por SM: cada bloco varre muitos itens e amortiza
  zerar/descarregar o histograma.
- O contexto CUDA é inicializado (`cudaFree(0)`) antes de qualquer medição.

Validação: suportes, tabela de ranks e CSR pré-processado idênticos ao
sequencial nas duas versões de contagem, nos 6 datasets (chess 0,5; mushroom
0,05; retail 0,001 e 0,00001 — todos os 16.470 itens frequentes; T10I4D100K
0,001; kosarak 0,001 e 0,0005; accidents 0,3) e nos 5 datasets minúsculos.

Contagem v1 × v2 (2.9), média de 50 execuções, ms, incluindo alocação e zeragem
do vetor de suporte:

| Dataset | v1 | v2 | v1/v2 |
| --- | ---: | ---: | ---: |
| chess | 0,027 | 0,013 | 2,1× |
| mushroom | 0,045 | 0,013 | 3,4× |
| retail | 0,094 | 0,024 | 3,9× |
| T10I4D100K | 0,181 | 0,017 | 10,9× |
| kosarak | 1,155 | 0,183 | 6,3× |
| accidents | 1,906 | 0,036 | 53× |

A contenção da v1 cresce com a concentração de ocorrências em poucos ids:
accidents (34 itens por transação em 468 ids, alguns em quase todas as
transações) é o pior caso. chess tem a mesma característica, mas é pequeno
demais (118 k ocorrências) para saturar. A v2 é o padrão (`--count=v2`).

Tempos da GPU (ms): H2D 1–3,3; contagem 0,02–0,2; pré-processamento 0,7–2,9
(kosarak). No sequencial o pré-processamento custava 1–75 ms.

Em aberto: nada.

## Fase 3 — concluída

Feito: `src/gpu/build_tree.cu` + `src/gpu/tree_core.cuh`. Ordenação lexicográfica
com `thrust::sort` e comparador sobre segmentos do CSR, reescrita do CSR (3.1);
LCP (3.2); `exclusive_scan` dos nós novos (3.3); item/profundidade/caminho
criador (3.4); sparse table de mínimos em log n passos + busca binária para
pais (previous smaller value, 3.5) e contadores (next smaller or equal, 3.6,
generalizado para pesos com W = scan dos pesos, já cobrindo a 4.1);
verificação cruzada com `sort_by_key` + `reduce_by_key` (3.7); cabeçalho com
`stable_sort_by_key` + `lower_bound` + `reduce_by_key` (3.8); validação no
`--debug` e no alvo `tests` (3.9); medição (3.10, `scripts/bench_build.py`).

Decisões:
- As funções de dispositivo recebem `(tid, nthreads)` para servirem tanto a uma
  grade inteira quanto a um único bloco (Fase 4).
- Os contadores usam `node_tx[v]` (caminho que criou o nó): um kernel por nó em
  vez de um por transação, o que equilibra melhor a carga em kosarak
  (transações de até 2.497 itens).
- A sparse table virou uma subetapa própria (`construcao_sparse`) para que o
  custo de pais e contadores apareça separado do custo da tabela.
- O pico de memória é estimado amostrando `cudaMemGetInfo` ao fim de cada
  subetapa; não captura temporários internos do Thrust e tem granularidade de
  2 MB (por isso aparece 0 MB nos datasets pequenos).
- A versão LSD da ordenação (opcional em 3.1) não foi feita: o comparador não
  ficou lento demais (≤ 8,7 ms) e a LSD exigiria 2.497 passes em kosarak.
- Os benchmarks rodam dentro de `systemd-run --user --scope -p MemoryMax=20G`:
  numa primeira tentativa o sequencial minerando kosarak 0,0001 esgotou a RAM e
  foi morto. Com o cgroup, um estouro mata só o nosso processo e não aciona o
  OOM killer global contra processos de outros usuários.
- `--no-mine` em ambos os binários para medir só até a construção.

Validação: árvore idêntica à sequencial (formato textual, número de nós,
contadores 3.6 = 3.7, suporte e offsets do cabeçalho) em 19 configurações —
chess 0,8/0,6/0,5/0,3; mushroom 0,2/0,05/0,01; accidents 0,6/0,3/0,1;
T10I4D100K 0,01/0,001/0,0001; retail 0,01/0,001/0,00001; kosarak
0,01/0,001/0,0001. O alvo `tests` (339 verificações, 0 falhas) cobre os casos de
borda pedidos (duplicadas, prefixo da seguinte, tamanho 1, todas iguais, uma
transação, transações vazias, muitos irmãos na raiz), 30 bases aleatórias e 20
árvores ponderadas.

Tempos (mediana de 5, ms) — `results/build.csv`:

| Dataset | minsup | nós | seq | GPU | speedup | sort | LCP | scan | nós | sparse | pais | cont. | cabeç. | mem. GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 15.653 | 1,05 | 2,74 | 0,4× | 2,20 | 0,03 | 0,03 | 0,06 | 0,04 | 0,02 | 0,02 | 0,33 | <2 MB |
| mushroom | 0,05 | 22.292 | 2,27 | 3,73 | 0,6× | 3,23 | 0,03 | 0,03 | 0,06 | 0,05 | 0,02 | 0,02 | 0,33 | <2 MB |
| accidents | 0,1 | 2.869.852 | 394,5 | 11,13 | 35,4× | 8,69 | 0,22 | 0,03 | 0,51 | 0,16 | 0,07 | 0,06 | 1,40 | 126 MB |
| T10I4D100K | 0,0001 | 714.718 | 50,5 | 3,34 | 15,1× | 1,99 | 0,04 | 0,03 | 0,34 | 0,11 | 0,03 | 0,03 | 0,77 | 40 MB |
| retail | 0,0001 | 653.217 | 45,7 | 2,56 | 17,8× | 1,24 | 0,04 | 0,03 | 0,33 | 0,11 | 0,03 | 0,03 | 0,75 | 38 MB |
| kosarak | 0,0001 | 4.581.576 | 549,3 | 10,17 | 54,0× | 6,38 | 0,18 | 0,04 | 1,05 | 0,26 | 0,17 | 0,19 | 1,90 | 228 MB |

A ordenação lexicográfica domina em todos os casos (60–90 % da construção),
como previsto; o segundo maior custo é o cabeçalho (outra ordenação). Pais e
contadores via sparse table custam < 0,2 ms mesmo com 4,6 M nós. Nos datasets
minúsculos a GPU perde porque o custo fixo (~2,5 ms, dominado pelas alocações
e lançamentos do merge sort do Thrust) supera a construção sequencial inteira.

Em aberto: um alocador com cache para os temporários do Thrust reduziria o custo
fixo nos datasets pequenos (não feito).

## Fase 4 — concluída

Feito: `src/gpu/mine.cu`, kernel `mine_static_kernel` (um bloco de 256 threads
por item frequente da árvore raiz). Extração da base por bloco com scan das
profundidades (4.2); histograma ponderado, filtro e compactação (4.3);
construção condicional em nível de bloco reaproveitando `tree_core.cuh` com
`(threadIdx, blockDim)` (4.4); atalho de caminho único (4.5); laço com pilha
explícita (4.6); saída em duas passadas (4.7); lista de estouro (4.8);
instrumentação por bloco com `clock64` (4.9, `--block-stats`); validação (4.10).

Decisões:
- **Pais e contadores na árvore condicional** usam a mesma sparse table da
  Fase 3, construída em nível de bloco (log n passos com `__syncthreads`), em
  vez da varredura com pilha por uma thread. Ficou paralelo e reaproveita código
  já validado; não vira gargalo.
- **Ordenação lexicográfica em bloco**: merge sort de índices em que cada
  elemento acha sua posição na intercalação por busca binária na outra metade
  (estável). Evita `cub::BlockMergeSort`, que exige tamanho fixo por bloco.
- **Arena com duas pontas** por bloco: por baixo a pilha de quadros, o sufixo e
  as árvores condicionais (LIFO, liberadas ao desempilhar); por cima os
  temporários da construção (liberados ao fim de cada construção). Todas as
  threads mantêm cópias idênticas dos ponteiros da arena, já que fazem a mesma
  sequência de alocações.
- **Ocupação limitada a 2 blocos/SM** (256 blocos residentes) com 40 KB de
  memória compartilhada dinâmica por bloco. Ela guarda o histograma condicional
  (até 10.240 itens; acima disso vai para a arena) e garante arenas grandes:
  até 35 % da memória livre (máx. 6 GB) / 256 ≈ 24 MB por bloco. Um bloco pega
  uma arena livre por `atomicCAS` num vetor de travas; como no máximo 256 blocos
  ficam residentes, sempre há uma livre.
- **Estouro (4.8)**: o bloco registra o sufixo X = S ∪ {y} e segue. O host
  projeta a base `{t ∩ [0, min X) : t ⊇ X}` direto na GPU (as transações estão
  em ranks crescentes, então os itens < min X são um prefixo de cada uma), monta
  a árvore com `build_tree` da Fase 3 e relança a mineração com prefixo X. Não
  foi preciso cair no sequencial.
- **Saída**: `out_items`, `out_start` (int64), `out_len`, `out_sup`; a passada 1
  conta por raiz (64 bits), o que permite fazer a passada 2 em lotes de raízes
  cuja saída cabe em 256 M itens (plano B da tabela de contingências). Combinações
  de caminho único são contadas analiticamente: 2^L − 1 conjuntos e
  (2^L − 1)·|S| + L·2^(L−1) itens; na escrita, cada thread acha sua posição
  pela contagem de bits 1 dos inteiros menores que sua máscara.

Validação: idêntica ao sequencial em chess 0,8/0,7/0,6; mushroom 0,2/0,1;
retail 0,01/0,002; T10I4D100K 0,01/0,002; kosarak 0,01/0,002; accidents
0,6/0,4 (`scripts/validate_gpu.sh`). Com arenas reduzidas (`--arena-kb 512` e
`64`) para forçar o caminho de estouro, a saída continua idêntica (kosarak
0,002: 470 tarefas estouradas; T10I4D100K 0,002: 697; retail 0,002: 475).
`tests`: 535 verificações, 0 falhas.

Primeiras medidas (ms, uma execução, mineração = passada 1 + passada 2):

| Dataset | minsup | seq | GPU estática | passada 1 | máx/média por bloco |
| --- | --- | ---: | ---: | ---: | ---: |
| chess | 0,6 | 57 | 246 | 119 | 6,0 |
| chess | 0,5 | 228 | 876 | 409 | 5,4 |
| mushroom | 0,05 | 248 | 237 | 109 | 10,7 |
| accidents | 0,3 | 144 | 183 | 87 | 3,4 |
| T10I4D100K | 0,001 | 68 | 4,0 | 1,9 | 7,4 |
| retail | 0,001 | 32 | 1,2 | 0,6 | 6,0 |
| kosarak | 0,001 | 1.119 | 1.008 | 486 | 131,9 |

Leitura: nos esparsos com muitos itens (T10I4D100K, retail) a estática já é
17–27× mais rápida que o sequencial. Nos densos ela perde: chess tem só 37
itens frequentes, ou seja, 37 blocos numa GPU com 128 SMs, e cada bloco
processa milhares de árvores condicionais em série. Em kosarak um único bloco
(~550 ms) define o tempo do kernel: máx/média = 132. A saída em duas passadas
dobra o custo da mineração por construção. Histograma em
`results/blocks_static_kosarak.png`.

Em aberto: a estática é limitada pela posse fixa item → bloco (esperado; é o
que a Fase 5 ataca).

## Fase 5 — concluída

Feito (no mesmo `src/gpu/mine.cu`, kernel `mine_dynamic_kernel`): fila global
circular de descritores com `head`/`tail` atômicos e capacidade configurável
(`--queue`, 5.1); blocos persistentes em número igual ao máximo residente
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SMs = 2 × 128, 5.2); árvores
condicionais viram tarefas (5.3); limiar `--grain` sobre o número de itens da
base condicional filtrada (5.4); contador global de pendentes com
`__threadfence` antes de publicar (5.5); fila cheia ou pool esgotado →
processamento local em profundidade (5.6); duas passadas, estouro e
instrumentação reaproveitados (5.7); validação (5.8).

Decisões:
- **Fila MPMC sem travas**: além de `head`/`tail`, cada posição tem um número
  de sequência (posição + 1 quando publicada, 0 quando livre). O produtor
  reserva a posição com `atomicCAS` em `tail`, espera a posição ser liberada
  pelo consumidor anterior, escreve, faz `__threadfence` e publica; o consumidor
  reserva com `atomicCAS` em `head` e espera a publicação. As esperas são
  limitadas porque todos os blocos estão residentes (kernel persistente).
- **Terminação**: `pending` é incrementado antes de publicar uma tarefa e
  decrementado quando a tarefa e toda a sua subárvore local terminam; um bloco
  só sai com a fila vazia e `pending == 0`.
- **Pool global** por bump atômico (até 15 % da memória livre, máx. 3 GB) para
  as árvores enfileiradas; nunca é liberado durante um kernel. Se acabar, os
  filhos são processados localmente.
- As tarefas iniciais (uma por item raiz) são escritas na fila pelo host.
- Como o destino de cada filho (fila ou pilha local) depende do estado da fila,
  os estouros de arena podem variar entre as passadas; a passada 2 tem 12,5 % de
  folga no buffer e repete com o dobro se encher.
- **grain padrão = 64** (o plano sugeria 256): foi o melhor ou próximo do melhor
  em todos os datasets (tabela abaixo).

Validação: saída idêntica ao sequencial em 7 configurações × grain
0/256/4096 (21 casos); com fila de 4 posições (o host sobe para 2 × nº de
raízes), pool de 1 MB e arena de 512 KB (estouros); `tests` inclui grain
0/64/∞. **Estresse: 420 execuções (7 configurações × 3 grains × 20), nenhum
deadlock nem diferença** (`results/stress_dynamic.txt`).

Mineração (ms, uma execução) e razão máx/média do tempo ocupado por bloco:

| Dataset | minsup | estática | din. g=0 | din. g=64 | din. g=256 | din. g=1024 | din. g=4096 | máx/média est. | máx/média din. g=64 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 884,5 | 51,1 | **27,8** | 75,7 | 589,3 | 902,6 | 5,4 | 1,09 |
| mushroom | 0,05 | 236,9 | **8,1** | 19,8 | 69,9 | 99,1 | 184,5 | 10,7 | 2,90 |
| accidents | 0,3 | 182,6 | 81,8 | **80,8** | 82,1 | 84,4 | 91,4 | 3,4 | 9,94 |
| T10I4D100K | 0,001 | 4,0 | 2,9 | **2,6** | 4,8 | 4,7 | 4,1 | 7,6 | 1,43 |
| retail | 0,001 | **1,2** | 1,4 | 1,5 | 1,4 | 1,3 | 1,2 | 6,2 | 1,64 |
| kosarak | 0,001 | 990,7 | 48,0 | **40,4** | 43,0 | 373,0 | 1.007,5 | 131,6 | 1,01 |

Leitura: a dinâmica resolve o desbalanceamento (kosarak: máx/média 132 → 1,01,
mineração 25× mais rápida; chess 32×). Grains grandes reduzem o número de
tarefas na fila e devolvem o comportamento da estática. Em accidents a razão
continua ~10 porque as primeiras árvores condicionais têm bases enormes (340 k
transações): a construção de uma árvore é indivisível entre blocos, e um bloco
sozinho leva ~80 ms nela. Em retail não há o que ganhar: a mineração já leva 1 ms.

Em aberto: dividir a construção das árvores condicionais muito grandes entre
vários blocos (ou mandá-las para a rotina de grade inteira da Fase 3) atacaria
o caso accidents.

## Fase 6 — concluída

Feito: `scripts/bench.py` (main/grain/blocks; 6.1, 6.2, 6.4, 6.5), `scripts/plots.py`
(5 gráficos + `results/tables.md`, 6.8), `results/RESULTS.md` (6.9) e
`scripts/run_all.sh`. 360 medidas (6 datasets × 4 suportes × 3 implementações
× 5) mais 180 do experimento de grain, **todas válidas** (hash da saída igual
ao do sequencial).

Decisões:
- Suportes finais: chess 0,5–0,35; mushroom 0,05–0,01; accidents 0,3–0,1;
  T10I4D100K e retail 0,0002–0,00003; kosarak 0,002–0,00085. O sequencial vai de
  0,17 s a 80 s. Abaixo disso (ex.: kosarak 0,0006) a saída não cabe nos 20 GB
  de RAM por processo que adotamos na máquina compartilhada.
- **Correção (registrada depois):** uma versão anterior desta nota dizia que a
  escrita canônica passou a usar um buffer único + offsets (`write_canonical`).
  Essa mudança foi desenhada mas **nunca aplicada**: o comando que a continha
  foi bloqueado por uma checagem de segurança e só o build foi reexecutado.
  Todos os benchmarks usaram a escrita original (`to_canonical_lines` +
  `write_lines`), que coube no limite de 20 GB (pico de 13,8 GB em mushroom
  0,01, 90,7 M itemsets). O buffer único fica como melhoria futura.
- Saídas temporárias em `results/tmp/` (disco), não em `/tmp`, que é tmpfs e
  consumiria RAM compartilhada.
- O benchmark escolhe uma GPU com < 10 % de uso e < 2 GB ocupados; se as duas
  estiverem ocupadas por outros, espera 60 s e tenta de novo. Isso aconteceu
  durante a execução e o script esperou, sem interferir.
- A etapa `transferencia_d2h` inclui a montagem do CSR de itemsets no host
  (registrado nas tabelas).

Resultados principais (detalhes em `results/RESULTS.md`):
- Construção: 11–37× sobre o sequencial nas árvores grandes; 34–170× sem a
  ordenação lexicográfica, que ocupa 43–86 % do tempo. Perde (0,4–0,9×) nas
  árvores de ~20 k nós (chess, mushroom).
- Mineração: dinâmica 1,8–43× sobre o sequencial e 1,1–32× sobre a estática;
  a estática perde do sequencial em 13 das 24 configurações.
- Balanceamento: máx/média 3,4–1.644 na estática contra 1,0–2,3 na dinâmica em
  19/24 configurações.
- grain ótimo entre 16 e 64; ≥ 1024 devolve o comportamento da estática.
- Ponta a ponta: núcleo 3,4–39× mais rápido; o total é dominado pela escrita
  canônica (CPU) quando há dezenas de milhões de itemsets.

Limitações conhecidas: escrita canônica sequencial; reserva fixa e grande de
arenas (pico de 6–13,5 GB na GPU durante a mineração); construção de árvores
condicionais gigantes num único bloco (accidents); custo fixo do Thrust nas
árvores pequenas.

## Checklist de entrega

- [x] `fpg_seq` e `fpg_gpu` compilam a partir de um clone limpo com os comandos do README, sem warnings.
- [x] `tests` passa (535 verificações), incluindo os casos de borda da Fase 3.
      No teste do clone limpo apareceu um bug: o `.gitignore` (`data/`) também
      ignorava `tests/data/`. Corrigido para `/data/`.
- [x] `scripts/run_all.sh` reproduz todos os resultados.
- [x] `results/RESULTS.md` contém gráficos e interpretação.
- [x] `NOTES.md` registra o hardware, as decisões e as limitações conhecidas.

## Depois da Fase 6 — suportes padrão, correção da mineração e organização

Em ordem cronológica:

1. **Artigo IEEE** (`paper/`), escrito sobre o conjunto de suportes reduzidos.
   O Nsight Compute não pôde ser usado (contadores exigem administrador) e o
   Nsight Systems da máquina está sem o importador; os limites de hardware
   vieram de um microbenchmark próprio (`tests/microbench.cu`).
2. **GPU fixa.** As duas RTX 4090 estão em links PCIe x16 e x4, e a escolha
   automática da GPU livre fazia as transferências variarem. Todas as medições
   passaram a usar a placa `00000000:01:00.0` (x16), via `FPG_GPU_BUS`. A
   espera por GPU livre passou a olhar processos de outros usuários, e não a
   utilização média (que mostrava a nossa própria execução anterior).
3. **Suportes padrão como conjunto principal.** Os suportes mais usados na
   literatura FIMI (a tabela do plano) estão em `results/`; os reduzidos ficam
   como experimento complementar (`FPG_SET=low`, `results/low_support/`).
   Com os suportes padrão a maior parte das configurações é pequena, e custos
   fixos da GPU passam a importar.
4. **Correção: preparação da mineração fora do cronômetro.** As arenas (até
   6 GB) e o pool (até 3 GB) eram reservados com `device_vector::resize`, que
   zera a memória, num trecho que nenhuma etapa media: 18–34 ms por execução
   da dinâmica que não apareciam no CSV. Agora a memória é reservada com
   `cudaMalloc` sem inicialização (é sempre escrita antes de ser lida), e a
   preparação aparece como `mineracao_preparacao` (7–14 ms, só alocação).
   Medições anteriores ficaram em `results/pre_fix/`.
5. **Relatório por fase gerado por script** (`scripts/relatorio_fases.py`),
   com antes/depois da correção: `reports/relatorio_fases_suporte_padrao.md`.
6. **Baselines em GPU:** Apriori (estilo GPApriori) validado; Eclat (estilo
   Frontier Expansion) com bug conhecido. Os testes do Eclat só rodam com
   `./build/tests --with-eclat`.
7. **Organização para publicação:** `results/RESULTS.md` foi substituído por
   `results/tables.md` e pelos relatórios gerados; `run_all.sh` ganhou etapas
   (`STEPS`) e conjuntos (`FPG_SET`); `TODO.md`, `TUTORIAL.md`,
   `requirements.txt` e `docs/plano_original.md` foram criados. A rodada do
   conjunto reduzido com a correção foi interrompida e seus resultados
   apagados; refazer é o TODO 1.1.

Pendências: [TODO.md](TODO.md).
