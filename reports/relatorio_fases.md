# FP-Growth em GPU: relatório detalhado por fase

Este relatório acompanha cada fase do FP-Growth do começo ao fim. Para cada uma,
mostra:

- como ela é implementada originalmente (versão sequencial em CPU) e quanto tempo leva;
- as ideias que testamos para acelerá-la, na ordem em que foram tentadas;
- o desempenho de cada ideia, os detalhes de implementação e por que cada mecanismo foi escolhido.

No fim, compara a versão final com a original, com o speedup de cada fase e o
speedup total.

Todos os números vêm dos CSVs em `results/`, medidos numa única RTX 4090: a do
slot PCIe x16, barramento `0000:01:00.0`. Valores em milissegundos são
medianas de 5 execuções, depois de 1 de aquecimento descartada.

---

## Sumário

1. [Resumo dos resultados](#1-resumo-dos-resultados)
2. [Ambiente, dados e método](#2-ambiente-dados-e-método)
3. [O algoritmo e onde o tempo está na versão original](#3-o-algoritmo-e-onde-o-tempo-está-na-versão-original)
4. [Fase 1 — Leitura](#4-fase-1--leitura-cpu)
5. [Fase 2 — Contagem de suporte](#5-fase-2--contagem-de-suporte)
6. [Fase 3 — Pré-processamento](#6-fase-3--pré-processamento)
7. [Transferência CPU → GPU](#7-transferência-cpu--gpu-h2d)
8. [Fase 4 — Construção da FP-tree](#8-fase-4--construção-da-fp-tree)
9. [Fase 5 — Mineração](#9-fase-5--mineração)
10. [Fase 6 — Retorno e escrita da saída](#10-fase-6--retorno-d2h-e-escrita-da-saída)
11. [Comparação final: tudo junto](#11-comparação-final-tudo-junto)
12. [O que falta e o ganho que ainda é possível](#12-o-que-falta-e-o-ganho-que-ainda-é-possível)
13. [Reprodução](#13-reprodução)

---

## 1. Resumo dos resultados

Na tabela, **speedup = tempo do sequencial / tempo da versão final em GPU**
(mineração dinâmica, `grain` 64). Faixa e média geométrica sobre as 24
configurações (6 datasets × 4 suportes).

| Fase | Ideia final | Speedup mín. | Speedup mediano | Speedup máx. | Média geométrica |
| --- | --- | ---: | ---: | ---: | ---: |
| Contagem de suporte | histograma por bloco em memória compartilhada | 0,06× | 0,6× | 64× | 1,1× |
| Pré-processamento | scans + ordenação segmentada | 1,2× | 7,7× | 46× | 6,7× |
| Construção da FP-tree | ordenação + LCP + scan + sparse table | 0,3× | 15× | 45× | 6,6× |
| Mineração | fila global dinâmica + blocos persistentes | 1,5× | 12× | 41× | 11,6× |
| **Núcleo** (as 4 fases acima) | — | **3,1×** | **12×** | **37×** | **11,8×** |
| Núcleo + transferências | — | 0,9× | 3,6× | 20× | 4,3× |
| **Ponta a ponta** (com leitura e escrita) | — | **0,96×** | **1,4×** | **3,5×** | **1,5×** |

Somando as 24 configurações: o núcleo do algoritmo cai de **38,6 s para
3,85 s (10,0×)**, mas o tempo total cai só de **184,4 s para 171,1 s (1,08×)**.
A diferença está nas fases que continuam na CPU, leitura e, principalmente,
escrita da saída, que nenhuma das ideias deste trabalho tocou.

Leituras principais:

- **Construção:** grandes ganhos quando a árvore é grande, 10–45× com 0,5–3 M nós.
  Nos datasets com árvores de ~20 mil nós (chess, mushroom) a GPU perde, por
  custo fixo de ~2–3 ms.
- **Mineração:** a ideia que mais pesou foi a fila dinâmica. A versão estática
  (um bloco por item) é mais lenta que a CPU em 12 das 24 configurações; a
  dinâmica é mais rápida que a CPU em todas e chega a 33× mais rápida que a estática.
- **Contagem:** o kernel em si é 3,8–124× mais rápido que a CPU, mas dentro do
  pipeline essa fase é dominada por custo fixo (alocação e primeiro lançamento)
  e fica, em média, empatada com a CPU. Ela vale menos de 1 % do tempo.
- **Ponta a ponta:** limitado pela lei de Amdahl. Em chess e mushroom, mesmo um
  núcleo de tempo zero daria no máximo 1,1–1,4×, porque 72–92 % do tempo
  sequencial é ler e escrever arquivos.

---

## 2. Ambiente, dados e método

**Hardware.** NVIDIA RTX 4090: 128 SMs, 24 GB e banda de pico de 1008 GB/s;
medimos 920 GB/s numa cópia D2D. Ela fica num slot PCIe Gen4 x16 (26,6 GB/s com
memória *pinned*, 24,2 GB/s *pageable*). CPU AMD Ryzen 9 7900 (12 núcleos),
60 GB de RAM, CUDA 12.4, g++ 13, `-O3`.

A máquina tem uma segunda RTX 4090 num link x4 (6,4 GB/s). Ela não foi usada,
porque deixaria as transferências 4× mais lentas. Cada execução registra o
barramento PCI, e `scripts/paper_data.py` falha se alguma tiver rodado em outra placa.

**Datasets.** Seis bases do repositório FIMI, obtidas de um espelho da Wayback
Machine porque o site original está fora do ar.

| Dataset | Tipo | Transações | Itens distintos | Tam. médio | Tam. máx. | Suportes avaliados |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| chess | denso, pequeno | 3.196 | 75 | 37,0 | 37 | 0,5 · 0,45 · 0,4 · 0,35 |
| mushroom | denso, pequeno | 8.124 | 119 | 23,0 | 23 | 0,05 · 0,03 · 0,02 · 0,01 |
| accidents | denso, grande | 340.183 | 468 | 33,8 | 51 | 0,3 · 0,2 · 0,15 · 0,1 |
| T10I4D100K | esparso, sintético | 100.000 | 870 | 10,1 | 29 | 0,0002 · 0,0001 · 0,00005 · 0,00003 |
| retail | esparso, real | 88.162 | 16.470 | 10,3 | 76 | 0,0002 · 0,0001 · 0,00005 · 0,00003 |
| kosarak | esparso, grande | 990.002 | 41.270 | 8,1 | 2.497 | 0,002 · 0,001 · 0,0009 · 0,00085 |

Os suportes foram escolhidos para o sequencial levar entre 0,17 s e 80 s.
Suportes menores em kosarak (ex.: 0,0006) geram mais saída do que cabe no
limite de 20 GB de RAM por processo que adotamos, já que a máquina é compartilhada.

**O que cada configuração produz** (suporte intermediário → menor suporte):

| Dataset | Itens frequentes | Nós da FP-tree | Itemsets frequentes |
| --- | ---: | ---: | ---: |
| chess | 39 → 45 | 18,6 k → 26,0 k | 2,8 M → 15,1 M |
| mushroom | 82 → 96 | 25,3 k → 26,6 k | 10,0 M → 90,8 M |
| accidents | 48 → 75 | 1,59 M → 2,87 M | 0,89 M → 10,7 M |
| T10I4D100K | 867 → 869 | 715 k → 715 k | 0,41 M → 6,2 M |
| retail | 8.919 → 12.889 | 653 k → 673 k | 0,24 M → 20,6 M |
| kosarak | 1.259 → 1.461 | 2,90 M → 3,08 M | 0,76 M → 6,5 M |

A tabela deixa claro o contraste que guia todo o trabalho:

- **densos:** árvore minúscula, mas uma explosão de itemsets;
- **esparsos:** árvore enorme, com milhares de itens de carga muito desigual.

**Corretude.** A saída é canônica (itens em ordem crescente, linhas ordenadas),
então comparar duas saídas é um `cmp`.

- O sequencial é idêntico ao `fpgrowth` 6.21 de Christian Borgelt em 12 casos e
  às saídas calculadas à mão em 5 datasets minúsculos.
- Toda execução da GPU no benchmark teve o hash da saída comparado com o do
  sequencial: 360 execuções principais mais 180 de `grain`, **nenhuma inválida**.
- A suíte `tests` tem 535 verificações.
- A versão dinâmica rodou 420 vezes seguidas sem travar.

---

## 3. O algoritmo e onde o tempo está na versão original

O FP-Growth tem quatro passos algorítmicos, que aqui ficam entre uma leitura e
uma escrita:

```
 leitura ─► contagem ─► pré-processamento ─► construção da FP-tree ─► mineração ─► escrita
  (CPU)       (1)              (2)                    (3)                 (4)        (CPU)
              └──────────────── núcleo do algoritmo (vai para a GPU) ────────┘
```

1. **Contagem:** suporte de cada item.
2. **Pré-processamento:** descarta itens infrequentes, renumera os frequentes
   por suporte decrescente (o *rank*) e ordena cada transação.
3. **Construção:** insere as transações numa árvore de prefixos com contadores
   (a FP-tree) e numa tabela de cabeçalho que liga todos os nós de cada item.
4. **Mineração:** para cada item `y`, do menos para o mais frequente:
   - emite `S ∪ {y}`;
   - extrai a *base condicional* (os caminhos da raiz até cada nó de `y`, com o peso do nó);
   - constrói a árvore condicional e repete a mineração nela.

   Se uma árvore é um caminho único, todas as combinações dos seus nós são
   emitidas diretamente.

As transações ficam em formato **CSR**: um vetor `items[]` com os ids de todas
as transações em sequência e um vetor `offsets[]` de tamanho n+1. É compacto,
contíguo e é o que a GPU lê bem.

**Tempo da versão original por fase** (sequencial, suporte intermediário, ms e
% do total):

| Dataset (minsup) | Leitura | Contagem | Pré-proc. | Construção | Mineração | Escrita | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess (0,45) | 1,0 (0 %) | 0,05 (0 %) | 0,8 (0 %) | 1,5 (0 %) | 478 (24 %) | 1.521 (76 %) | 2.003 |
| mushroom (0,03) | 1,2 (0 %) | 0,07 (0 %) | 1,0 (0 %) | 2,5 (0 %) | 770 (12 %) | 5.626 (88 %) | 6.396 |
| accidents (0,2) | 80 (5 %) | 4,5 (0 %) | 91 (6 %) | 229 (15 %) | 682 (44 %) | 462 (30 %) | 1.549 |
| T10I4D100K (0,0001) | 8,5 (2 %) | 0,4 (0 %) | 8,4 (2 %) | 53 (10 %) | 324 (63 %) | 124 (24 %) | 518 |
| retail (0,0001) | 9,8 (3 %) | 0,4 (0 %) | 8,1 (3 %) | 48 (16 %) | 157 (53 %) | 72 (24 %) | 296 |
| kosarak (0,001) | 154 (7 %) | 4,3 (0 %) | 65 (3 %) | 304 (15 %) | 1.113 (53 %) | 452 (22 %) | 2.088 |

Essa distribuição define onde vale a pena investir:

- **Mineração** é a maior fase algorítmica em todos os datasets (12–63 % do total).
- **Construção** pesa 10–29 % nos esparsos em suporte alto e é desprezível nos densos.
- **Contagem e pré-processamento** juntos ficam abaixo de 7 %.
- **Escrita**, que é igual em todas as versões, domina os densos (76–92 %) e
  qualquer configuração com dezenas de milhões de itemsets.

(Tabela com todas as 24 configurações no
[apêndice A](#apêndice-a--tempo-sequencial-por-fase-todas-as-configurações).)

---

## 4. Fase 1 — Leitura (CPU)

**Implementação original.** O arquivo inteiro é lido de uma vez (`fread`) e
convertido em CSR por um único laço que acumula dígitos e fecha uma transação a
cada `\n`. Itens repetidos dentro de uma transação são removidos (`sort` +
`unique` por transação). Código em `src/common/fimi.cpp`.

**Tempo:** 0,9–1,3 ms nos datasets pequenos, 8–10 ms em T10I4D100K e retail,
81 ms em accidents e 153 ms em kosarak (66 MB de texto, ~0,43 GB/s).

**Ideias de melhoria:** nenhuma implementada. A leitura é idêntica nas três
versões (sequencial, GPU estática e GPU dinâmica), para que a comparação isole
o algoritmo. Ela só pesa nos esparsos com suporte alto: 19 % do tempo
sequencial em kosarak 0,002. Duas opções ficam como trabalho futuro:

- parsing paralelo em blocos de linhas com os 12 núcleos, com ganho ideal de
  ~12×, limitado pela leitura do disco;
- enviar o texto bruto para a GPU e fazer o parsing lá, com um scan sobre os
  separadores.

---

## 5. Fase 2 — Contagem de suporte

### 5.1 Implementação original

Um vetor de contadores do tamanho do maior id e um laço sobre todos os itens:
`++sup[x]`. É o melhor caso possível para a CPU: acesso sequencial a
`items[]`, e o vetor de contadores cabe no cache (no máximo 41 mil inteiros,
165 KB, em kosarak).

**Tempo original:** 0,05 ms (chess) a 4,5 ms (accidents, 11,5 M ocorrências),
cerca de 2,6 ns por ocorrência. É sempre menos de 1 % do total.

### 5.2 Ideia 1 — `atomicAdd` global (kernel v1)

**O que é.** Uma thread por ocorrência executa `atomicAdd(&support[items[p]], 1)`
na memória global.

**Por que começar por ela.** É a tradução direta do laço sequencial e serve de
referência. Na GPU, atômicos em memória global são resolvidos no cache L2:
cada endereço processa uma atualização por vez.

**Desempenho** (kernel isolado, média de 50 lançamentos, incluindo alocar e zerar o vetor):

| Dataset | Ocorrências | Sequencial (ms) | v1 (ms) | v1 vs. seq |
| --- | ---: | ---: | ---: | ---: |
| chess | 118 k | 0,046 | 0,026 | 1,8× |
| mushroom | 187 k | 0,074 | 0,044 | 1,7× |
| accidents | 11,5 M | 4,47 | 1,89 | 2,4× |
| T10I4D100K | 1,01 M | 0,417 | 0,169 | 2,5× |
| retail | 909 k | 0,386 | 0,087 | 4,4× |
| kosarak | 8,0 M | 4,30 | 1,08 | 4,0× |

O ganho é pequeno: 1,7–4,4×. O problema é a **contenção**. Em accidents cada
transação tem 34 itens escolhidos entre só 468 ids, e alguns ids aparecem em
quase todas as transações. Milhões de atômicos caem nos mesmos poucos endereços
e se serializam no L2. Por isso accidents, com 11,5 M ocorrências, é mais lento
que kosarak, com 8 M.

### 5.3 Ideia 2 — Histograma privado por bloco em memória compartilhada (kernel v2)

**O que é.**

1. Cada bloco de 256 threads mantém seu próprio histograma de 12.288 contadores
   (48 KB) na memória compartilhada.
2. As threads do bloco percorrem as ocorrências fazendo `atomicAdd` no
   histograma *compartilhado*.
3. No fim, cada bloco soma seu histograma no vetor global, com um `atomicAdd`
   por contador não nulo.

**Detalhes.**

- **Faixas de ids:** se há mais ids que contadores (retail tem 16 k ids,
  kosarak 41 k), uma segunda dimensão da grade (`gridDim.y`) divide os ids em
  faixas. Cada bloco conta só a sua faixa e relê a base inteira: retail usa 2
  faixas, kosarak 4.
- **Tamanho da grade:** no máximo 4 blocos por SM, para cada bloco varrer muitos
  itens e amortizar o custo de zerar e descarregar seu histograma.

**Por que esse mecanismo.**

- **Memória compartilhada:** os atômicos ali são resolvidos dentro do SM, com
  latência de poucos ciclos.
- **Menos disputa:** cada endereço passa a ser disputado só pelas 256 threads
  de um bloco, e não pela GPU inteira.
- **Menos tráfego global:** o vetor global recebe um atômico por contador e
  por bloco, e não um por ocorrência.
- **Faixas:** reler a base algumas vezes é barato, porque a base cabe nos 72 MB
  de L2. É bem mais simples que um histograma hierárquico.

**Desempenho:**

| Dataset | Faixas | v1 (ms) | v2 (ms) | v2 vs. v1 | v2 vs. seq | Mínimo teórico* (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 1 | 0,026 | 0,012 | 2,1× | 3,8× | 0,5 |
| mushroom | 1 | 0,044 | 0,012 | 3,6× | 6,0× | 0,7 |
| accidents | 1 | 1,893 | 0,036 | **52,6×** | **124×** | 46 |
| T10I4D100K | 1 | 0,169 | 0,016 | 10,7× | 26× | 4 |
| retail | 2 | 0,087 | 0,022 | 3,9× | 17× | 4 |
| kosarak | 4 | 1,075 | 0,173 | 6,2× | 25× | 32 |

\* ler todas as ocorrências uma vez (4 bytes cada) a 1008 GB/s.

O maior ganho está onde a contenção era o problema: accidents, com 53× sobre a
v1. Nos datasets maiores a v2 fica a 0,8–6,2× do mínimo teórico. Em accidents
ela é até mais rápida que o "mínimo" porque a base está no L2, que é mais
rápido que a DRAM. Em kosarak a distância vem das 4 faixas, ou seja, 4 leituras
da base. Nos pequenos, ~12 µs é o piso de lançamento mais alocação e não há o
que ganhar.

### 5.4 Resultado da fase dentro do pipeline

Dentro do programa completo, a fase de contagem na GPU mede 0,02–0,96 ms,
**muito mais que o kernel** (0,012–0,17 ms), e o tempo varia de uma
configuração para outra sem relação com o tamanho da base. Por isso o speedup
da fase no pipeline vai de 0,06× a 64× (média geométrica 1,1×).

O custo extra é fixo: alocar o vetor de suporte e, provavelmente, carregar o
módulo CUDA no primeiro lançamento de cada kernel. No CUDA 12 o carregamento é
preguiçoso por padrão. **Essa causa não foi medida isoladamente.** O teste
direto seria rodar com `CUDA_MODULE_LOADING=EAGER` e um alocador com cache.

Como a fase vale menos de 1 % do tempo, isso não muda o resultado total.

---

## 6. Fase 3 — Pré-processamento

### 6.1 Implementação original

Quatro passos:

1. **Tabela de ranks:** os ids com suporte ≥ minsup são ordenados por (suporte
   decrescente, id crescente). O desempate pelo id torna a ordem determinística,
   e a GPU usa exatamente a mesma regra. Isso gera `id_to_rank[]` e `rank_to_id[]`.
2. **Filtragem:** para cada transação, os itens infrequentes são descartados e
   cada id é trocado pelo seu rank.
3. **Ordenação:** cada transação é ordenada por rank crescente com `std::sort`.
4. **Descarte:** transações que ficaram vazias são removidas.

**Tempo original:**

| Dataset | minsup | Sequencial (ms) |
| --- | --- | ---: |
| chess / mushroom | todos | 0,7–1,0 |
| T10I4D100K / retail | todos | 8,0–8,5 |
| kosarak | 0,001 / 0,00085 | 65 / 67 |
| accidents | 0,2 / 0,1 | 91 / 105 |

O custo está na ordenação por transação: 34 itens em média em accidents e
transações de até 2.497 itens em kosarak.

### 6.2 Ideia — tudo como primitivas paralelas (uma única ideia, implementada)

**O que é.** A fase vira uma sequência de primitivas sem nenhum controle de
fluxo dependente dos dados:

| Passo | Primitiva | Por que essa primitiva |
| --- | --- | --- |
| Ranks | `copy_if` dos ids frequentes + `thrust::sort` com comparador (suporte desc., id asc.) + `scatter` | ordenar ≤ 41 k ids é trivial; o comparador reproduz o desempate da CPU |
| Filtragem e remapeamento | kernel com 1 thread por ocorrência: `rank = id_to_rank[id]`, `flag = rank ≥ 0` | totalmente independente por elemento |
| Posição de cada item mantido | `exclusive_scan(flag)` | o scan é o jeito padrão de compactar em paralelo |
| Novo tamanho de cada transação | `cub::DeviceSegmentedReduce::Sum(flag)` | soma por segmento sem atômicos |
| Escrita compactada | kernel de *scatter* nas posições do scan | |
| Descarte de transações vazias | `copy_if` dos tamanhos > 0 + `inclusive_scan` → novos offsets | |
| Ordenação de cada transação | `cub::DeviceSegmentedSort::SortKeys` | ordena milhares de segmentos curtos de uma vez, escolhendo o algoritmo pelo tamanho do segmento |

**Por que não um kernel "uma thread por transação" que filtra e ordena.**
Transações de tamanho muito diferente (1 a 2.497 em kosarak) deixariam
threads ociosas no mesmo warp, e a ordenação dentro da thread seria serial. As
primitivas dividem o trabalho por *ocorrência*, não por transação, e as
bibliotecas (CUB) já tratam segmentos de tamanhos variados.

**Desempenho:**

| Dataset | minsup | Sequencial (ms) | GPU (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| chess | 0,45 / 0,35 | 0,77 / 0,72 | 0,60 / 0,59 | 1,3× / 1,2× |
| mushroom | 0,03 / 0,01 | 0,96 / 0,96 | 0,76 / 0,77 | 1,3× / 1,3× |
| accidents | 0,2 / 0,1 | 90,8 / 105 | 2,21 / 2,28 | **41× / 46×** |
| T10I4D100K | 0,0001 / 0,00003 | 8,4 / 8,1 | 0,98 / 0,99 | 8,6× / 8,1× |
| retail | 0,0001 / 0,00003 | 8,1 / 8,4 | 1,06 / 1,09 | 7,7× / 7,7× |
| kosarak | 0,001 / 0,00085 | 65,3 / 66,5 | 2,76 / 2,92 | **24× / 23×** |

**Corretude:** a tabela de ranks e o CSR pré-processado são idênticos aos do
sequencial em todos os datasets (modo `--debug` e suíte `tests`).

**Distância do limite.** O tráfego obrigatório é de cerca de 6 passadas de 4
bytes por ocorrência: ~0,2 ms em kosarak a 920 GB/s. A GPU leva 2,8 ms, cerca
de 14× o limite. A diferença está nas ~10 chamadas de biblioteca, cada uma com
alocação de temporários e lançamento. Fundir filtragem, flag e scatter num
kernel só e usar um alocador com cache aproximaria a fase do limite. Mas, de
novo, a fase pesa pouco no total.

---

## 7. Transferência CPU → GPU (H2D)

Esta fase não existe no sequencial: é o preço de usar a GPU.

**Implementação.** Dois `cudaMemcpy` (items e offsets) a partir de memória
*pageable* (`std::vector`).

| Dataset | Bytes | Tempo (ms) | Banda efetiva |
| --- | ---: | ---: | ---: |
| chess | 0,5 MB | 0,83 | 0,6 GB/s |
| mushroom | 0,8 MB | 0,94 | 0,8 GB/s |
| T10I4D100K | 4,4 MB | 1,24 | 3,6 GB/s |
| retail | 4,0 MB | 1,12 | 3,6 GB/s |
| kosarak | 36 MB | 3,05 | 11,8 GB/s |
| accidents | 47 MB | 3,23 | 14,7 GB/s |

Nos datasets pequenos o tempo é dominado pela alocação dos vetores na GPU (a
medição inclui o `resize`), não pela cópia. Nos grandes a banda chega a
12–15 GB/s, abaixo dos 24 GB/s que o link alcança. Na placa x4 kosarak levaria
cerca de 6 ms.

**Melhoria possível, não implementada:** memória *pinned* e cópia em pedaços
sobrepostos à contagem, que esconderia quase todo o custo.

---

## 8. Fase 4 — Construção da FP-tree

Esta é a contribuição central do trabalho.

### 8.1 Implementação original

A FP-tree clássica com nós em vetores e índices no lugar de ponteiros, em
`src/seq/fpgrowth_seq.cpp`. A partir da raiz, cada transação desce pela árvore
item a item:

- se o filho com aquele item existe, incrementa o contador;
- senão, cria o nó.

Cada nó criado é ligado à lista de irmãos do pai e à lista ligada do seu item
(a tabela de cabeçalho).

**Busca de filho.** Uma lista de irmãos faria a raiz de kosarak ter mais de mil
filhos para varrer a cada transação. Por isso a busca usa uma tabela hash
aberta com chave (pai, item).

**Tempo original:**

| Dataset | Nós | Sequencial (ms) |
| --- | ---: | ---: |
| chess | 16–26 k | 1,1–1,6 |
| mushroom | 22–27 k | 2,2–2,5 |
| T10I4D100K | 715 k | 51–53 |
| retail | 626–673 k | 47–50 |
| kosarak | 2,0–3,1 M | 179–320 |
| accidents | 0,53–2,87 M | 100–402 |

São cerca de 70–140 ns por nó: o custo vem de acessos aleatórios à memória (hash
e contadores) que não cabem no cache.

**Por que a construção é difícil em GPU.** A forma óbvia de paralelizar é
inserir várias transações ao mesmo tempo. Isso exige criar filhos de forma
atômica e incrementar contadores com atômicos, e os nós perto da raiz recebem
uma atualização por transação. É o mesmo padrão de contenção que derrubou a
contagem v1. Além disso, o formato da árvore passaria a depender da ordem de
execução das threads. **Essa ideia foi descartada antes de ser implementada.**

### 8.2 Ideia 1 — Nós como prefixos distintos: ordenação lexicográfica + LCP + scan

**Observação central.** Os nós da FP-tree são exatamente os prefixos distintos
das transações já ordenadas internamente. Depois de ordenar as transações
*lexicograficamente* (um prefixo vem antes das suas extensões), transações com
o mesmo prefixo ficam vizinhas. Assim, a transação `Tᵢ` só cria nós a partir da
profundidade `LCPᵢ`, o maior prefixo comum com a transação anterior.

**Passos.**

1. **Ordenação lexicográfica** dos índices das transações
   (`construcao_sort`): `thrust::sort` com um comparador que percorre os dois
   segmentos do CSR item a item. Depois o CSR é reescrito na nova ordem.
2. **LCP** (`construcao_lcp`): uma thread por transação compara `Tᵢ` com `Tᵢ₋₁`.
3. **Ids dos nós** (`construcao_scan`): `novosᵢ = lenᵢ − LCPᵢ` e
   `base = exclusive_scan(novos)`. O nó da transação `i` na profundidade `d`
   recebe o id `baseᵢ + (d − LCPᵢ)`. O total de nós é a soma.
4. **Item, profundidade e transação criadora** de cada nó (`construcao_nos`):
   uma thread por transação escreve seus nós novos.

O resultado é que os ids saem em **pré-ordem** da árvore, sem ponteiros: a
árvore vira vetores `item[]`, `count[]`, `parent[]` e `depth[]`.

Exemplo (Han, Pei e Yin, 2000; ranks c < f < a < b < m < p):

| i | Tᵢ ordenada | LCPᵢ | nós novos | baseᵢ | ids criados |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | c f a b m | 0 | 5 | 0 | 0(c) 1(f) 2(a) 3(b) 4(m) |
| 1 | c f a m p | 3 | 2 | 5 | 5(m) 6(p) |
| 2 | c f a m p | 5 | 0 | 7 | — (duplicada) |
| 3 | c b p | 1 | 2 | 7 | 7(b) 8(p) |
| 4 | f b | 0 | 2 | 9 | 9(f) 10(b) |

**Por que ordenação com comparador e não radix.** As transações têm tamanho
variável (até 2.497 itens). Um radix sort LSD por coluna precisaria de tantas
passadas quanto a maior transação. O comparador funciona para qualquer tamanho,
e o merge sort do Thrust é estável e bem otimizado. O custo dele é medido
abaixo, e a seção 8.5 estima quanto um radix ganharia.

**Por que um scan para os ids.** É o jeito de dar a cada transação um
intervalo de ids exclusivo sem atômicos. O resultado é determinístico e
contíguo, o que deixa os acessos posteriores coalescidos.

### 8.3 Ideia 2 — Pais e contadores com sparse table (sem atômicos)

Pais e contadores dependem de *outras* transações. Ambos viram consultas de
mínimo em intervalo sobre o vetor LCP.

- **Pai** de um nó da transação `i` na profundidade `d`:
  - se `d > LCPᵢ`, é o nó anterior da própria transação;
  - se `d = LCPᵢ > 0`, foi criado por `k = maior índice < i com LCPₖ < LCPᵢ`
    (*previous smaller value*), e o id é `baseₖ + LCPᵢ − 1 − LCPₖ`;
  - se `d = 0`, é a raiz.
- **Contador** do nó criado em `(k, d)`: o nó é compartilhado por todas as
  transações seguintes enquanto o LCP ficar acima de `d`. Com
  `m = menor índice > k com LCPₘ ≤ d` (*next smaller or equal*), o contador é
  `m − k`. Com pesos (usado na mineração), é `W[m] − W[k]`, com `W` = scan dos pesos.
  Transações duplicadas não criam nós e entram na contagem automaticamente.

**Mecanismo.** As duas consultas são respondidas por busca binária
descendente numa **sparse table** de mínimos, `S_j[i] = min(LCP[i .. i+2^j))`:

- a tabela é construída em ⌈log₂ n⌉ passos paralelos (`construcao_sparse`);
- depois, uma thread por nó calcula seu pai (`construcao_pais`) e seu contador
  (`construcao_contadores`) de forma independente, sem atômicos.

**Alternativas consideradas e por que perderam.**

- **Varredura com pilha por uma única thread.** Resolve os dois problemas em
  O(n), mas é serial. Seria aceitável na árvore global, mas não nas árvores
  condicionais construídas dentro de cada bloco (seção 9).
- **Contadores com `reduce_by_key`** sobre o id do nó percorrido por cada
  posição (i, d). Funciona e foi mantido como **verificação cruzada** no modo
  debug, mas exige uma ordenação de todas as ocorrências. A sparse table é mais
  barata e reaproveita a estrutura dos pais.

**Por que a sparse table e não uma árvore de segmentos.** A construção é um
*stencil* trivialmente paralelo, a consulta tem custo O(log n) sem
ramificações complexas, e a memória n·log n cabe com folga: 80 MB em kosarak.

### 8.4 Ideia 3 — Tabela de cabeçalho por ordenação

A lista ligada por item da versão original vira um vetor `header_nodes[]`
agrupado por item:

- `stable_sort_by_key` dos ids de nó pela chave `item`;
- `lower_bound` vetorizado, que dá `header_offsets[]`;
- `reduce_by_key`, que dá o suporte de cada item (`construcao_cabecalho`).

**Por quê:** a mineração precisa, para cada item, de todos os seus nós em
memória contígua. A ordenação estável mantém os nós de cada item em pré-ordem.

### 8.5 Desempenho da construção

Mediana de 5, ms (versão final; estática e dinâmica executam o mesmo código de construção):

| Dataset | minsup | Nós | Seq. | GPU | Speedup | sort | LCP | scan | nós | sparse | pais | cont. | cabeç. | % sort | Speedup sem o sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 15,7 k | 1,06 | 3,48 | 0,3× | 3,05 | 0,025 | 0,017 | 0,045 | 0,036 | 0,016 | 0,018 | 0,271 | 88 % | 2,5× |
| chess | 0,35 | 26,0 k | 1,64 | 3,61 | 0,5× | 3,17 | 0,026 | 0,017 | 0,048 | 0,037 | 0,016 | 0,021 | 0,275 | 88 % | 3,7× |
| mushroom | 0,05 | 22,3 k | 2,24 | 2,54 | 0,9× | 2,09 | 0,025 | 0,018 | 0,049 | 0,039 | 0,016 | 0,015 | 0,281 | 82 % | 5,0× |
| mushroom | 0,01 | 26,6 k | 2,45 | 3,56 | 0,7× | 3,13 | 0,023 | 0,018 | 0,044 | 0,039 | 0,016 | 0,016 | 0,274 | 88 % | 5,7× |
| accidents | 0,3 | 528 k | 100 | 9,80 | 10,2× | 8,26 | 0,180 | 0,026 | 0,325 | 0,155 | 0,029 | 0,026 | 0,793 | 84 % | 65× |
| accidents | 0,1 | 2,87 M | 402 | 10,9 | **37,0×** | 8,56 | 0,198 | 0,023 | 0,484 | 0,149 | 0,071 | 0,058 | 1,33 | 79 % | 174× |
| T10I4D100K | 0,0002 | 715 k | 53,3 | 3,44 | 15,5× | 2,12 | 0,029 | 0,021 | 0,348 | 0,108 | 0,030 | 0,026 | 0,790 | 62 % | 40× |
| T10I4D100K | 0,00003 | 715 k | 51,2 | 2,17 | 23,6× | 0,96 | 0,027 | 0,019 | 0,308 | 0,096 | 0,028 | 0,025 | 0,711 | 44 % | 42× |
| retail | 0,0002 | 626 k | 49,6 | 3,59 | 13,8× | 2,24 | 0,027 | 0,022 | 0,356 | 0,108 | 0,029 | 0,026 | 0,789 | 63 % | 37× |
| retail | 0,00003 | 673 k | 48,8 | 3,26 | 15,0× | 2,03 | 0,026 | 0,020 | 0,326 | 0,097 | 0,028 | 0,024 | 0,709 | 62 % | 40× |
| kosarak | 0,002 | 1,99 M | 179 | 7,01 | 25,5× | 4,76 | 0,131 | 0,025 | 0,489 | 0,255 | 0,070 | 0,058 | 1,03 | 68 % | 79× |
| kosarak | 0,0009 | 3,02 M | 318 | 7,12 | **44,7×** | 4,51 | 0,129 | 0,024 | 0,652 | 0,247 | 0,109 | 0,115 | 1,31 | 63 % | 122× |

(Tabela com as 24 configurações no [apêndice B](#apêndice-b--construção-todas-as-configurações).)

**Leitura.**

- **Árvores grandes** (0,5 M a 3,1 M nós): 10–45× mais rápida que a inserção
  sequencial. O ganho cresce com o tamanho; accidents vai de 10× com 0,53 M nós
  a 37× com 2,87 M.
- **Árvores pequenas** (chess e mushroom, 16–27 mil nós): a GPU perde (0,3–0,9×).
  A construção sequencial inteira leva 1–2,5 ms, menos que o custo fixo da
  ordenação na GPU, que tem dezenas de lançamentos e alocações de temporários.
- **A ordenação lexicográfica domina**, com 42–88 % do tempo da fase. O
  cabeçalho, que é outra ordenação, vem em segundo.
- **Pais e contadores**, as etapas que substituem a busca por ponteiros, custam
  no máximo 0,12 ms mesmo com 3,1 M nós.
- **Variação da ordenação:** o tempo do sort oscila entre execuções (ex.:
  T10I4D100K 0,96 a 2,1 ms com o mesmo número de nós). É ruído do custo fixo de
  alocação, não do trabalho.

**Quão perto do limite estão as subetapas** (kosarak 0,00085: 962 k transações, 3,08 M nós):

| Subetapa | Tráfego mínimo (MB) | Tempo (ms) | GB/s | % da banda medida (920 GB/s) |
| --- | ---: | ---: | ---: | ---: |
| LCP | 63 | 0,137 | 462 | 50 % |
| Escrita dos nós | 61 | 0,676 | 90 | 10 % |
| Sparse table | 242 | 0,257 | 944 | 103 % |
| Pais | 49 | 0,114 | 433 | 47 % |
| Contadores | 49 | 0,120 | 409 | 44 % |

Notas sobre a tabela:

- A sparse table opera na banda máxima; passa de 100 % porque parte dos dados
  está no L2.
- LCP, pais e contadores ficam em 44–50 %, limitados pelos acessos aleatórios
  das buscas binárias.
- A **escrita dos nós** é a ineficiente (10 %): uma thread escreve todos os nós
  novos de sua transação em sequência, com escritas não coalescidas. Um warp por
  transação longa, ou uma expansão por scan com uma thread por nó, resolveria.

**Ganho ainda possível com uma ordenação melhor.** Se a ordenação levasse o
tempo de um radix sort de 64 bits com o mesmo número de chaves (medido: 0,07 ms
em chess, 0,43 ms em kosarak), a construção chegaria a até **159×** sobre o
sequencial (accidents 0,1). As transações não cabem numa palavra, mas uma
versão híbrida funciona: radix pelos primeiros ranks ou por um hash do prefixo,
e o comparador só para desempatar dentro de grupos pequenos.

**Corretude:** a árvore é idêntica à sequencial (formato textual canônico e
número de nós) em 19 configurações; os contadores sem atômicos batem com os do
`reduce_by_key`; e a suíte cobre transações duplicadas, prefixo da seguinte,
tamanho 1, todas iguais, vazias e árvores ponderadas.

---

## 9. Fase 5 — Mineração

### 9.1 Implementação original

Recursão clássica (`Miner::mine_tree` em `src/seq/fpgrowth_seq.cpp`):

```
mine(T, S):
  se T é um caminho único: emite todas as combinações dos nós ∪ S; retorna
  para cada item y de T, do menos para o mais frequente:
     emite S ∪ {y} com o suporte de y
     cnt ← histograma ponderado dos itens nos caminhos raiz → nós de y
     C   ← árvore com esses caminhos, sem os itens com cnt < minsup
     mine(C, S ∪ {y})
```

As árvores condicionais mantêm a **ordem global de ranks**, sem reordenar por
frequência condicional. O resultado é o mesmo, e os caminhos extraídos já saem
ordenados. A GPU faz igual.

**Tempo original:** 86 ms (retail 0,0002) a 6,4 s (mushroom 0,01). É a maior
fase algorítmica em todos os datasets.

### 9.2 Ideia 1 — Mineração estática: um bloco de threads por item

**O que é.** O kernel `mine_static_kernel` lança um bloco de 256 threads para
cada item frequente da árvore global. O bloco resolve sozinho toda a recursão
do seu item:

1. emite `{x}`;
2. constrói a árvore condicional de `x`;
3. continua com uma **pilha explícita** de quadros (árvore, próximo item,
   tamanho do sufixo).

Recursão de verdade em GPU seria limitada pela pilha de cada thread e
impossível de dividir entre as threads do bloco.

**Mecanismos e o motivo de cada um.**

| Mecanismo | Como funciona | Por que foi escolhido |
| --- | --- | --- |
| **Extração da base em bloco** | scan das profundidades dos nós de `y` dá a posição de cada caminho; cada thread sobe pelos pais escrevendo o caminho de trás para frente | o caminho já sai em ordem crescente de rank, sem ordenar; o tamanho de cada caminho é conhecido (`depth`), então não há alocação dinâmica |
| **Histograma condicional em memória compartilhada** | feito durante a extração, com até 10.240 itens em shared; acima disso vai para a memória global | mesma razão da contagem v2 (seção 5.3) |
| **Construção da árvore condicional pelo bloco** | as *mesmas* funções de dispositivo da construção global (`tree_core.cuh`), parametrizadas por `(threadIdx, blockDim)` em vez de `(id global, tamanho da grade)` | reaproveita código já validado: LCP, scan, sparse table, pais, contadores |
| **Merge sort de bloco** | cada elemento acha sua posição no resultado da intercalação por busca binária na outra metade (estável) | `cub::BlockMergeSort` exige tamanho fixo por bloco; as bases têm tamanho qualquer |
| **Pais e contadores pela sparse table em bloco** | log n passos com `__syncthreads` | a alternativa sugerida (varredura com pilha por uma thread) deixaria 255 threads paradas |
| **Arena de duas pontas por bloco** | por baixo: quadros, sufixo e árvores condicionais (LIFO, liberadas ao desempilhar); por cima: temporários da construção, liberados ao fim de cada árvore | sem `malloc` em GPU; todas as threads fazem a mesma sequência de alocações e guardam cópias idênticas dos ponteiros, então não há sincronização para alocar |
| **2 blocos por SM** (40 KB de shared dinâmica) | 256 blocos residentes; cada um pega uma arena livre com `atomicCAS` num vetor de travas | arenas grandes (~24 MB cada, até 35 % da memória livre); como no máximo 256 blocos estão residentes, sempre há arena livre |
| **Atalho de caminho único** | se cada caminho estende o anterior, a árvore é um caminho de L nós e as 2^L − 1 combinações são emitidas em paralelo; o tamanho total da saída é calculado analiticamente, e cada thread acha a posição da máscara k contando os bits 1 de 1..k−1 | nenhum atômico por itemset; nos densos, a maioria dos itemsets sai por aqui |
| **Saída em duas passadas** | passada 1 só conta itemsets e itens *por item raiz*; o host aloca buffers exatos; passada 2 escreve, reservando posições com `atomicAdd` | sem alocação dinâmica nem buffers estourando; a contagem por raiz permite fazer a passada 2 em lotes quando a saída não cabe na GPU |
| **Estouro de arena** | o bloco registra o itemset X e segue; o host projeta a base em X *na GPU* (`{t ∩ [0, min X) : t ⊇ X}`), constrói a árvore com a rotina global da seção 8 e relança a mineração com prefixo X | correto sem depender do tamanho da arena; o fallback sequencial nunca foi necessário |

**Desempenho:**

| Dataset | minsup | Itemsets | Seq. (ms) | Estática (ms) | Estática vs. seq. | Bloco mais lento / média |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 1,27 M | 229 | 880 | **0,3×** | 5,4 |
| chess | 0,35 | 15,1 M | 2.421 | 6.043 | **0,4×** | 4,4 |
| mushroom | 0,05 | 3,76 M | 250 | 248 | 1,0× | 10,7 |
| mushroom | 0,01 | 90,8 M | 6.424 | 4.747 | 1,4× | 7,9 |
| accidents | 0,3 | 0,15 M | 145 | 196 | **0,7×** | 3,4 |
| accidents | 0,1 | 10,7 M | 4.942 | 10.678 | **0,5×** | 9,1 |
| T10I4D100K | 0,0002 | 0,13 M | 186 | 10,2 | **18,3×** | 4,9 |
| T10I4D100K | 0,00003 | 6,17 M | 1.589 | 292 | 5,4× | 7,3 |
| retail | 0,0002 | 0,07 M | 85,6 | 6,3 | 13,5× | 22,3 |
| retail | 0,00003 | 20,6 M | 2.887 | 5.572 | **0,5×** | 1.644 |
| kosarak | 0,002 | 0,04 M | 386 | 67,7 | 5,7× | 16,0 |
| kosarak | 0,00085 | 6,46 M | 4.816 | 9.374 | **0,5×** | 221 |

A estática é **mais lenta que a CPU em 12 das 24 configurações**. Funciona bem
só quando há muitos itens com carga parecida (T10I4D100K, e retail e kosarak em
suporte alto). Os motivos estão na distribuição do tempo por bloco (suporte
intermediário):

| Dataset | Blocos (itens) | Bloco mais lento (ms) | Média (ms) | Eficiência* |
| --- | ---: | ---: | ---: | ---: |
| chess | 39 | 968 | 187 | 3 % |
| mushroom | 82 | 208 | 25,6 | 4 % |
| accidents | 48 | 580 | 103 | 3 % |
| T10I4D100K | 867 | 15,1 | 2,26 | 51 % |
| retail | 8.919 | 29,4 | 0,14 | 17 % |
| kosarak | 1.259 | 558 | 4,22 | 4 % |

\* trabalho total dividido por 256 blocos residentes, sobre o tempo do bloco
mais lento. É a fração do tempo em que a GPU inteira poderia estar ocupada.

Três problemas:

- **Pouco paralelismo nos densos:** chess tem 39 itens, ou seja, 39 blocos
  numa GPU que comporta 256. Cada bloco constrói centenas de milhares de árvores
  condicionais em série.
- **Desbalanceamento extremo nos esparsos:** em kosarak 0,001, 1.258 blocos
  terminam em poucos ms e um único bloco leva 558 ms, o tempo do kernel inteiro.
  Em retail 0,00003 a razão entre o bloco mais lento e a média chega a 1.644.
- **Custo dobrado pela saída em duas passadas:** a passada 1 é 47 % do tempo de
  mineração em quase todas as configurações.

### 9.3 Ideia 2 — Mineração dinâmica: fila global de tarefas com blocos persistentes

**O que é.** O kernel `mine_dynamic_kernel` lança exatamente o número de blocos
que cabem residentes ao mesmo tempo (256). Esses blocos não terminam:

1. cada bloco repete o ciclo "pegar tarefa da fila → processar";
2. a fila começa com uma tarefa por item raiz;
3. quando um bloco constrói uma árvore condicional *grande* (ver `grain`, seção
   9.4), ele a copia para um pool global e a publica como tarefa nova, que
   qualquer bloco ocioso pode pegar;
4. árvores pequenas são resolvidas localmente, em profundidade, como na estática.

Assim, uma subárvore pesada de kosarak deixa de ficar presa num bloco: seus
filhos se espalham pelos 256.

**Mecanismos e o motivo de cada um.**

| Mecanismo | Como funciona | Por que foi escolhido |
| --- | --- | --- |
| **Blocos persistentes** | grade = blocos residentes (API de ocupação × SMs) | um bloco que espera por tarefa nunca impede outro de rodar, o que é condição para a fila não travar; e não há custo de lançar kernels novos |
| **Fila circular MPMC sem travas** | descritores num anel, `head` e `tail` de 64 bits e um número de sequência por posição (posição + 1 quando publicada, 0 quando livre). O produtor reserva com `atomicCAS(tail)`, espera a posição ficar livre, escreve, faz `__threadfence` e publica; o consumidor reserva com `atomicCAS(head)` e espera a publicação | o número de sequência resolve a corrida entre "reservei a posição" e "o dado está escrito" sem travas; as esperas são curtas porque todos os blocos estão residentes |
| **Terminação por contador de pendentes** | incrementado *antes* de publicar uma tarefa, decrementado quando a tarefa e toda a sua subárvore local terminam; um bloco sai quando a fila está vazia e o contador é zero | sem esse contador, um bloco poderia sair enquanto outro ainda vai publicar filhos |
| **Pool global por *bump* atômico** | árvores publicadas são copiadas para um pool (até 15 % da memória livre) com um `atomicAdd` | a arena do bloco é LIFO e seria liberada; a cópia desacopla a tarefa do bloco que a criou |
| **Degradação graciosa** | fila cheia ou pool esgotado → o filho é processado localmente | nunca perde corretude; no pior caso volta a ser a versão estática |
| **Duas passadas com folga** | o destino de cada filho (fila ou local) depende do estado da fila, então os estouros podem variar entre passadas; a passada 2 tem 12,5 % de folga e repete com o dobro se encher | corretude mesmo com não determinismo |

**Desempenho** (`grain` = 64):

| Dataset | minsup | Seq. (ms) | Estática (ms) | Dinâmica (ms) | Din. vs. seq. | Din. vs. est. | Mais lento / média: est. → din. | Tarefas na fila |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| chess | 0,5 | 229 | 880 | 28,0 | 8,2× | **31×** | 5,4 → 1,09 | 4.190 |
| chess | 0,35 | 2.421 | 6.043 | 277 | 8,8× | 22× | 4,4 → 1,01 | 38.203 |
| mushroom | 0,05 | 250 | 248 | 20,4 | 12× | 12× | 10,7 → 2,78 | 2.533 |
| mushroom | 0,02 | 1.661 | 976 | 40,2 | **41×** | 24× | 10,8 → 1,11 | 10.552 |
| mushroom | 0,01 | 6.424 | 4.747 | 173 | 37× | 27× | 7,9 → 4,88 | 33.890 |
| accidents | 0,3 | 145 | 196 | 94,7 | 1,5× | 2,1× | 3,4 → 10,2 | 2.917 |
| accidents | 0,1 | 4.942 | 10.678 | 1.047 | 4,7× | 10× | 9,1 → 1,41 | 158.955 |
| T10I4D100K | 0,0002 | 186 | 10,2 | 9,2 | 20× | 1,1× | 4,9 → 1,50 | 4.297 |
| T10I4D100K | 0,00003 | 1.589 | 292 | 138 | 11× | 2,1× | 7,3 → 1,03 | 41.098 |
| retail | 0,0002 | 85,6 | 6,3 | 5,6 | 15× | 1,1× | 22 → 1,30 | 3.909 |
| retail | 0,00003 | 2.887 | 5.572 | 424 | 6,8× | 13× | 1.644 → 2,34 | 16.078 |
| kosarak | 0,001 | 1.113 | 993 | 40,4 | 28× | 25× | 132 → 1,01 | 21.462 |
| kosarak | 0,00085 | 4.816 | 9.374 | 285 | 17× | **33×** | 221 → 1,00 | 268.824 |

(Todas as 24 no [apêndice C](#apêndice-c--mineração-todas-as-configurações).)

Resumo: a dinâmica é mais rápida que a estática em **todas** as 24
configurações (1,1–33×) e mais rápida que a CPU em **todas** (1,5–41×).

**Por dataset:**

- **Poucos itens** (chess): 22–31× sobre a estática; os 256 blocos passam a
  trabalhar.
- **Carga concentrada** (kosarak, retail em suporte baixo): 13–33×. A razão
  entre o bloco mais lento e a média vai de 132–1.644 para 1,0–2,3.
- **Carga já uniforme** (T10I4D100K): ganho pequeno, 1,1–2,1×.
- **Árvores condicionais gigantes no primeiro nível** (accidents): 2,1–10×.
  Construir uma árvore é indivisível entre blocos, e os primeiros itens de
  accidents têm bases de centenas de milhares de caminhos. Em accidents 0,3, um
  único bloco gasta ~90 ms nelas.

**Eficiência de balanceamento** (suporte intermediário; mesma métrica da tabela da estática):

| Dataset | Estática | Dinâmica |
| --- | ---: | ---: |
| chess | 3 % | **96 %** |
| mushroom | 4 % | 62 % |
| accidents | 3 % | 24 % |
| T10I4D100K | 51 % | 79 % |
| retail | 17 % | 77 % |
| kosarak | 4 % | **99 %** |

### 9.4 Ideia 3 — Ajuste da granularidade (`grain`)

**O que é.** O limiar, em itens da base condicional filtrada, a partir do qual
uma árvore filha vai para a fila em vez de ser resolvida localmente.

**Por que existe.** Enfileirar uma tarefa custa copiar a árvore para o pool e
passar pela fila. Para tarefas minúsculas esse custo supera o ganho de
balanceamento; para tarefas grandes, não enfileirar devolve o desbalanceamento.

**Desempenho** (mineração dinâmica em ms, suporte intermediário; o melhor em negrito):

| Dataset | Estática | g = 0 | g = 16 | g = 64 | g = 256 | g = 1024 | g = 4096 | Melhor vs. estática | 64 vs. melhor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 1.711 | 115 | 61,9 | **60,4** | 110 | 692 | 1.740 | 28× | 1,00× |
| mushroom | 375 | 19,2 | **17,8** | 26,8 | 121 | 177 | 294 | 21× | 1,5× |
| accidents | 1.029 | **246** | 256 | 267 | 257 | 266 | 295 | 4,2× | 1,1× |
| T10I4D100K | 28,8 | 20,0 | **18,4** | 18,8 | 19,3 | 19,6 | 24,1 | 1,6× | 1,02× |
| retail | 57,5 | 20,4 | 13,6 | **12,5** | 57,4 | 55,5 | 55,5 | 4,6× | 1,00× |
| kosarak | 993 | 48,0 | 43,3 | **40,7** | 43,1 | 378 | 1.019 | 24× | 1,00× |

**Leitura.**

- **`grain` = 0** (tudo para a fila): o custo das cópias aparece; em chess
  custa 2× mais que com 64.
- **`grain` ≥ 1024:** quase nada vai para a fila, e a dinâmica volta a ter o
  tempo da estática (kosarak: 1.019 ms contra 40,7 ms).
- **Escolha:** os melhores valores ficam entre 16 e 64 em todos os datasets.
  O padrão escolhido foi **64**, que fica no máximo 1,5× do melhor (mushroom).
- **Ajuste por dataset:** escolher o grain por dataset daria no máximo 1,5×
  extra em mushroom e nada nos outros.

### 9.5 Limites e o que ainda falta na mineração

- **Balanceamento:** onde a eficiência é 62–79 % (mushroom, retail, T10I4D100K),
  um balanceamento perfeito daria no máximo 1,3–1,6×.
- **Tarefa indivisível** (accidents, 24 %): dividir a construção das árvores
  condicionais gigantes entre vários blocos, ou mandá-las para a rotina de grade
  inteira (que já existe para os estouros), daria até ~4×.
- **Duas passadas:** a passada 1 é 28–50 % do tempo (mediana 47 %). Uma passada
  única com buffers otimistas, que crescem e retomam se estourarem, daria até 2×.
- **Memória:** as arenas e o pool são reservados de forma fixa. O pico da GPU
  fica em 9,3–13,5 GB na dinâmica e 6,2–10,3 GB na estática, independentemente
  do dataset. Só accidents 0,1 teve estouro de arena (14 tarefas, resolvidas na
  GPU).

---

## 10. Fase 6 — Retorno (D2H) e escrita da saída

**Implementação original (escrita).** Cada itemset tem os ranks traduzidos para
ids, os itens ordenados e é formatado como `"i1 i2 … (suporte)"`. Todas as
linhas são ordenadas (`std::sort` de strings) e gravadas. Tudo em uma thread.
É idêntico nas três versões, para que a saída seja comparável byte a byte.

**D2H + montagem** (só na GPU). Os buffers de saída (itens, início, tamanho e
suporte de cada itemset) são copiados e convertidos para o formato CSR de
itemsets do host. A medição desta etapa inclui as duas coisas, e **não foi
separada**.

| Dataset | minsup | Itemsets | D2H + montagem, GPU (ms) | Escrita, seq. (ms) | Escrita, GPU (ms) | ns/itemset (seq.) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chess | 0,35 | 15,1 M | 630 | 9.376 | 11.256 | 621 |
| mushroom | 0,01 | 90,8 M | 6.820 | 72.213 | 74.563 | 796 |
| accidents | 0,1 | 10,7 M | 2.812 | 7.476 | 7.828 | 699 |
| T10I4D100K | 0,00003 | 6,17 M | 204 | 2.793 | 3.087 | 453 |
| retail | 0,00003 | 20,6 M | 894 | 15.931 | 15.160 | 772 |
| kosarak | 0,00085 | 6,46 M | 284 | 4.433 | 4.660 | 687 |

Observações:

- **Custo da escrita:** 390–800 ns por itemset. Com dezenas de milhões de
  itemsets, a escrita sozinha leva de segundos a mais de um minuto, mais que
  todo o resto do programa junto.
- **Escrita mais lenta na GPU:** para a mesma saída, ela é 2–27 % mais lenta
  que no sequencial em chess, mushroom e accidents. Nos esparsos varia de −5 %
  a +30 %, conforme a configuração. A hipótese, **não verificada**, é que os
  itemsets chegam da GPU numa ordem mais embaralhada (reservas por `atomicAdd`
  e lotes), o que deixa a ordenação das linhas mais cara.
- **D2H + montagem:** chega a 6,8 s em mushroom 0,01. A cópia em si é limitada
  pelo PCIe (~24 GB/s), então a maior parte deve ser a montagem item a item no
  host (`ItemsetList::add`); isso também não foi separado na medição. Gerar a
  saída já no formato final na GPU, ou montar o CSR com cópias em bloco,
  eliminaria esse custo.

**Ideias de melhoria (nenhuma implementada):**

1. formatação e ordenação paralelas nos 12 núcleos, com ganho ideal de ~12× na escrita;
2. formatação na GPU, deixando só a transferência do texto;
3. buffer único de texto com offsets ordenados, que reduz à metade a memória
   por itemset. Esta foi desenhada, mas não aplicada; ver `NOTES.md`.

---

## 11. Comparação final: tudo junto

**Versão final:** leitura (CPU) → H2D → contagem v2 → pré-processamento por
primitivas → construção por ordenação + LCP + sparse table → mineração dinâmica
(`grain` 64) → D2H → escrita (CPU).

### 11.1 Speedup de cada fase e total, por configuração

Speedup = tempo do sequencial / tempo da versão final. Núcleo = contagem +
pré-processamento + construção + mineração. "Amdahl" é o teto do speedup
ponta a ponta se o núcleo levasse tempo zero: `T_seq / (T_leitura + T_escrita)`.

| Dataset | minsup | Contagem | Pré-proc. | Construção | Mineração | Núcleo seq. → GPU (ms) | **Núcleo** | Núcleo + transf. | Total seq. → GPU (ms) | **Total** | Amdahl |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| chess | 0,5 | 0,1× | 1,4× | 0,3× | 8,2× | 231 → 32,9 | **7,0×** | 2,7× | 882 → 867 | **1,0×** | 1,4× |
| chess | 0,45 | 0,1× | 1,3× | 0,6× | 7,9× | 480 → 64,2 | **7,5×** | 2,6× | 2.003 → 2.041 | **1,0×** | 1,3× |
| chess | 0,4 | 0,1× | 1,3× | 0,6× | 8,8× | 1.119 → 132 | **8,5×** | 2,7× | 4.842 → 4.907 | **1,0×** | 1,3× |
| chess | 0,35 | 0,1× | 1,2× | 0,5× | 8,8× | 2.424 → 282 | **8,6×** | 2,7× | 11.899 → 12.227 | **1,0×** | 1,3× |
| mushroom | 0,05 | 0,1× | 1,2× | 0,9× | 12,3× | 253 → 24,5 | **10,4×** | 1,4× | 2.244 → 2.240 | **1,0×** | 1,1× |
| mushroom | 0,03 | 3,4× | 1,3× | 0,7× | 28,8× | 773 → 31,0 | **25,0×** | 1,6× | 6.396 → 6.252 | **1,0×** | 1,1× |
| mushroom | 0,02 | 0,1× | 1,3× | 0,7× | 41,4× | 1.664 → 45,2 | **36,8×** | 1,7× | 15.182 → 14.863 | **1,0×** | 1,1× |
| mushroom | 0,01 | 3,5× | 1,3× | 0,7× | 37,2× | 6.428 → 177 | **36,3×** | 0,9× | 78.642 → 81.893 | **1,0×** | 1,1× |
| accidents | 0,3 | 64× | 33× | 10,2× | 1,5× | 327 → 107 | **3,1×** | 2,8× | 464 → 313 | **1,5×** | 3,4× |
| accidents | 0,2 | 6,4× | 41× | 24,2× | 2,7× | 1.006 → 268 | **3,8×** | 3,3× | 1.549 → 906 | **1,7×** | 2,9× |
| accidents | 0,15 | 5,4× | 44× | 36,8× | 4,0× | 2.058 → 417 | **4,9×** | 3,8× | 3.748 → 2.333 | **1,6×** | 2,2× |
| accidents | 0,1 | 6,2× | 46× | 37,0× | 4,7× | 5.453 → 1.061 | **5,1×** | 1,4× | 13.015 → 11.882 | **1,1×** | 1,7× |
| T10I4D100K | 0,0002 | 0,5× | 7,6× | 15,5× | 20,3× | 249 → 14,6 | **17,0×** | 12,9× | 298 → 101 | **2,9×** | 6,5× |
| T10I4D100K | 0,0001 | 0,6× | 8,6× | 17,5× | 17,2× | 386 → 23,5 | **16,4×** | 11,0× | 518 → 196 | **2,6×** | 3,9× |
| T10I4D100K | 0,00005 | 13,5× | 8,5× | 17,1× | 14,1× | 808 → 57,3 | **14,1×** | 7,1× | 1.571 → 975 | **1,6×** | 2,1× |
| T10I4D100K | 0,00003 | 0,6× | 8,1× | 23,6× | 11,5× | 1.648 → 141 | **11,7×** | 4,8× | 4.449 → 3.479 | **1,3×** | 1,6× |
| retail | 0,0002 | 0,5× | 6,7× | 13,8× | 15,2× | 144 → 11,3 | **12,8×** | 10,4× | 171 → 74,3 | **2,3×** | 6,4× |
| retail | 0,0001 | 0,6× | 7,7× | 21,6× | 12,6× | 214 → 16,5 | **13,0×** | 9,1× | 296 → 136 | **2,2×** | 3,6× |
| retail | 0,00005 | 0,5× | 7,8× | 15,0× | 9,3× | 488 → 51,1 | **9,5×** | 4,8× | 1.273 → 909 | **1,4×** | 1,6× |
| retail | 0,00003 | 0,6× | 7,7× | 15,0× | 6,8× | 2.944 → 430 | **6,9×** | 2,2× | 18.883 → 16.602 | **1,1×** | 1,2× |
| kosarak | 0,002 | 4,4× | 21× | 25,5× | 22,6× | 624 → 27,7 | **22,5×** | 19,5× | 790 → 227 | **3,5×** | 4,8× |
| kosarak | 0,001 | 20× | 24× | 37,3× | 27,6× | 1.487 → 51,5 | **28,9×** | 17,6× | 2.088 → 705 | **3,0×** | 3,4× |
| kosarak | 0,0009 | 4,5× | 24× | 44,7× | 22,0× | 2.151 → 90,8 | **23,7×** | 13,1× | 3.445 → 1.493 | **2,3×** | 2,7× |
| kosarak | 0,00085 | 4,4× | 23× | 37,5× | 16,9× | 5.207 → 297 | **17,5×** | 8,9× | 9.783 → 5.452 | **1,8×** | 2,1× |

### 11.2 Agregado

| Medida | Mín. | Mediana | Máx. | Média geométrica |
| --- | ---: | ---: | ---: | ---: |
| Contagem | 0,06× | 0,6× | 64× | 1,1× |
| Pré-processamento | 1,2× | 7,7× | 46× | 6,7× |
| Construção | 0,3× | 15× | 45× | 6,6× |
| Mineração | 1,5× | 12× | 41× | 11,6× |
| **Núcleo** | **3,1×** | **12×** | **37×** | **11,8×** |
| Núcleo + transferências | 0,9× | 3,6× | 20× | 4,3× |
| **Total ponta a ponta** | **0,96×** | **1,4×** | **3,5×** | **1,5×** |
| Teto de Amdahl (núcleo = 0) | 1,1× | 1,9× | 6,5× | — |

Soma das 24 configurações (cada uma pesa pelo seu tempo):

| Versão | Núcleo (s) | Total (s) |
| --- | ---: | ---: |
| Sequencial | 38,6 | 184,4 |
| GPU estática | 51,0 | 216,0 |
| **GPU dinâmica (final)** | **3,85** | **171,1** |
| Speedup final | **10,0×** | **1,08×** |

O total somado é dominado por mushroom 0,01, que sozinho leva 79 s, dos quais
72 s são de escrita.

### 11.3 Contribuição de cada ideia

| Ideia | Fase | Ganho isolado medido |
| --- | --- | --- |
| Histograma em shared (v2) sobre atômicos globais (v1) | contagem | 2,1–53× no kernel |
| Primitivas (scan + sort segmentado) sobre CPU | pré-processamento | 1,2–46× |
| Ordenação + LCP + scan + sparse table sobre inserção | construção | 10–45× com árvores ≥ 0,5 M nós; 0,3–0,9× com árvores pequenas |
| Mineração estática (1 bloco/item) sobre CPU | mineração | 0,3–18×; perde em 12/24 |
| Fila dinâmica sobre estática | mineração | 1,1–33× |
| `grain` 64 sobre `grain` 0 | mineração | até 1,9× (chess) |
| `grain` 64 sobre `grain` 4096 | mineração | até 25× (kosarak) |

**Onde o tempo está na versão final** (suporte intermediário):

| Dataset | Leitura | H2D | Núcleo (GPU) | D2H + montagem | Escrita | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chess 0,45 | 0 % | 0 % | 3 % | 6 % | 89 % | 2.041 ms |
| kosarak 0,001 | 22 % | 0 % | 7 % | 4 % | 62 % | 705 ms |

A GPU deixou o núcleo do algoritmo pequeno o bastante para que **a leitura e a
escrita na CPU virassem quase todo o tempo**. É exatamente o que o teto de
Amdahl prevê.

---

## 12. O que falta e o ganho que ainda é possível

Em ordem de impacto esperado no tempo total:

| # | Melhoria | Fase | Ganho esperado | Base da estimativa |
| --- | --- | --- | --- | --- |
| 1 | Escrita paralela (12 núcleos) ou formatação na GPU | escrita | até ~12× na escrita; ponta a ponta nos densos sai de ~1,0× para perto do speedup do núcleo | a escrita chega a 89–91 % do tempo final (chess, mushroom) |
| 2 | Montar a saída em bloco (sem `add` por itemset) ou no formato final na GPU | D2H | até 6,8 s em mushroom 0,01 | D2H + montagem é até 24 % do tempo final |
| 3 | Parsing paralelo ou na GPU | leitura | até ~12× na leitura | leitura é 22–69 % do final em kosarak (suportes 0,001 e 0,002) |
| 4 | Saída em passada única | mineração | até 2× na mineração | passada 1 = 47 % |
| 5 | Dividir as construções condicionais gigantes entre blocos | mineração | até ~4× em accidents | eficiência de 24 % |
| 6 | Ordenação lexicográfica por radix híbrido | construção | construção até 159× sobre seq. (hoje 37×) | radix de 64 bits medido |
| 7 | Alocador com cache e carregamento antecipado de módulos | contagem, pré-proc., construção | elimina ~2–3 ms fixos; vira ganho nas árvores pequenas | kernels isolados 10–100× mais rápidos que a fase |
| 8 | Memória *pinned* + sobreposição | H2D | ~1–3 ms | banda medida 12–15 GB/s contra 24 GB/s |

---

## 13. Reprodução

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./scripts/download_data.sh
PY=/caminho/python3 ./scripts/run_all.sh      # validação + benchmarks + gráficos
```

Todas as medições usam a GPU do barramento `0000:01:00.0` (slot x16),
configurável por `FPG_GPU_BUS`. Arquivos de onde vêm os números deste relatório:

| Arquivo | Conteúdo |
| --- | --- |
| `results/raw.csv` | tempo de cada etapa de cada execução |
| `results/runs.csv` | itemsets, nós, estouros, tarefas, desbalanceamento, memória e barramento PCI |
| `results/grain.csv` | experimento de granularidade |
| `results/blocks/*.csv` | tempo, tarefas e itemsets de cada bloco |
| `results/micro/` | limites de hardware e kernels de contagem isolados |

---

## Apêndice A — Tempo sequencial por fase (todas as configurações)

ms (% do total).

| Dataset | minsup | Leitura | Contagem | Pré-proc. | Construção | Mineração | Escrita | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 0,96 (0 %) | 0,047 (0 %) | 0,82 (0 %) | 1,06 (0 %) | 229 (26 %) | 646 (73 %) | 882 |
| chess | 0,45 | 1,02 (0 %) | 0,046 (0 %) | 0,77 (0 %) | 1,54 (0 %) | 478 (24 %) | 1.521 (76 %) | 2.003 |
| chess | 0,4 | 1,02 (0 %) | 0,046 (0 %) | 0,77 (0 %) | 1,46 (0 %) | 1.117 (23 %) | 3.727 (77 %) | 4.842 |
| chess | 0,35 | 0,94 (0 %) | 0,046 (0 %) | 0,72 (0 %) | 1,64 (0 %) | 2.421 (20 %) | 9.376 (79 %) | 11.899 |
| mushroom | 0,05 | 1,30 (0 %) | 0,073 (0 %) | 0,94 (0 %) | 2,24 (0 %) | 250 (11 %) | 1.983 (88 %) | 2.244 |
| mushroom | 0,03 | 1,22 (0 %) | 0,073 (0 %) | 0,96 (0 %) | 2,48 (0 %) | 770 (12 %) | 5.626 (88 %) | 6.396 |
| mushroom | 0,02 | 1,28 (0 %) | 0,074 (0 %) | 0,97 (0 %) | 2,35 (0 %) | 1.661 (11 %) | 13.493 (89 %) | 15.182 |
| mushroom | 0,01 | 1,23 (0 %) | 0,075 (0 %) | 0,96 (0 %) | 2,45 (0 %) | 6.424 (8 %) | 72.213 (92 %) | 78.642 |
| accidents | 0,3 | 81,1 (17 %) | 4,46 (1 %) | 76,8 (17 %) | 100 (22 %) | 145 (31 %) | 55,7 (12 %) | 464 |
| accidents | 0,2 | 80,4 (5 %) | 4,46 (0 %) | 90,8 (6 %) | 229 (15 %) | 682 (44 %) | 462 (30 %) | 1.549 |
| accidents | 0,15 | 81,3 (2 %) | 4,48 (0 %) | 101 (3 %) | 346 (9 %) | 1.607 (43 %) | 1.610 (43 %) | 3.748 |
| accidents | 0,1 | 81,0 (1 %) | 4,47 (0 %) | 105 (1 %) | 402 (3 %) | 4.942 (38 %) | 7.476 (57 %) | 13.015 |
| T10I4D100K | 0,0002 | 8,56 (3 %) | 0,42 (0 %) | 8,52 (3 %) | 53,3 (18 %) | 186 (63 %) | 37,2 (12 %) | 298 |
| T10I4D100K | 0,0001 | 8,45 (2 %) | 0,42 (0 %) | 8,36 (2 %) | 52,9 (10 %) | 324 (63 %) | 124 (24 %) | 518 |
| T10I4D100K | 0,00005 | 8,67 (1 %) | 0,42 (0 %) | 8,31 (1 %) | 50,9 (3 %) | 748 (48 %) | 750 (48 %) | 1.571 |
| T10I4D100K | 0,00003 | 8,41 (0 %) | 0,40 (0 %) | 8,10 (0 %) | 51,2 (1 %) | 1.589 (36 %) | 2.793 (63 %) | 4.449 |
| retail | 0,0002 | 9,65 (6 %) | 0,39 (0 %) | 7,95 (5 %) | 49,6 (29 %) | 85,6 (50 %) | 17,1 (10 %) | 171 |
| retail | 0,0001 | 9,79 (3 %) | 0,39 (0 %) | 8,14 (3 %) | 48,4 (16 %) | 157 (53 %) | 72,1 (24 %) | 296 |
| retail | 0,00005 | 9,23 (1 %) | 0,38 (0 %) | 8,20 (1 %) | 47,0 (4 %) | 432 (34 %) | 772 (61 %) | 1.273 |
| retail | 0,00003 | 9,70 (0 %) | 0,39 (0 %) | 8,38 (0 %) | 48,8 (0 %) | 2.887 (15 %) | 15.931 (84 %) | 18.883 |
| kosarak | 0,002 | 153 (19 %) | 4,32 (1 %) | 55,3 (7 %) | 179 (23 %) | 386 (49 %) | 12,7 (2 %) | 790 |
| kosarak | 0,001 | 154 (7 %) | 4,28 (0 %) | 65,3 (3 %) | 304 (15 %) | 1.113 (53 %) | 452 (22 %) | 2.088 |
| kosarak | 0,0009 | 153 (4 %) | 4,32 (0 %) | 65,6 (2 %) | 318 (9 %) | 1.763 (51 %) | 1.146 (33 %) | 3.445 |
| kosarak | 0,00085 | 153 (2 %) | 4,22 (0 %) | 66,5 (1 %) | 320 (3 %) | 4.816 (49 %) | 4.433 (45 %) | 9.783 |

## Apêndice B — Construção (todas as configurações)

ms; subetapas da GPU.

| Dataset | minsup | Nós | Seq. | GPU | Speedup | sort | LCP | scan | nós | sparse | pais | cont. | cabeç. | % sort | Sem sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 15,7 k | 1,06 | 3,48 | 0,3× | 3,05 | 0,025 | 0,017 | 0,045 | 0,036 | 0,016 | 0,018 | 0,271 | 88 % | 2,5× |
| chess | 0,45 | 18,6 k | 1,54 | 2,62 | 0,6× | 2,19 | 0,026 | 0,017 | 0,045 | 0,038 | 0,016 | 0,016 | 0,270 | 83 % | 3,6× |
| chess | 0,4 | 20,0 k | 1,46 | 2,64 | 0,6× | 2,19 | 0,026 | 0,017 | 0,045 | 0,037 | 0,016 | 0,021 | 0,273 | 83 % | 3,2× |
| chess | 0,35 | 26,0 k | 1,64 | 3,61 | 0,5× | 3,17 | 0,026 | 0,017 | 0,048 | 0,037 | 0,016 | 0,021 | 0,275 | 88 % | 3,7× |
| mushroom | 0,05 | 22,3 k | 2,24 | 2,54 | 0,9× | 2,09 | 0,025 | 0,018 | 0,049 | 0,039 | 0,016 | 0,015 | 0,281 | 82 % | 5,0× |
| mushroom | 0,03 | 25,3 k | 2,48 | 3,47 | 0,7× | 3,02 | 0,022 | 0,018 | 0,049 | 0,038 | 0,015 | 0,016 | 0,271 | 87 % | 5,6× |
| mushroom | 0,02 | 26,1 k | 2,35 | 3,58 | 0,7× | 3,14 | 0,022 | 0,018 | 0,049 | 0,039 | 0,017 | 0,016 | 0,271 | 88 % | 5,4× |
| mushroom | 0,01 | 26,6 k | 2,45 | 3,56 | 0,7× | 3,13 | 0,023 | 0,018 | 0,044 | 0,039 | 0,016 | 0,016 | 0,274 | 88 % | 5,7× |
| accidents | 0,3 | 528 k | 100 | 9,80 | 10,2× | 8,26 | 0,180 | 0,026 | 0,325 | 0,155 | 0,029 | 0,026 | 0,793 | 84 % | 65× |
| accidents | 0,2 | 1,59 M | 229 | 9,47 | 24,2× | 7,77 | 0,175 | 0,023 | 0,353 | 0,129 | 0,044 | 0,038 | 0,951 | 82 % | 135× |
| accidents | 0,15 | 2,27 M | 346 | 9,42 | 36,8× | 7,39 | 0,189 | 0,023 | 0,416 | 0,136 | 0,058 | 0,050 | 1,15 | 78 % | 171× |
| accidents | 0,1 | 2,87 M | 402 | 10,9 | 37,0× | 8,56 | 0,198 | 0,023 | 0,484 | 0,149 | 0,071 | 0,058 | 1,33 | 79 % | 174× |
| T10I4D100K | 0,0002 | 715 k | 53,3 | 3,44 | 15,5× | 2,12 | 0,029 | 0,021 | 0,348 | 0,108 | 0,030 | 0,026 | 0,790 | 62 % | 40× |
| T10I4D100K | 0,0001 | 715 k | 52,9 | 3,02 | 17,5× | 1,83 | 0,025 | 0,019 | 0,300 | 0,101 | 0,028 | 0,024 | 0,695 | 61 % | 45× |
| T10I4D100K | 0,00005 | 715 k | 50,9 | 2,99 | 17,1× | 1,82 | 0,022 | 0,019 | 0,304 | 0,095 | 0,026 | 0,023 | 0,688 | 61 % | 44× |
| T10I4D100K | 0,00003 | 715 k | 51,2 | 2,17 | 23,6× | 0,96 | 0,027 | 0,019 | 0,308 | 0,096 | 0,028 | 0,025 | 0,711 | 44 % | 42× |
| retail | 0,0002 | 626 k | 49,6 | 3,59 | 13,8× | 2,24 | 0,027 | 0,022 | 0,356 | 0,108 | 0,029 | 0,026 | 0,789 | 63 % | 37× |
| retail | 0,0001 | 653 k | 48,4 | 2,24 | 21,6× | 1,05 | 0,025 | 0,019 | 0,307 | 0,093 | 0,026 | 0,024 | 0,687 | 47 % | 41× |
| retail | 0,00005 | 666 k | 47,0 | 3,14 | 15,0× | 1,96 | 0,025 | 0,019 | 0,309 | 0,095 | 0,025 | 0,023 | 0,697 | 62 % | 40× |
| retail | 0,00003 | 673 k | 48,8 | 3,26 | 15,0× | 2,03 | 0,026 | 0,020 | 0,326 | 0,097 | 0,028 | 0,024 | 0,709 | 62 % | 40× |
| kosarak | 0,002 | 1,99 M | 179 | 7,01 | 25,5× | 4,76 | 0,131 | 0,025 | 0,489 | 0,255 | 0,070 | 0,058 | 1,03 | 68 % | 79× |
| kosarak | 0,001 | 2,90 M | 304 | 8,16 | 37,3× | 5,58 | 0,130 | 0,030 | 0,653 | 0,254 | 0,105 | 0,108 | 1,30 | 68 % | 118× |
| kosarak | 0,0009 | 3,02 M | 318 | 7,12 | 44,7× | 4,51 | 0,129 | 0,024 | 0,652 | 0,247 | 0,109 | 0,115 | 1,31 | 63 % | 122× |
| kosarak | 0,00085 | 3,08 M | 320 | 8,55 | 37,5× | 5,85 | 0,136 | 0,024 | 0,677 | 0,257 | 0,114 | 0,120 | 1,37 | 68 % | 119× |

## Apêndice C — Mineração (todas as configurações)

ms; "P1" = fração do tempo na passada de contagem; "Mais lento / média" =
tempo do bloco mais lento sobre a média.

| Dataset | minsup | Itemsets | Seq. | Estática | Est. vs. seq. | Dinâmica | Din. vs. seq. | Din. vs. est. | P1 est. | P1 din. | Mais lento / média est. | Mais lento / média din. | Na fila |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,5 | 1,27 M | 229 | 880 | 0,3× | 28,0 | 8,2× | 31× | 47 % | 47 % | 5,4 | 1,09 | 4.190 |
| chess | 0,45 | 2,83 M | 478 | 1.711 | 0,3× | 60,2 | 7,9× | 28× | 47 % | 47 % | 5,2 | 1,04 | 8.594 |
| chess | 0,4 | 6,44 M | 1.117 | 2.814 | 0,4× | 128 | 8,8× | 22× | 47 % | 47 % | 4,0 | 1,02 | 17.816 |
| chess | 0,35 | 15,1 M | 2.421 | 6.043 | 0,4× | 277 | 8,8× | 22× | 47 % | 48 % | 4,4 | 1,01 | 38.203 |
| mushroom | 0,05 | 3,76 M | 250 | 248 | 1,0× | 20,4 | 12× | 12× | 47 % | 46 % | 10,7 | 2,78 | 2.533 |
| mushroom | 0,03 | 9,99 M | 770 | 375 | 2,1× | 26,7 | 29× | 14× | 47 % | 46 % | 8,1 | 1,63 | 5.138 |
| mushroom | 0,02 | 22,6 M | 1.661 | 976 | 1,7× | 40,2 | 41× | 24× | 47 % | 46 % | 10,8 | 1,11 | 10.552 |
| mushroom | 0,01 | 90,8 M | 6.424 | 4.747 | 1,4× | 173 | 37× | 27× | 28 % | 34 % | 7,9 | 4,88 | 33.890 |
| accidents | 0,3 | 0,15 M | 145 | 196 | 0,7× | 94,7 | 1,5× | 2,1× | 48 % | 46 % | 3,4 | 10,24 | 2.917 |
| accidents | 0,2 | 0,89 M | 682 | 1.029 | 0,7× | 256 | 2,7× | 4,0× | 47 % | 49 % | 5,6 | 4,46 | 15.754 |
| accidents | 0,15 | 2,68 M | 1.607 | 2.079 | 0,8× | 405 | 4,0× | 5,1× | 47 % | 48 % | 5,3 | 2,95 | 44.497 |
| accidents | 0,1 | 10,7 M | 4.942 | 10.678 | 0,5× | 1.047 | 4,7× | 10× | 47 % | 47 % | 9,1 | 1,41 | 158.955 |
| T10I4D100K | 0,0002 | 0,13 M | 186 | 10,2 | 18× | 9,19 | 20× | 1,1× | 48 % | 47 % | 4,9 | 1,50 | 4.297 |
| T10I4D100K | 0,0001 | 0,41 M | 324 | 28,8 | 11× | 18,8 | 17× | 1,5× | 48 % | 48 % | 6,5 | 1,26 | 9.759 |
| T10I4D100K | 0,00005 | 1,92 M | 748 | 93,3 | 8,0× | 53,3 | 14× | 1,8× | 47 % | 48 % | 6,2 | 1,05 | 24.483 |
| T10I4D100K | 0,00003 | 6,17 M | 1.589 | 292 | 5,4× | 138 | 11× | 2,1× | 48 % | 48 % | 7,3 | 1,03 | 41.098 |
| retail | 0,0002 | 0,07 M | 85,6 | 6,34 | 14× | 5,62 | 15× | 1,1× | 49 % | 49 % | 22,3 | 1,30 | 3.909 |
| retail | 0,0001 | 0,24 M | 157 | 57,5 | 2,7× | 12,5 | 13× | 4,6× | 48 % | 48 % | 208 | 1,31 | 5.690 |
| retail | 0,00005 | 1,51 M | 432 | 767 | 0,6× | 46,2 | 9,3× | 17× | 47 % | 47 % | 1.018 | 1,36 | 9.041 |
| retail | 0,00003 | 20,6 M | 2.887 | 5.572 | 0,5× | 424 | 6,8× | 13× | 47 % | 49 % | 1.644 | 2,34 | 16.078 |
| kosarak | 0,002 | 0,04 M | 386 | 67,7 | 5,7× | 17,1 | 23× | 4,0× | 48 % | 50 % | 16,0 | 1,54 | 1.262 |
| kosarak | 0,001 | 0,76 M | 1.113 | 993 | 1,1× | 40,4 | 28× | 25× | 47 % | 49 % | 132 | 1,01 | 21.462 |
| kosarak | 0,0009 | 1,79 M | 1.763 | 1.793 | 1,0× | 80,0 | 22× | 22× | 47 % | 48 % | 136 | 1,00 | 58.292 |
| kosarak | 0,00085 | 6,46 M | 4.816 | 9.374 | 0,5× | 285 | 17× | 33× | 48 % | 50 % | 221 | 1,00 | 268.824 |
