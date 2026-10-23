# FP-Growth em GPU: relatório detalhado por fase — suportes padrão

Esta é uma cópia de `reports/relatorio_fases.md` refeita com os **suportes mínimos mais usados com estes datasets na literatura FIMI** (os da tabela do plano original), em vez dos suportes reduzidos escolhidos para que o sequencial levasse de 0,17 s a 80 s. As descrições das implementações são as mesmas; todos os números, tabelas e leituras foram recalculados a partir de `results/` por `scripts/relatorio_fases.py`. A comparação com o relatório original mostra como as conclusões dependem do tamanho do problema.

Para cada fase, o relatório mostra:

- como ela é implementada originalmente (versão sequencial em CPU) e quanto tempo leva;
- as ideias que testamos para acelerá-la, na ordem em que foram tentadas;
- o desempenho de cada ideia, os detalhes de implementação e por que cada mecanismo foi escolhido.

No fim, compara a versão final com a original, com o speedup de cada fase e o speedup total.

Todos os números vêm dos CSVs em `results/`, medidos numa única RTX 4090: a do slot PCIe x16, barramento `0000:01:00.0`. Valores em milissegundos são medianas de 5 execuções, depois de 1 de aquecimento descartada.

---

## Sumário

1. Resumo dos resultados
2. Ambiente, dados e método
3. O algoritmo e onde o tempo está na versão original
4. Fase 1 — Leitura
5. Fase 2 — Contagem de suporte
6. Fase 3 — Pré-processamento
7. Transferência CPU → GPU
8. Fase 4 — Construção da FP-tree
9. Fase 5 — Mineração
10. Fase 6 — Retorno e escrita da saída
11. Custo fixo fora das etapas medidas
12. Comparação final: tudo junto
13. O que falta e o ganho que ainda é possível
14. Comparação com o relatório de suportes reduzidos
15. Reprodução

---

## 1. Resumo dos resultados

Na tabela, **speedup = tempo do sequencial / tempo da versão final em GPU** (mineração dinâmica, `grain` 64). Faixa e média geométrica sobre as 24 configurações (6 datasets × 4 suportes).

| Fase | Ideia final | Speedup mín. | Speedup mediano | Speedup máx. | Média geométrica |
| --- | --- | ---: | ---: | ---: | ---: |
| Contagem de suporte | histograma por bloco em memória compartilhada | 0,06× | 0,6× | 68× | 1,0× |
| Pré-processamento | scans + ordenação segmentada | 0,5× | 7,5× | 35× | 5,3× |
| Construção da FP-tree | ordenação + LCP + scan + sparse table | 0,1× | 5,2× | 43× | 2,9× |
| Mineração | fila global dinâmica + blocos persistentes | 0,01× | 2,0× | 22× | 1,3× |
| **Núcleo** (as 4 fases acima) | — | **0,1×** | **4,0×** | **24×** | **2,8×** |
| Núcleo + transferências | — | 0,1× | 2,8× | 16× | 2,2× |
| **Ponta a ponta** (com leitura e escrita) | — | **0,3×** | **1,4×** | **3,9×** | **1,4×** |

Somando as 24 configurações: o núcleo do algoritmo cai de **4,33 s para 0,64 s (6,8×)**, e o tempo total cai de **8,98 s para 5,80 s (1,5×)**.

Leituras principais:

- **Com os suportes padrão, a maior parte das configurações é pequena.** A mineração sequencial leva de 0,054 ms a 1.192 ms e o tempo total sequencial de 4,39 ms a 2.255 ms. Nessa escala, custos fixos da GPU (lançamentos, alocações, duas passadas) pesam tanto quanto o trabalho.
- **Construção:** ganha 2,9×–43× nas árvores com 100 mil nós ou mais (13 configurações) e perde (0,1×–0,7×) nas 8 configurações de chess e mushroom, cujas árvores têm até 22,3 k nós.
- **Mineração:** a dinâmica é mais rápida que a CPU em 16 das 24 configurações. Perde nas 8 em que a mineração sequencial inteira leva no máximo 10,0 ms (chess 0,8, chess 0,7, mushroom 0,3, mushroom 0,2, accidents 0,8, accidents 0,6, retail 0,01, retail 0,005). A fila dinâmica só faz diferença onde há carga grande e desigual: até 24× sobre a estática; nas configurações leves fica empatada ou um pouco atrás (em 14 delas).
- **Custo fixo corrigido:** na primeira rodada, a preparação da mineração (reserva e zeragem de 6–9 GB de arenas e pool) ficava fora do cronômetro de qualquer etapa e custava dezenas de ms por execução. A memória deixou de ser zerada e a preparação passou a ser medida dentro da mineração: agora custa 6,9–13,6 ms (seção 11).
- **Ponta a ponta:** 0,3×–3,9×. A GPU fica mais lenta que a CPU em 5 configurações (chess 0,8, chess 0,7, mushroom 0,3, mushroom 0,2, retail 0,01), todas com tempo sequencial total de no máximo 30 ms.

---

## 2. Ambiente, dados e método

**Hardware.** NVIDIA RTX 4090: 128 SMs, 24 GB e banda de pico de 1008 GB/s; medimos 920 GB/s numa cópia D2D. Ela fica num slot PCIe Gen4 x16 (26,6 GB/s com memória *pinned*, 24,2 GB/s *pageable*). CPU AMD Ryzen 9 7900 (12 núcleos), 60 GB de RAM, CUDA 12.4, g++ 13, `-O3`.

A máquina tem uma segunda RTX 4090 num link x4 (6,4 GB/s). Ela não foi usada. Cada execução registra o barramento PCI, e os scripts falham se alguma tiver rodado em outra placa.

**Datasets.** Seis bases do repositório FIMI, obtidas de um espelho da Wayback Machine porque o site original está fora do ar.

| Dataset | Tipo | Transações | Itens distintos | Tam. médio | Tam. máx. | Suportes avaliados |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| chess | denso, pequeno | 3.196 | 75 | 37,0 | 37 | 0,8 · 0,7 · 0,6 · 0,5 |
| mushroom | denso, pequeno | 8.124 | 119 | 23,0 | 23 | 0,3 · 0,2 · 0,1 · 0,05 |
| accidents | denso, grande | 340.183 | 468 | 33,8 | 51 | 0,8 · 0,6 · 0,4 · 0,3 |
| T10I4D100K | esparso, sintético | 100.000 | 870 | 10,1 | 29 | 0,01 · 0,005 · 0,002 · 0,001 |
| retail | esparso, real | 88.162 | 16.470 | 10,3 | 76 | 0,01 · 0,005 · 0,002 · 0,001 |
| kosarak | esparso, grande | 990.002 | 41.270 | 8,1 | 2.497 | 0,01 · 0,005 · 0,002 · 0,001 |

Os suportes são os mais usados com estes datasets na literatura FIMI (a tabela do plano original). O conjunto com suportes reduzidos está descrito em `reports/relatorio_fases.md` (números anteriores à correção da seção 11); suas medições serão refeitas em `results/low_support/` (TODO 1.1).

**O que cada configuração produz** (maior suporte → menor suporte):

| Dataset | Itens frequentes | Nós da FP-tree | Itemsets frequentes |
| --- | ---: | ---: | ---: |
| chess | 19 → 37 | 720 → 15,7 k | 8,2 k → 1,27 M |
| mushroom | 28 → 73 | 1,9 k → 22,3 k | 2,7 k → 3,76 M |
| accidents | 11 → 32 | 608 → 528,0 k | 149 → 149,5 k |
| T10I4D100K | 375 → 797 | 515,1 k → 711,0 k | 385 → 27,5 k |
| retail | 70 → 2.117 | 31,0 k → 433,4 k | 159 → 7,6 k |
| kosarak | 54 → 1.259 | 294,5 k → 2,90 M | 383 → 759,4 k |

Comparado ao conjunto reduzido, as saídas são de 10× a 1.000× menores (no máximo 3,76 M itemsets) e várias árvores são pequenas: o regime em que custos fixos da GPU importam.

**Corretude.** A saída é canônica (itens em ordem crescente, linhas ordenadas), então comparar duas saídas é um `cmp`.

- O sequencial é idêntico ao `fpgrowth` 6.21 de Christian Borgelt em 12 casos e às saídas calculadas à mão em 5 datasets minúsculos.
- Toda execução do benchmark teve o hash da saída comparado com o do sequencial: 360 execuções principais mais 180 de `grain`, **nenhuma inválida**.
- A suíte `tests` tem 535 verificações; a versão dinâmica rodou 420 vezes seguidas sem travar.

---

## 3. O algoritmo e onde o tempo está na versão original

O FP-Growth tem quatro passos algorítmicos, que aqui ficam entre uma leitura e uma escrita:

```
 leitura ─► contagem ─► pré-processamento ─► construção da FP-tree ─► mineração ─► escrita
  (CPU)       (1)              (2)                    (3)                 (4)        (CPU)
              └──────────────── núcleo do algoritmo (vai para a GPU) ────────┘
```

1. **Contagem:** suporte de cada item.
2. **Pré-processamento:** descarta itens infrequentes, renumera os frequentes por suporte decrescente (o *rank*) e ordena cada transação.
3. **Construção:** insere as transações numa árvore de prefixos com contadores (a FP-tree) e numa tabela de cabeçalho que liga todos os nós de cada item.
4. **Mineração:** para cada item `y`, do menos para o mais frequente, emite `S ∪ {y}`, extrai a *base condicional*, constrói a árvore condicional e repete a mineração nela; árvores que são um caminho único têm todas as combinações emitidas diretamente.

As transações ficam em formato **CSR**: um vetor `items[]` com os ids de todas as transações em sequência e um vetor `offsets[]` de tamanho n+1.

**Tempo da versão original por fase** (sequencial, suporte intermediário, ms e % do total):

| Dataset (minsup) | Leitura | Contagem | Pré-proc. | Construção | Mineração | Escrita | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess (0,6) | 0,897 (1 %) | 0,047 (0 %) | 0,774 (1 %) | 1,02 (1 %) | 47,1 (31 %) | 103 (67 %) | 153 |
| mushroom (0,1) | 1,32 (0 %) | 0,075 (0 %) | 0,971 (0 %) | 1,57 (0 %) | 45,3 (14 %) | 267 (84 %) | 317 |
| accidents (0,4) | 80,7 (26 %) | 4,46 (1 %) | 71,6 (23 %) | 79,8 (26 %) | 58,1 (19 %) | 10,6 (3 %) | 306 |
| T10I4D100K (0,002) | 8,36 (7 %) | 0,415 (0 %) | 8,13 (7 %) | 50,5 (41 %) | 51,7 (42 %) | 3,39 (3 %) | 123 |
| retail (0,002) | 9,65 (17 %) | 0,398 (1 %) | 5,82 (10 %) | 21,5 (38 %) | 18,6 (33 %) | 0,631 (1 %) | 56,9 |
| kosarak (0,002) | 153 (19 %) | 4,34 (1 %) | 55,8 (7 %) | 181 (22 %) | 399 (49 %) | 12,8 (2 %) | 810 |

No conjunto inteiro (apêndice A):

- **Mineração:** 0 %–54 % do total.
- **Construção:** 0 %–49 %; pesa mais nos esparsos (T10I4D100K chega perto de metade do tempo).
- **Leitura:** 0 %–62 %; com saídas pequenas, a leitura passa a ser uma fatia grande do tempo nos datasets grandes (accidents em suporte alto, kosarak).
- **Escrita:** 0 %–89 %; só domina chess e mushroom nos suportes mais baixos.

---

## 4. Fase 1 — Leitura (CPU)

**Implementação original.** O arquivo inteiro é lido de uma vez (`fread`) e convertido em CSR por um único laço que acumula dígitos e fecha uma transação a cada `\n`. Itens repetidos dentro de uma transação são removidos. Código em `src/common/fimi.cpp`.

**Tempo:** 0,909–1,32 ms em chess e mushroom, 8,51–9,61 ms em T10I4D100K e retail, 80 ms em accidents e 153 ms em kosarak.

**Ideias de melhoria:** nenhuma implementada; a leitura é idêntica nas três versões. Com os suportes padrão ela pesa bem mais que no conjunto reduzido: até 62 % do tempo sequencial e uma fração ainda maior do tempo da GPU (seção 12). Opções futuras: parsing paralelo com os 12 núcleos (ganho ideal ~12×) ou parsing na GPU com um scan sobre os separadores.

---

## 5. Fase 2 — Contagem de suporte

### 5.1 Implementação original

Um vetor de contadores do tamanho do maior id e um laço sobre todos os itens: `++sup[x]`. É o melhor caso possível para a CPU: acesso sequencial e o vetor cabe no cache.

**Tempo original:** 0,046 ms (chess) a 4,48 ms (accidents). Independe do suporte e é sempre menos de 1 % do total.

### 5.2 Ideia 1 — `atomicAdd` global (kernel v1)

Uma thread por ocorrência executa `atomicAdd(&support[items[p]], 1)` na memória global. É a tradução direta do laço sequencial; os atômicos são resolvidos no L2, um por endereço de cada vez.

### 5.3 Ideia 2 — Histograma privado por bloco em memória compartilhada (kernel v2)

Cada bloco de 256 threads mantém um histograma de 12.288 contadores (48 KB) em memória compartilhada e, no fim, soma no vetor global um `atomicAdd` por contador não nulo. Se há mais ids que contadores, `gridDim.y` divide os ids em faixas e cada bloco relê a base (retail 2 faixas, kosarak 4). No máximo 4 blocos por SM.

**Desempenho** (kernel isolado, média de 50 lançamentos; independe do suporte):

| Dataset | Ocorrências | Sequencial (ms) | v1 (ms) | v2 (ms) | v2 vs. v1 | v2 vs. seq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 118,3 k | 0,046 | 0,026 | 0,012 | 2,1× | 3,8× |
| mushroom | 186,9 k | 0,075 | 0,044 | 0,012 | 3,6× | 6,1× |
| accidents | 11,50 M | 4,48 | 1,89 | 0,036 | 53× | 124× |
| T10I4D100K | 1,01 M | 0,420 | 0,169 | 0,016 | 11× | 27× |
| retail | 908,6 k | 0,394 | 0,087 | 0,022 | 3,9× | 18× |
| kosarak | 8,02 M | 4,35 | 1,08 | 0,173 | 6,2× | 25× |

O maior ganho está onde a contenção era o problema (accidents). Estes números são os mesmos do relatório original, porque a contagem não depende do suporte.

### 5.4 Resultado da fase dentro do pipeline

Dentro do programa, a fase de contagem na GPU mede 0,020–0,990 ms, muito mais que o kernel, e o speedup da fase vai de 0,06× a 68× (média geométrica 1,0×). O custo extra é fixo (alocação do vetor e, provavelmente, carregamento preguiçoso do módulo CUDA no primeiro lançamento; **não medido isoladamente**). A fase vale menos de 1 % do tempo.

---

## 6. Fase 3 — Pré-processamento

### 6.1 Implementação original

Tabela de ranks (suporte decrescente, id crescente), filtragem e troca de id por rank, `std::sort` por transação e descarte das transações vazias.

### 6.2 Ideia — tudo como primitivas paralelas

`copy_if` + `thrust::sort` para os ranks; um kernel com uma thread por ocorrência para filtrar e remapear; `exclusive_scan` das flags; `cub::DeviceSegmentedReduce::Sum` para os novos tamanhos; *scatter*; `copy_if` + `inclusive_scan` para descartar as vazias; `cub::DeviceSegmentedSort::SortKeys` para ordenar cada transação. O trabalho é dividido por ocorrência, não por transação.

**Desempenho** (maior / menor suporte):

| Dataset | minsup | Sequencial (ms) | GPU (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| chess | 0,8 / 0,5 | 0,283 / 0,810 | 0,592 / 0,600 | 0,5× / 1,4× |
| mushroom | 0,3 / 0,05 | 0,664 / 0,958 | 0,696 / 0,749 | 1,0× / 1,3× |
| accidents | 0,8 / 0,3 | 30,8 / 74,7 | 1,96 / 2,16 | 16× / 35× |
| T10I4D100K | 0,01 / 0,001 | 7,81 / 8,50 | 0,994 / 0,979 | 7,9× / 8,7× |
| retail | 0,01 / 0,001 | 3,20 / 7,33 | 0,914 / 1,03 | 3,5× / 7,1× |
| kosarak | 0,01 / 0,001 | 34,0 / 64,2 | 2,35 / 2,76 | 14× / 23× |

Faixa: 0,5×–35×. Com suporte alto sobram poucos itens frequentes e a versão sequencial fica barata, então a GPU (dominada por ~10 chamadas de biblioteca com alocação e lançamento) ganha menos ou empata. O tráfego obrigatório em kosarak é de ~0,2 ms; a GPU leva ~2,5 ms.

---

## 7. Transferência CPU → GPU (H2D)

Dois `cudaMemcpy` (items e offsets) a partir de memória *pageable*.

| Dataset | Bytes | Tempo (ms) | Banda efetiva |
| --- | ---: | ---: | ---: |
| chess | 0,5 MB | 0,819 | 0,6 GB/s |
| mushroom | 0,8 MB | 0,847 | 0,9 GB/s |
| accidents | 47,4 MB | 3,14 | 15,1 GB/s |
| T10I4D100K | 4,4 MB | 1,10 | 4,0 GB/s |
| retail | 4,0 MB | 1,26 | 3,2 GB/s |
| kosarak | 36,0 MB | 3,06 | 11,8 GB/s |

Nos datasets pequenos o tempo é dominado pela alocação dos vetores na GPU, não pela cópia. Melhoria possível, não implementada: memória *pinned* e cópia sobreposta à contagem.

---

## 8. Fase 4 — Construção da FP-tree

### 8.1 Implementação original

FP-tree clássica com nós em vetores e índices no lugar de ponteiros; busca de filho por tabela hash (pai, item); lista ligada por item na tabela de cabeçalho.

### 8.2–8.4 Ideias implementadas

1. **Nós como prefixos distintos:** ordenação lexicográfica das transações (`thrust::sort` com comparador), LCP com a transação anterior, `exclusive_scan` de `len − LCP` para os ids dos nós (em pré-ordem), escrita de item, profundidade e transação criadora.
2. **Pais e contadores sem atômicos:** *previous smaller value* e *next smaller or equal* sobre o LCP, respondidos por busca binária numa sparse table de mínimos.
3. **Tabela de cabeçalho por ordenação:** `stable_sort_by_key` por item, `lower_bound` e `reduce_by_key`.

(Detalhes, alternativas descartadas e o exemplo de Han, Pei e Yin: seção 8 de `relatorio_fases.md`.)

### 8.5 Desempenho da construção

Mediana, ms (maior e menor suporte de cada dataset; todas as configurações no apêndice B):

| Dataset | minsup | Nós | Seq. | GPU | Speedup | sort | LCP | scan | nós | sparse | pais | cont. | cabeç. | % sort | Speedup sem o sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,8 | 720 | 0,290 | 2,79 | 0,1× | 2,43 | 0,022 | 0,017 | 0,041 | 0,036 | 0,015 | 0,015 | 0,200 | 87 % | 0,8× |
| chess | 0,5 | 15,7 k | 1,08 | 3,47 | 0,3× | 3,04 | 0,025 | 0,017 | 0,045 | 0,036 | 0,015 | 0,021 | 0,267 | 88 % | 2,5× |
| mushroom | 0,3 | 1,9 k | 0,683 | 3,12 | 0,2× | 2,75 | 0,025 | 0,019 | 0,042 | 0,037 | 0,016 | 0,015 | 0,207 | 88 % | 1,9× |
| mushroom | 0,05 | 22,3 k | 2,35 | 3,47 | 0,7× | 3,04 | 0,022 | 0,018 | 0,049 | 0,038 | 0,017 | 0,015 | 0,272 | 88 % | 5,4× |
| accidents | 0,8 | 608 | 13,1 | 5,28 | 2,5× | 4,73 | 0,115 | 0,022 | 0,048 | 0,125 | 0,017 | 0,015 | 0,210 | 89 % | 23× |
| accidents | 0,3 | 528,0 k | 98,7 | 8,89 | 11× | 7,53 | 0,160 | 0,023 | 0,284 | 0,137 | 0,026 | 0,022 | 0,701 | 85 % | 72× |
| T10I4D100K | 0,01 | 515,1 k | 28,6 | 1,99 | 14× | 0,887 | 0,022 | 0,019 | 0,285 | 0,119 | 0,023 | 0,021 | 0,619 | 45 % | 26× |
| T10I4D100K | 0,001 | 711,0 k | 51,6 | 3,08 | 17× | 1,87 | 0,024 | 0,019 | 0,308 | 0,101 | 0,026 | 0,023 | 0,697 | 61 % | 43× |
| retail | 0,01 | 31,0 k | 2,41 | 1,35 | 1,8× | 0,857 | 0,021 | 0,018 | 0,043 | 0,101 | 0,016 | 0,014 | 0,267 | 64 % | 4,9× |
| retail | 0,001 | 433,4 k | 25,2 | 3,16 | 7,9× | 2,01 | 0,024 | 0,020 | 0,290 | 0,109 | 0,022 | 0,020 | 0,631 | 63 % | 22× |
| kosarak | 0,01 | 294,5 k | 34,8 | 6,10 | 5,7× | 4,84 | 0,126 | 0,024 | 0,224 | 0,238 | 0,023 | 0,019 | 0,613 | 79 % | 28× |
| kosarak | 0,001 | 2,90 M | 310 | 7,13 | 43× | 4,54 | 0,131 | 0,024 | 0,654 | 0,246 | 0,107 | 0,109 | 1,31 | 64 % | 119× |

**Leitura.**

- **Árvores com 100 mil nós ou mais** (13 configurações): 2,9×–43× mais rápida que a inserção sequencial, crescendo com o tamanho da árvore (kosarak vai de 5,7× com 294,5 k nós a 43× com 2,90 M).
- **Árvores pequenas** (abaixo de 100 mil nós, 11 configurações): 0,1×–5,3×. A GPU perde em 8 delas (chess 0,8, chess 0,7, chess 0,6, chess 0,5, mushroom 0,3, mushroom 0,2, mushroom 0,1, mushroom 0,05): a construção sequencial leva 0,290–2,35 ms, menos que o custo fixo da ordenação na GPU (~2–3 ms).
- **A ordenação lexicográfica domina**: 45 %–90 % do tempo da fase. Sem ela, o resto seria 0,8×–119× mais rápido que o sequencial.
- **Pais e contadores** custam no máximo 0,109 ms mesmo com 2,90 M nós.

**Ganho ainda possível com uma ordenação melhor.** Com a ordenação no tempo de um radix sort de 64 bits do mesmo número de chaves (medido), a construção chegaria a até 102× sobre o sequencial (kosarak 0,001), e a GPU passaria a ganhar em 7 das 8 configurações em que hoje perde; a exceção é chess 0,8 (de 0,1× para 0,7×).

---

## 9. Fase 5 — Mineração

### 9.1 Implementação original

Recursão clássica com atalho de caminho único; as árvores condicionais mantêm a ordem global de ranks. **Tempo original:** 0,054 ms a 1.192 ms.

### 9.2 Ideia 1 — Mineração estática: um bloco de threads por item

Um bloco de 256 threads por item frequente resolve toda a recursão do item com pilha explícita, arena de duas pontas, construção condicional em bloco (as mesmas funções da construção global), merge sort de bloco, histograma em shared, atalho de caminho único analítico, saída em duas passadas e tratamento de estouro por projeção na GPU. (Tabela de mecanismos e motivos: seção 9.2 de `relatorio_fases.md`.)

**Desempenho:**

| Dataset | minsup | Itemsets | Seq. (ms) | Estática (ms) | Estática vs. seq. | Bloco mais lento / média |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chess | 0,8 | 8,2 k | 1,74 | 15,0 | **0,1×** | 4,4 |
| chess | 0,5 | 1,27 M | 230 | 881 | **0,3×** | 5,4 |
| mushroom | 0,3 | 2,7 k | 0,408 | 5,18 | **0,08×** | 5,2 |
| mushroom | 0,05 | 3,76 M | 248 | 259 | **1,0×** | 10,7 |
| accidents | 0,8 | 149 | 0,054 | 3,87 | **0,01×** | 2,3 |
| accidents | 0,3 | 149,5 k | 145 | 186 | **0,8×** | 3,4 |
| T10I4D100K | 0,01 | 385 | 22,3 | 5,65 | 3,9× | 2,4 |
| T10I4D100K | 0,001 | 27,5 k | 67,4 | 9,59 | 7,0× | 7,4 |
| retail | 0,01 | 159 | 0,775 | 3,91 | **0,2×** | 2,2 |
| retail | 0,001 | 7,6 k | 32,5 | 6,95 | 4,7× | 6,0 |
| kosarak | 0,01 | 383 | 18,8 | 7,33 | 2,6× | 2,7 |
| kosarak | 0,001 | 759,4 k | 1.192 | 1.001 | 1,2× | 131,9 |

A estática é **mais lenta que a CPU em 12 das 24 configurações** (chess 0,8, chess 0,7, chess 0,6, chess 0,5, mushroom 0,3, mushroom 0,2, mushroom 0,05, accidents 0,8, accidents 0,6, accidents 0,3, retail 0,01, retail 0,005), todas de datasets densos, onde há poucos itens (poucos blocos) e, nos suportes altos, pouquíssimo trabalho. Nos esparsos ela ganha de 0,2× a 11×, porque há centenas de itens com carga parecida; o pior caso é kosarak 0,001, em que um único item domina o kernel (razão bloco mais lento / média de 132).

Distribuição do tempo por bloco (suporte intermediário):

| Dataset | minsup | Blocos (itens) | Bloco mais lento (ms) | Média (ms) | Eficiência* |
| --- | --- | ---: | ---: | ---: | ---: |
| chess | 0,6 | 34 | 134 | 22,5 | 2 % |
| mushroom | 0,1 | 56 | 22,0 | 3,54 | 4 % |
| accidents | 0,4 | 30 | 19,9 | 9,47 | 6 % |
| T10I4D100K | 0,002 | 741 | 2,07 | 0,207 | 29 % |
| retail | 0,002 | 956 | 0,168 | 0,052 | 100 % |
| kosarak | 0,002 | 568 | 37,9 | 2,38 | 14 % |

\* trabalho total dividido por 256 blocos residentes, sobre o tempo do bloco mais lento.

### 9.3 Ideia 2 — Mineração dinâmica: fila global de tarefas com blocos persistentes

256 blocos persistentes consomem uma fila global MPMC sem travas (número de sequência por posição); árvores condicionais com base ≥ `grain` são copiadas para um pool global e publicadas como tarefas; terminação por contador de pendentes; fila ou pool cheio → processamento local. (Tabela de mecanismos: seção 9.3 de `relatorio_fases.md`.)

**Desempenho** (`grain` = 64):

| Dataset | minsup | Seq. (ms) | Estática (ms) | Dinâmica (ms) | Din. vs. seq. | Din. vs. est. | Mais lento / média: est. → din. | Tarefas na fila |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| chess | 0,8 | 1,74 | 15,0 | 9,26 | 0,2× | 1,6× | 4,4 → 9,50 | 50 |
| chess | 0,5 | 230 | 881 | 36,5 | 6,3× | 24× | 5,4 → 1,09 | 4.190 |
| mushroom | 0,3 | 0,408 | 5,18 | 8,37 | 0,05× | 0,6× | 5,2 → 27,55 | 23 |
| mushroom | 0,05 | 248 | 259 | 32,0 | 7,7× | 8,1× | 10,7 → 2,78 | 2.533 |
| accidents | 0,8 | 0,054 | 3,87 | 7,53 | 0,01× | 0,5× | 2,3 → 51,04 | 5 |
| accidents | 0,3 | 145 | 186 | 90,0 | 1,6× | 2,1× | 3,4 → 10,30 | 2.917 |
| T10I4D100K | 0,01 | 22,3 | 5,65 | 9,40 | 2,4× | 0,6× | 2,4 → 1,64 | 1 |
| T10I4D100K | 0,001 | 67,4 | 9,59 | 13,3 | 5,1× | 0,7× | 7,4 → 1,41 | 1.054 |
| retail | 0,01 | 0,775 | 3,91 | 7,69 | 0,1× | 0,5× | 2,2 → 8,18 | 17 |
| retail | 0,001 | 32,5 | 6,95 | 11,7 | 2,8× | 0,6× | 6,0 → 1,63 | 1.096 |
| kosarak | 0,01 | 18,8 | 7,33 | 10,8 | 1,7× | 0,7× | 2,7 → 12,17 | 39 |
| kosarak | 0,001 | 1.192 | 1.001 | 53,9 | 22× | 19× | 131,9 → 1,01 | 21.462 |

Resumo: a dinâmica é mais rápida que a CPU em 16 das 24 configurações (0,01×–22×, média geométrica 1,3×) e 0,5×–24× em relação à estática.

**Por regime:**

- **Configurações minúsculas** (chess 0,8, chess 0,7, mushroom 0,3, mushroom 0,2, accidents 0,8, accidents 0,6, retail 0,01, retail 0,005): a mineração sequencial leva 0,054–10,0 ms e a dinâmica 7,53–13,0 ms: são duas passadas, cada uma com lançamento, sincronização e cópia de contadores, e árvores condicionais construídas por um bloco com dezenas de barreiras.
- **Carga leve** (esparsos em suporte alto, accidents): dinâmica e estática ficam próximas; a dinâmica perde por pouco em 14 configurações (mushroom 0,3, accidents 0,8, accidents 0,6, accidents 0,4, T10I4D100K 0,01, T10I4D100K 0,005, T10I4D100K 0,002, T10I4D100K 0,001, retail 0,01, retail 0,005, retail 0,002, retail 0,001, kosarak 0,01, kosarak 0,005), por 0,5×–0,9×. Sem tarefas grandes, a fila só acrescenta o custo de cópia e de sincronização.
- **Carga grande e desigual** (chess ≤ 0,6, mushroom 0,05, kosarak ≤ 0,002): é onde a fila compensa — chess 0,5 24×, kosarak 0,001 19× sobre a estática.
- **Razão máx/média da dinâmica alta nos suportes altos** (até 51): não indica desbalanceamento, e sim que há menos tarefas do que os 256 blocos; a maioria fica ociosa e a métrica perde o sentido.

**Eficiência de balanceamento** (suporte intermediário):

| Dataset | minsup | Estática | Dinâmica |
| --- | --- | ---: | ---: |
| chess | 0,6 | 2 % | 81 % |
| mushroom | 0,1 | 4 % | 11 % |
| accidents | 0,4 | 6 % | 6 % |
| T10I4D100K | 0,002 | 29 % | 61 % |
| retail | 0,002 | 100 % | 61 % |
| kosarak | 0,002 | 14 % | 68 % |

### 9.4 Ideia 3 — Ajuste da granularidade (`grain`)

Mineração dinâmica em ms, suporte intermediário; o melhor em negrito:

| Dataset | minsup | Estática | g = 0 | g = 16 | g = 64 | g = 256 | g = 1024 | g = 4096 | Melhor vs. estática | 64 vs. melhor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,6 | 238 | 19,2 | **14,0** | 15,4 | 52,4 | 246 | 251 | 17× | 1,1× |
| mushroom | 0,1 | 42,6 | **10,3** | 10,3 | 20,8 | 28,0 | 46,7 | 46,8 | 4,1× | 2,0× |
| accidents | 0,4 | 40,7 | **44,8** | 44,8 | 46,6 | 46,4 | 45,4 | 45,0 | 0,9× | 1,0× |
| T10I4D100K | 0,002 | 9,37 | 12,0 | **11,7** | 12,2 | 14,6 | 14,6 | 14,3 | 0,8× | 1,0× |
| retail | 0,002 | 6,11 | **10,5** | 10,9 | 11,1 | 10,6 | 11,0 | 11,0 | 0,6× | 1,0× |
| kosarak | 0,002 | 72,9 | **26,5** | 26,9 | 27,4 | 26,7 | 52,8 | 78,7 | 2,7× | 1,0× |

O melhor grain fica entre 0 e 16. O padrão 64 fica a até 2,0× do melhor (mushroom 0,1; a causa não foi investigada); nos outros datasets fica a no máximo 1,1×. Grains ≥ 1024 fazem a dinâmica voltar ao tempo da estática onde há carga desigual (chess, mushroom, kosarak).

### 9.5 Limites e o que ainda falta na mineração

- **Duas passadas:** a passada 1 é 1 %–47 % do tempo de mineração. Nas configurações minúsculas o custo fixo de cada passada é o que impede a GPU de ganhar; uma passada única com buffers otimistas daria até 2×.
- **Tarefa indivisível** (accidents): as primeiras árvores condicionais são construídas por um bloco só.
- **Preparação:** reservar e liberar 6–9 GB de arenas e pool ainda custa 6,9–13,6 ms por execução (seção 11); dimensionar pela necessidade reduziria esse custo nas configurações pequenas.

---

## 10. Fase 6 — Retorno (D2H) e escrita da saída

Escrita idêntica nas três versões (tradução rank → id, ordenação dos itens, formatação e ordenação das linhas, em uma thread). D2H + montagem do CSR de itemsets no host, medidos juntos.

| Dataset | minsup | Itemsets | D2H + montagem, GPU (ms) | Escrita, seq. (ms) | Escrita, GPU (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| chess | 0,5 | 1,27 M | 55,0 | 652 | 757 |
| mushroom | 0,05 | 3,76 M | 154 | 2.011 | 2.017 |
| accidents | 0,3 | 149,5 k | 5,13 | 55,6 | 58,5 |
| T10I4D100K | 0,001 | 27,5 k | 0,443 | 7,16 | 7,86 |
| retail | 0,001 | 7,6 k | 0,083 | 1,72 | 1,85 |
| kosarak | 0,001 | 759,4 k | 28,4 | 460 | 448 |

Com os suportes padrão a saída é pequena (até 3,76 M itemsets), e a escrita só domina chess e mushroom nos menores suportes (mushroom 0,05: 2.011 de 2.255 ms no sequencial).

---

## 11. Custo fixo fora das etapas medidas (corrigido)

Na primeira rodada com os suportes padrão apareceu um custo que o relatório original não mostrava: na GPU, a soma das etapas medidas era bem menor que o tempo total.

**De onde vinha.** A preparação da mineração reservava as arenas dos blocos (até 35 % da memória livre, 6 GB) e, na dinâmica, o pool (até 15 %, 3 GB) com `thrust::device_vector::resize`, que **zera** toda a memória, e esse trecho ficava fora do cronômetro da etapa "mineração", que media só os kernels. A memória não precisa ser zerada: a arena é sempre escrita antes de ser lida.

**Correção.** Arenas e pool passaram a ser reservados com `cudaMalloc` sem inicialização, e a reserva e a liberação entram na etapa de mineração, como a subetapa `mineracao_preparacao`. Todos os números deste relatório já são da versão corrigida.

| Dataset | minsup | Total dinâmica antes (ms) | Fora das etapas antes (ms) | Total dinâmica depois (ms) | Preparação, agora medida (ms) | Fora das etapas depois (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chess | 0,8 | 28,7 | 18,7 | 18,7 | 7,33 | 1,32 |
| chess | 0,5 | 863 | 22,0 | 859 | 8,72 | 3,70 |
| mushroom | 0,3 | 26,7 | 18,7 | 16,5 | 7,46 | 0,418 |
| mushroom | 0,05 | 2.250 | 34,0 | 2.220 | 11,7 | 10,0 |
| accidents | 0,8 | 116 | 21,4 | 105 | 7,27 | 3,58 |
| accidents | 0,3 | 266 | 22,6 | 257 | 7,23 | 6,43 |
| T10I4D100K | 0,01 | 35,2 | 20,7 | 25,3 | 8,85 | 1,78 |
| T10I4D100K | 0,001 | 47,6 | 22,1 | 37,3 | 10,7 | 2,04 |
| retail | 0,01 | 33,2 | 18,3 | 23,0 | 7,56 | 1,42 |
| retail | 0,001 | 40,5 | 22,1 | 30,9 | 10,2 | 2,42 |
| kosarak | 0,01 | 192 | 21,6 | 182 | 7,01 | 4,53 |
| kosarak | 0,001 | 708 | 31,8 | 709 | 13,6 | 9,00 |

Antes, 18–34 ms por execução da dinâmica ficavam fora de qualquer etapa. Agora a preparação custa 6,9–13,6 ms (só a reserva e a liberação da memória) e está dentro da mineração; o que sobra fora das etapas é 0,4–10,0 ms. O tempo total da dinâmica caiu até 30 ms por execução (mediana 10 ms; nas execuções longas a diferença fica dentro do ruído), e o número de configurações em que a GPU perde ponta a ponta foi de 7 para 5.

---

## 12. Comparação final: tudo junto

**Versão final:** leitura (CPU) → H2D → contagem v2 → pré-processamento por primitivas → construção por ordenação + LCP + sparse table → mineração dinâmica (`grain` 64) → D2H → escrita (CPU).

### 12.1 Speedup de cada fase e total, por configuração

Speedup = tempo do sequencial / tempo da versão final. Núcleo = contagem + pré-processamento + construção + mineração. "Amdahl" é o teto do speedup ponta a ponta se o núcleo levasse tempo zero: `T_seq / (T_leitura + T_escrita)`.

| Dataset | minsup | Contagem | Pré-proc. | Construção | Mineração | Núcleo seq. → GPU (ms) | **Núcleo** | Núcleo + transf. | Total seq. → GPU (ms) | **Total** | Amdahl |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| chess | 0,8 | 2,3× | 0,5× | 0,1× | 0,2× | 2,36 → 12,7 | **0,2×** | 0,2× | 6,38 → 18,7 | **0,3×** | 1,6× |
| chess | 0,7 | 0,06× | 0,6× | 0,1× | 1,0× | 10,9 → 15,1 | **0,7×** | 0,6× | 29,6 → 36,5 | **0,8×** | 1,6× |
| chess | 0,6 | 0,06× | 1,3× | 0,3× | 3,1× | 49,0 → 20,1 | **2,4×** | 1,6× | 153 → 142 | **1,1×** | 1,5× |
| chess | 0,5 | 0,06× | 1,4× | 0,3× | 6,3× | 232 → 41,3 | **5,6×** | 2,4× | 885 → 859 | **1,0×** | 1,4× |
| mushroom | 0,3 | 0,10× | 1,0× | 0,2× | 0,05× | 1,83 → 13,0 | **0,1×** | 0,1× | 4,39 → 16,5 | **0,3×** | 1,7× |
| mushroom | 0,2 | 0,10× | 1,4× | 0,4× | 0,3× | 6,47 → 17,7 | **0,4×** | 0,3× | 29,5 → 41,9 | **0,7×** | 1,3× |
| mushroom | 0,1 | 0,10× | 1,3× | 0,5× | 2,2× | 47,9 → 25,7 | **1,9×** | 1,0× | 317 → 312 | **1,0×** | 1,2× |
| mushroom | 0,05 | 0,1× | 1,3× | 0,7× | 7,7× | 251 → 37,0 | **6,8×** | 1,3× | 2.255 → 2.220 | **1,0×** | 1,1× |
| accidents | 0,8 | 6,6× | 16× | 2,5× | 0,01× | 48,5 → 15,4 | **3,1×** | 2,6× | 128 → 105 | **1,2×** | 1,6× |
| accidents | 0,6 | 68× | 25× | 4,6× | 0,3× | 93,4 → 21,9 | **4,3×** | 3,7× | 177 → 114 | **1,6×** | 2,2× |
| accidents | 0,4 | 6,1× | 33× | 9,2× | 1,3× | 214 → 56,4 | **3,8×** | 3,5× | 306 → 159 | **1,9×** | 3,4× |
| accidents | 0,3 | 6,4× | 35× | 11× | 1,6× | 323 → 102 | **3,2×** | 2,9× | 460 → 257 | **1,8×** | 3,4× |
| T10I4D100K | 0,01 | 13× | 7,9× | 9,8× | 2,4× | 59,1 → 13,3 | **4,4×** | 4,1× | 68,0 → 25,3 | **2,7×** | 7,7× |
| T10I4D100K | 0,005 | 0,6× | 8,5× | 22× | 3,2× | 91,7 → 14,5 | **6,3×** | 5,8× | 100 → 26,7 | **3,7×** | 11× |
| T10I4D100K | 0,002 | 0,6× | 8,2× | 17× | 4,3× | 111 → 16,8 | **6,6×** | 6,1× | 123 → 32,0 | **3,9×** | 10× |
| T10I4D100K | 0,001 | 11× | 8,7× | 17× | 5,1× | 128 → 17,4 | **7,4×** | 6,8× | 146 → 37,3 | **3,9×** | 9,3× |
| retail | 0,01 | 0,6× | 3,5× | 1,8× | 0,1× | 6,78 → 10,7 | **0,6×** | 0,6× | 17,1 → 23,0 | **0,7×** | 1,7× |
| retail | 0,005 | 0,5× | 4,4× | 4,7× | 0,5× | 15,9 → 11,6 | **1,4×** | 1,2× | 25,6 → 24,5 | **1,0×** | 2,6× |
| retail | 0,002 | 0,6× | 5,8× | 7,1× | 1,7× | 46,3 → 15,8 | **2,9×** | 2,7× | 56,9 → 28,9 | **2,0×** | 5,5× |
| retail | 0,001 | 0,5× | 7,1× | 12× | 2,8× | 65,4 → 15,5 | **4,2×** | 3,9× | 76,4 → 30,9 | **2,5×** | 6,9× |
| kosarak | 0,01 | 4,4× | 14× | 5,6× | 1,7× | 92,0 → 20,3 | **4,5×** | 4,0× | 243 → 182 | **1,3×** | 1,6× |
| kosarak | 0,005 | 4,4× | 17× | 12× | 7,4× | 220 → 22,9 | **9,6×** | 8,5× | 373 → 186 | **2,0×** | 2,4× |
| kosarak | 0,002 | 4,5× | 22× | 27× | 15× | 640 → 37,2 | **17×** | 15× | 810 → 215 | **3,8×** | 4,9× |
| kosarak | 0,001 | 4,6× | 23× | 43× | 22× | 1.570 → 64,8 | **24×** | 16× | 2.193 → 709 | **3,1×** | 3,5× |

### 12.2 Agregado

| Medida | Mín. | Mediana | Máx. | Média geométrica |
| --- | ---: | ---: | ---: | ---: |
| Contagem | 0,06× | 0,6× | 68× | 1,0× |
| Pré-processamento | 0,5× | 7,5× | 35× | 5,3× |
| Construção | 0,1× | 5,2× | 43× | 2,9× |
| Mineração | 0,01× | 2,0× | 22× | 1,3× |
| **Núcleo** | 0,1× | 4,0× | 24× | 2,8× |
| Núcleo + transferências | 0,1× | 2,8× | 16× | 2,2× |
| **Total ponta a ponta** | 0,3× | 1,4× | 3,9× | 1,4× |
| Teto de Amdahl (núcleo = 0) | 1,1× | 2,3× | 11× | — |

Soma das 24 configurações:

| Versão | Núcleo (s) | Total (s) |
| --- | ---: | ---: |
| Sequencial | 4,327 | 8,983 |
| GPU estática | 3,037 | 8,160 |
| **GPU dinâmica (final)** | 0,639 | 5,802 |
| Speedup final | **6,8×** | **1,5×** |

Do total somado da GPU dinâmica, 0,08 s (1 %) ficam fora das etapas medidas.

### 12.3 Onde o tempo está na versão final (suporte intermediário)

| Dataset | Leitura | H2D | Núcleo (GPU) | D2H + montagem | Escrita | Fora das etapas | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess 0,6 | 1 % | 1 % | 14 % | 7 % | 77 % | 0 % | 142 ms |
| mushroom 0,1 | 0 % | 0 % | 8 % | 7 % | 83 % | 1 % | 312 ms |
| accidents 0,4 | 52 % | 2 % | 35 % | 0 % | 7 % | 3 % | 159 ms |
| T10I4D100K 0,002 | 27 % | 4 % | 53 % | 0 % | 12 % | 4 % | 32,0 ms |
| retail 0,002 | 34 % | 4 % | 55 % | 0 % | 3 % | 5 % | 28,9 ms |
| kosarak 0,002 | 72 % | 2 % | 17 % | 0 % | 6 % | 2 % | 215 ms |

O resto do tempo, fora do núcleo do algoritmo, é leitura e escrita na CPU e transferências.

---

## 13. O que falta e o ganho que ainda é possível

Em ordem de impacto esperado no tempo total, **com os suportes padrão**:

| # | Melhoria | Fase | Ganho esperado |
| --- | --- | --- | --- |
| 1 | Dimensionar arenas e pool pela necessidade (hoje 6–9 GB fixos) | mineração | a preparação ainda custa 6,9–13,6 ms por execução, só em `cudaMalloc`/`cudaFree` |
| 2 | Passada única na mineração | mineração | até 2× na mineração; decisivo nas configurações minúsculas |
| 3 | Parsing paralelo ou na GPU | leitura | leitura é a maior fase restante em kosarak e accidents |
| 4 | Ordenação lexicográfica por radix híbrido + alocador com cache | construção | elimina a derrota nas árvores pequenas |
| 5 | Escrita paralela ou na GPU | escrita | decisiva só em chess e mushroom nos menores suportes |
| 6 | Memória *pinned* + sobreposição | H2D | ~1–3 ms |

---

## 14. Comparação com o relatório de suportes reduzidos

**Pendente:** as medições do conjunto reduzido com a versão corrigida ainda não foram refeitas (TODO 1.1). Os números de `reports/relatorio_fases.md` são anteriores à correção da seção 11.

A diferença entre os dois relatórios é o tamanho do problema. Com suportes reduzidos, quase todas as configurações têm trabalho de sobra e a GPU vence em praticamente tudo. Com os suportes padrão, várias configurações são pequenas demais: a GPU ainda vence nos esparsos e onde há muitos itemsets, mas perde onde o problema inteiro leva poucos milissegundos, por causa de custos fixos (lançamentos, alocações, duas passadas). Um deles, a preparação da mineração, ficava fora das etapas medidas até a correção da seção 11; a correção vale para os dois conjuntos.

---

## 15. Reprodução

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./scripts/download_data.sh
FPG_SET=common PY=/caminho/python3 ./scripts/run_all.sh
FPG_SET=common python3 scripts/relatorio_fases.py reports/relatorio_fases_suporte_padrao.md
```

| Arquivo | Conteúdo |
| --- | --- |
| `results/raw.csv` | tempo de cada etapa de cada execução (suportes padrão) |
| `results/runs.csv` | itemsets, nós, estouros, tarefas, desbalanceamento, memória e barramento PCI |
| `results/grain.csv` | experimento de granularidade |
| `results/blocks/*.csv` | tempo, tarefas e itemsets de cada bloco |
| `results/micro/` | limites de hardware, kernels de contagem isolados, estatísticas por configuração |
| `results/low_support/` | o mesmo para os suportes reduzidos (`FPG_SET=low`; a refazer, TODO 1.1) |
| `results/pre_fix/` | medições anteriores à correção da preparação da mineração |

---

## Apêndice A — Tempo sequencial por fase (todas as configurações)

ms (% do total).

| Dataset | minsup | Leitura | Contagem | Pré-proc. | Construção | Mineração | Escrita | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,8 | 0,906 (14 %) | 0,046 (1 %) | 0,283 (4 %) | 0,290 (5 %) | 1,74 (27 %) | 3,06 (48 %) | 6,38 |
| chess | 0,7 | 0,912 (3 %) | 0,046 (0 %) | 0,377 (1 %) | 0,460 (2 %) | 10,0 (34 %) | 17,6 (60 %) | 29,6 |
| chess | 0,6 | 0,897 (1 %) | 0,047 (0 %) | 0,774 (1 %) | 1,02 (1 %) | 47,1 (31 %) | 103 (67 %) | 153 |
| chess | 0,5 | 1,02 (0 %) | 0,046 (0 %) | 0,810 (0 %) | 1,08 (0 %) | 230 (26 %) | 652 (74 %) | 885 |
| mushroom | 0,3 | 1,31 (30 %) | 0,075 (2 %) | 0,664 (15 %) | 0,683 (16 %) | 0,408 (9 %) | 1,20 (27 %) | 4,39 |
| mushroom | 0,2 | 1,20 (4 %) | 0,075 (0 %) | 0,895 (3 %) | 1,18 (4 %) | 4,33 (15 %) | 21,9 (74 %) | 29,5 |
| mushroom | 0,1 | 1,32 (0 %) | 0,075 (0 %) | 0,971 (0 %) | 1,57 (0 %) | 45,3 (14 %) | 267 (84 %) | 317 |
| mushroom | 0,05 | 1,33 (0 %) | 0,074 (0 %) | 0,958 (0 %) | 2,35 (0 %) | 248 (11 %) | 2.011 (89 %) | 2.255 |
| accidents | 0,8 | 78,6 (61 %) | 4,48 (3 %) | 30,8 (24 %) | 13,1 (10 %) | 0,054 (0 %) | 0,420 (0 %) | 128 |
| accidents | 0,6 | 80,2 (45 %) | 4,47 (3 %) | 52,8 (30 %) | 33,0 (19 %) | 3,15 (2 %) | 0,966 (1 %) | 177 |
| accidents | 0,4 | 80,7 (26 %) | 4,46 (1 %) | 71,6 (23 %) | 79,8 (26 %) | 58,1 (19 %) | 10,6 (3 %) | 306 |
| accidents | 0,3 | 81,3 (18 %) | 4,48 (1 %) | 74,7 (16 %) | 98,7 (21 %) | 145 (32 %) | 55,6 (12 %) | 460 |
| T10I4D100K | 0,01 | 8,61 (13 %) | 0,418 (1 %) | 7,81 (11 %) | 28,6 (42 %) | 22,3 (33 %) | 0,184 (0 %) | 68,0 |
| T10I4D100K | 0,005 | 8,55 (9 %) | 0,421 (0 %) | 8,37 (8 %) | 49,0 (49 %) | 34,0 (34 %) | 0,324 (0 %) | 100 |
| T10I4D100K | 0,002 | 8,36 (7 %) | 0,415 (0 %) | 8,13 (7 %) | 50,5 (41 %) | 51,7 (42 %) | 3,39 (3 %) | 123 |
| T10I4D100K | 0,001 | 8,47 (6 %) | 0,422 (0 %) | 8,50 (6 %) | 51,6 (35 %) | 67,4 (46 %) | 7,16 (5 %) | 146 |
| retail | 0,01 | 9,66 (57 %) | 0,396 (2 %) | 3,20 (19 %) | 2,41 (14 %) | 0,775 (5 %) | 0,367 (2 %) | 17,1 |
| retail | 0,005 | 9,57 (37 %) | 0,392 (2 %) | 4,18 (16 %) | 7,21 (28 %) | 4,12 (16 %) | 0,184 (1 %) | 25,6 |
| retail | 0,002 | 9,65 (17 %) | 0,398 (1 %) | 5,82 (10 %) | 21,5 (38 %) | 18,6 (33 %) | 0,631 (1 %) | 56,9 |
| retail | 0,001 | 9,37 (12 %) | 0,390 (1 %) | 7,33 (10 %) | 25,2 (33 %) | 32,5 (43 %) | 1,72 (2 %) | 76,4 |
| kosarak | 0,01 | 151 (62 %) | 4,36 (2 %) | 34,0 (14 %) | 34,8 (14 %) | 18,8 (8 %) | 0,194 (0 %) | 243 |
| kosarak | 0,005 | 153 (41 %) | 4,31 (1 %) | 41,9 (11 %) | 77,9 (21 %) | 95,8 (26 %) | 0,482 (0 %) | 373 |
| kosarak | 0,002 | 153 (19 %) | 4,34 (1 %) | 55,8 (7 %) | 181 (22 %) | 399 (49 %) | 12,8 (2 %) | 810 |
| kosarak | 0,001 | 158 (7 %) | 4,38 (0 %) | 64,2 (3 %) | 310 (14 %) | 1.192 (54 %) | 460 (21 %) | 2.193 |

## Apêndice B — Construção (todas as configurações)

| Dataset | minsup | Nós | Seq. | GPU | Speedup | sort | LCP | scan | nós | sparse | pais | cont. | cabeç. | % sort | Speedup sem o sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,8 | 720 | 0,290 | 2,79 | 0,1× | 2,43 | 0,022 | 0,017 | 0,041 | 0,036 | 0,015 | 0,015 | 0,200 | 87 % | 0,8× |
| chess | 0,7 | 2,2 k | 0,460 | 3,17 | 0,1× | 2,80 | 0,024 | 0,017 | 0,042 | 0,038 | 0,016 | 0,021 | 0,205 | 88 % | 1,2× |
| chess | 0,6 | 10,8 k | 1,02 | 2,45 | 0,4× | 2,02 | 0,025 | 0,017 | 0,044 | 0,036 | 0,015 | 0,015 | 0,278 | 82 % | 2,4× |
| chess | 0,5 | 15,7 k | 1,08 | 3,47 | 0,3× | 3,04 | 0,025 | 0,017 | 0,045 | 0,036 | 0,015 | 0,021 | 0,267 | 88 % | 2,5× |
| mushroom | 0,3 | 1,9 k | 0,683 | 3,12 | 0,2× | 2,75 | 0,025 | 0,019 | 0,042 | 0,037 | 0,016 | 0,015 | 0,207 | 88 % | 1,9× |
| mushroom | 0,2 | 6,5 k | 1,18 | 3,40 | 0,3× | 2,98 | 0,022 | 0,018 | 0,045 | 0,038 | 0,015 | 0,015 | 0,266 | 88 % | 2,8× |
| mushroom | 0,1 | 14,3 k | 1,57 | 2,43 | 0,6× | 2,00 | 0,024 | 0,018 | 0,043 | 0,036 | 0,018 | 0,015 | 0,274 | 82 % | 3,7× |
| mushroom | 0,05 | 22,3 k | 2,35 | 3,47 | 0,7× | 3,04 | 0,022 | 0,018 | 0,049 | 0,038 | 0,017 | 0,015 | 0,272 | 88 % | 5,4× |
| accidents | 0,8 | 608 | 13,1 | 5,28 | 2,5× | 4,73 | 0,115 | 0,022 | 0,048 | 0,125 | 0,017 | 0,015 | 0,210 | 89 % | 23× |
| accidents | 0,6 | 41,6 k | 33,0 | 6,21 | 5,3× | 5,56 | 0,123 | 0,022 | 0,052 | 0,126 | 0,017 | 0,017 | 0,285 | 90 % | 51× |
| accidents | 0,4 | 409,7 k | 79,8 | 8,66 | 9,2× | 7,40 | 0,150 | 0,022 | 0,277 | 0,139 | 0,025 | 0,021 | 0,630 | 85 % | 63× |
| accidents | 0,3 | 528,0 k | 98,7 | 8,89 | 11× | 7,53 | 0,160 | 0,023 | 0,284 | 0,137 | 0,026 | 0,022 | 0,701 | 85 % | 72× |
| T10I4D100K | 0,01 | 515,1 k | 28,6 | 1,99 | 14× | 0,887 | 0,022 | 0,019 | 0,285 | 0,119 | 0,023 | 0,021 | 0,619 | 45 % | 26× |
| T10I4D100K | 0,005 | 646,1 k | 49,0 | 2,99 | 16× | 1,80 | 0,024 | 0,019 | 0,309 | 0,098 | 0,025 | 0,022 | 0,700 | 60 % | 41× |
| T10I4D100K | 0,002 | 703,4 k | 50,5 | 3,01 | 17× | 1,82 | 0,031 | 0,019 | 0,311 | 0,097 | 0,025 | 0,023 | 0,690 | 61 % | 42× |
| T10I4D100K | 0,001 | 711,0 k | 51,6 | 3,08 | 17× | 1,87 | 0,024 | 0,019 | 0,308 | 0,101 | 0,026 | 0,023 | 0,697 | 61 % | 43× |
| retail | 0,01 | 31,0 k | 2,41 | 1,35 | 1,8× | 0,857 | 0,021 | 0,018 | 0,043 | 0,101 | 0,016 | 0,014 | 0,267 | 64 % | 4,9× |
| retail | 0,005 | 106,3 k | 7,21 | 2,47 | 2,9× | 1,89 | 0,023 | 0,019 | 0,097 | 0,092 | 0,017 | 0,015 | 0,322 | 76 % | 12× |
| retail | 0,002 | 292,5 k | 21,5 | 3,07 | 7,0× | 1,98 | 0,023 | 0,019 | 0,264 | 0,121 | 0,021 | 0,019 | 0,625 | 64 % | 20× |
| retail | 0,001 | 433,4 k | 25,2 | 3,16 | 7,9× | 2,01 | 0,024 | 0,020 | 0,290 | 0,109 | 0,022 | 0,020 | 0,631 | 63 % | 22× |
| kosarak | 0,01 | 294,5 k | 34,8 | 6,10 | 5,7× | 4,84 | 0,126 | 0,024 | 0,224 | 0,238 | 0,023 | 0,019 | 0,613 | 79 % | 28× |
| kosarak | 0,005 | 857,7 k | 77,9 | 6,56 | 12× | 5,03 | 0,130 | 0,024 | 0,320 | 0,252 | 0,039 | 0,032 | 0,751 | 77 % | 51× |
| kosarak | 0,002 | 1,99 M | 181 | 7,46 | 24× | 5,42 | 0,129 | 0,024 | 0,493 | 0,244 | 0,068 | 0,058 | 1,02 | 73 % | 88× |
| kosarak | 0,001 | 2,90 M | 310 | 7,13 | 43× | 4,54 | 0,131 | 0,024 | 0,654 | 0,246 | 0,107 | 0,109 | 1,31 | 64 % | 119× |

## Apêndice C — Mineração (todas as configurações)

ms; "P1" = fração do tempo na passada de contagem.

| Dataset | minsup | Itemsets | Seq. | Estática | Est. vs. seq. | Dinâmica | Din. vs. seq. | Din. vs. est. | P1 est. | P1 din. | Mais lento / média est. | Mais lento / média din. | Na fila |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess | 0,8 | 8,2 k | 1,74 | 15,0 | 0,1× | 9,26 | 0,2× | 1,6× | 36 % | 10 % | 4,4 | 9,50 | 50 |
| chess | 0,7 | 48,7 k | 10,0 | 51,6 | 0,2× | 10,5 | 1,0× | 4,9× | 44 % | 14 % | 4,2 | 2,62 | 243 |
| chess | 0,6 | 254,9 k | 47,1 | 238 | 0,2× | 15,3 | 3,1× | 16× | 46 % | 23 % | 6,0 | 1,33 | 1.043 |
| chess | 0,5 | 1,27 M | 230 | 881 | 0,3× | 36,5 | 6,3× | 24× | 46 % | 36 % | 5,4 | 1,09 | 4.190 |
| mushroom | 0,3 | 2,7 k | 0,408 | 5,18 | 0,08× | 8,37 | 0,05× | 0,6× | 13 % | 5 % | 5,2 | 27,55 | 23 |
| mushroom | 0,2 | 53,6 k | 4,33 | 14,1 | 0,3× | 13,0 | 0,3× | 1,1× | 34 % | 21 % | 9,3 | 30,77 | 71 |
| mushroom | 0,1 | 574,4 k | 45,3 | 42,6 | 1,1× | 20,8 | 2,2× | 2,1× | 41 % | 27 % | 6,2 | 8,75 | 473 |
| mushroom | 0,05 | 3,76 M | 248 | 259 | 1,0× | 32,0 | 7,7× | 8,1× | 45 % | 30 % | 10,7 | 2,78 | 2.533 |
| accidents | 0,8 | 149 | 0,054 | 3,87 | 0,01× | 7,53 | 0,01× | 0,5× | 3 % | 2 % | 2,3 | 51,04 | 5 |
| accidents | 0,6 | 2,1 k | 3,15 | 8,69 | 0,4× | 12,5 | 0,3× | 0,7× | 29 % | 20 % | 3,5 | 40,30 | 52 |
| accidents | 0,4 | 32,5 k | 58,1 | 40,7 | 1,4× | 44,8 | 1,3× | 0,9× | 45 % | 43 % | 2,1 | 16,73 | 685 |
| accidents | 0,3 | 149,5 k | 145 | 186 | 0,8× | 90,0 | 1,6× | 2,1× | 47 % | 45 % | 3,4 | 10,30 | 2.917 |
| T10I4D100K | 0,01 | 385 | 22,3 | 5,65 | 3,9× | 9,40 | 2,4× | 0,6× | 4 % | 3 % | 2,4 | 1,64 | 1 |
| T10I4D100K | 0,005 | 1,1 k | 34,0 | 6,47 | 5,3× | 10,6 | 3,2× | 0,6× | 6 % | 4 % | 3,0 | 1,41 | 94 |
| T10I4D100K | 0,002 | 13,3 k | 51,7 | 9,37 | 5,5× | 12,1 | 4,3× | 0,8× | 19 % | 8 % | 10,0 | 1,61 | 641 |
| T10I4D100K | 0,001 | 27,5 k | 67,4 | 9,59 | 7,0× | 13,3 | 5,1× | 0,7× | 20 % | 9 % | 7,4 | 1,41 | 1.054 |
| retail | 0,01 | 159 | 0,775 | 3,91 | 0,2× | 7,69 | 0,1× | 0,5× | 1 % | 1 % | 2,2 | 8,18 | 17 |
| retail | 0,005 | 580 | 4,12 | 4,68 | 0,9× | 8,37 | 0,5× | 0,6× | 2 % | 1 % | 1,8 | 2,76 | 86 |
| retail | 0,002 | 2,7 k | 18,6 | 6,11 | 3,0× | 11,0 | 1,7× | 0,6× | 5 % | 3 % | 3,5 | 1,63 | 403 |
| retail | 0,001 | 7,6 k | 32,5 | 6,95 | 4,7× | 11,7 | 2,8× | 0,6× | 8 % | 6 % | 6,0 | 1,63 | 1.096 |
| kosarak | 0,01 | 383 | 18,8 | 7,33 | 2,6× | 10,8 | 1,7× | 0,7× | 24 % | 16 % | 2,7 | 12,17 | 39 |
| kosarak | 0,005 | 1,6 k | 95,8 | 8,66 | 11× | 12,9 | 7,4× | 0,7× | 25 % | 20 % | 1,9 | 3,29 | 126 |
| kosarak | 0,002 | 39,5 k | 399 | 72,9 | 5,5× | 26,9 | 15× | 2,7× | 44 % | 31 % | 15,8 | 1,67 | 1.262 |
| kosarak | 0,001 | 759,4 k | 1.192 | 1.001 | 1,2× | 53,9 | 22× | 19× | 47 % | 36 % | 131,9 | 1,01 | 21.462 |
