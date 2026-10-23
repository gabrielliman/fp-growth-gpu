# Plano de execução: FP-Growth Paralelo em GPU

Oct 20, 2026 · @Gabriel

## Contexto e objetivo

O objetivo é implementar o FP-Growth em GPU (CUDA), com foco na construção da FP-tree por primitivas paralelas (sort, scan, reduce), e compará-lo com uma versão sequencial em CPU. A mineração também vai para a GPU em duas variantes: escalonamento estático e dinâmico.

O FP-Growth tem quatro etapas: (1) contagem de suporte dos itens; (2) filtragem e ordenação de cada transação por frequência decrescente; (3) construção da FP-tree, uma árvore de prefixos com contadores; (4) mineração recursiva via bases de padrões condicionais e FP-trees condicionais.

A ideia central da proposta: os nós da FP-tree correspondem exatamente aos prefixos distintos das transações já ordenadas. Ordenando as transações lexicograficamente e calculando o maior prefixo comum (LCP) entre vizinhas, cada posição além do LCP é um nó novo. A árvore fica linearizada em vetores (item, contador, pai), sem ponteiros.

**Entregáveis finais**

1. `fpg_seq`: FP-Growth sequencial em C++ (baseline e oráculo de corretude).
2. `fpg_gpu`: executável CUDA com as etapas 1 a 3 na GPU e duas variantes de mineração, selecionadas por flag (`--mine=static` e `--mine=dynamic`).
3. Suíte de testes que compara a saída da GPU com a do sequencial.
4. Scripts de benchmark que geram CSVs com tempo por etapa.
5. Scripts de análise que geram os gráficos do relatório.
6. `README.md` explicando como compilar, rodar e reproduzir os experimentos.

## Regras para o agente

Execute as fases em ordem. Não comece uma fase antes de os critérios de aceite da anterior passarem. Ao final de cada fase, faça commit e registre em `NOTES.md` o que foi feito, as decisões tomadas e os problemas em aberto.

**Stack**

- C++17 para o código de CPU e CUDA 12 para o de GPU.
- CMake para o build, com alvos `fpg_seq`, `fpg_gpu` e `tests`.
- Thrust e CUB, que vêm com o CUDA Toolkit, para sort, scan, reduce e segmented sort. Não use outras bibliotecas de GPU.
- Python 3 com pandas e matplotlib para benchmarks e gráficos.

**Estrutura do repositório**

```
fp-growth-gpu/
  CMakeLists.txt
  README.md
  NOTES.md
  data/            # datasets (fora do git; baixados por script)
  scripts/         # download, benchmark, plots
  src/common/      # leitura de dados, CSR, formato de saída, timers
  src/seq/         # FP-Growth sequencial
  src/gpu/         # kernels e host code CUDA
  tests/           # testes de corretude
  results/         # CSVs e gráficos
```

**Convenções obrigatórias**

- Entrada no formato FIMI: uma transação por linha, itens inteiros separados por espaço.
- Suporte mínimo passado como fração (`--minsup 0.01`), convertido para contagem absoluta com `ceil(minsup * n_transacoes)`.
- Saída canônica: um itemset por linha, itens em ordem crescente de id original, seguidos de `  (suporte) `. O arquivo inteiro é ordenado lexicograficamente. Assim, a validação é um `diff` simples.
- Toda chamada CUDA e todo kernel são verificados com uma macro `CUDA_CHECK`.
- Tempos medidos com `cudaEvent` na GPU e `std::chrono` na CPU, separados por etapa: leitura, transferência H2D, contagem, pré-processamento, construção, mineração, transferência D2H, escrita.
- Cada execução imprime uma linha CSV: `impl,dataset,minsup,etapa,tempo_ms`.

**Critério geral de corretude**

Para todo par (dataset, minsup) testado, a saída da GPU deve ser idêntica à do sequencial. Uma diferença é sempre bug; nunca ajuste o teste para passar.

## Fase 0: ambiente e datasets

Objetivo: um repositório que compila, com dados e leitor de entrada funcionando.

- [ ] 0.1 Criar a estrutura de pastas e o `CMakeLists.txt` com os três alvos. Detectar a arquitetura da GPU (`CMAKE_CUDA_ARCHITECTURES native`) e compilar com `-O3`.
- [ ] 0.2 Registrar em `NOTES.md` a GPU, a CPU, a versão do CUDA e a do compilador.
- [ ] 0.3 Escrever `scripts/download_data.sh` para baixar os datasets do repositório FIMI (fimi.uantwerpen.be/data): `chess`, `mushroom`, `retail`, `T10I4D100K`, `kosarak` e `accidents`.
- [ ] 0.4 Criar três datasets minúsculos à mão em `tests/data/` (5 a 10 transações), incluindo o exemplo do artigo de Han, Pei e Yin (2000), com a saída esperada calculada à mão.
- [ ] 0.5 Implementar em `src/common/` o leitor FIMI que produz CSR: `items[]` (int32) e `offsets[]` (tamanho n+1).
- [ ] 0.6 Implementar o escritor de saída canônica e a macro `CUDA_CHECK`.
- [ ] 0.7 Registrar em `NOTES.md` as estatísticas de cada dataset: número de transações, número de itens distintos, tamanho médio e máximo das transações.

**Critérios de aceite**

- `cmake --build` gera os três alvos sem warnings.
- O leitor reproduz as estatísticas publicadas dos datasets FIMI (por exemplo, `chess` tem 3.196 transações de 37 itens).

## Fase 1: baseline sequencial

Objetivo: um FP-Growth sequencial correto, que serve de oráculo para todos os testes e de referência de tempo (T\_seq).

- [ ] 1.1 Contagem de suporte: um vetor de contadores indexado pelo id do item.
- [ ] 1.2 Remapeamento: ordenar os itens frequentes por suporte decrescente (desempate pelo id original, crescente) e atribuir rank 0, 1, 2… Guardar a tabela rank → id original para a saída. Essa regra de desempate deve ser a mesma na GPU.
- [ ] 1.3 Pré-processamento: para cada transação, descartar itens infrequentes, trocar id por rank e ordenar por rank crescente.
- [ ] 1.4 Construção clássica da FP-tree com nós por ponteiro (filhos em `std::unordered_map` ou lista), contadores e tabela de cabeçalho com lista ligada por item.
- [ ] 1.5 Mineração recursiva clássica: para cada item do cabeçalho, do menos para o mais frequente, extrair a base condicional, construir a árvore condicional e recursar. Incluir o atalho de caminho único (gerar todas as combinações diretamente).
- [ ] 1.6 Escrever a saída canônica.
- [ ] 1.7 Instrumentar com timers por etapa e imprimir CSV.
- [ ] 1.8 Validar contra os datasets minúsculos (saída à mão) e contra uma implementação de referência (SPMF ou `fpgrowth` de Christian Borgelt) em `chess` e `mushroom` com 3 valores de suporte.
- [ ] 1.9 Exportar também a FP-tree global em formato textual canônico (um nó por linha: caminho de ranks da raiz até o nó, contador), ordenado. Ele será usado para validar a Fase 3.

**Critérios de aceite**

- Saída idêntica à referência em todos os casos testados.
- Tabela em `NOTES.md` com o tempo por etapa para cada dataset em dois suportes. Ela mostra onde o tempo está concentrado.

## Fase 2: contagem e pré-processamento na GPU

Objetivo: transações filtradas, remapeadas e ordenadas em CSR na memória da GPU, idênticas às do sequencial.

- [ ] 2.1 Transferir `items[]` e `offsets[]` para a GPU.
- [ ] 2.2 Kernel de contagem v1 (referência): `atomicAdd` global por item.
- [ ] 2.3 Kernel de contagem v2: histograma privado por bloco em memória compartilhada, seguido de soma global com `atomicAdd` por bloco. Se o número de itens distintos não couber na memória compartilhada, usar a v1 ou particionar os itens em faixas.
- [ ] 2.4 Gerar a tabela de ranks na GPU: ordenar pares (suporte, id) com `thrust::sort` usando a mesma regra de desempate da tarefa 1.2.
- [ ] 2.5 Kernel de filtragem e remapeamento: cada thread processa um item, marca se é frequente e o troca pelo rank.
- [ ] 2.6 Compactação: `thrust::exclusive_scan` sobre as flags para obter as posições, gerar `offsets` novos por transação (redução segmentada das flags + scan) e escrever o CSR filtrado. Descartar transações vazias.
- [ ] 2.7 Ordenar cada transação por rank com `cub::DeviceSegmentedSort::SortKeys`.
- [ ] 2.8 Copiar o CSR resultante para a CPU num modo de debug e comparar com o pré-processamento do sequencial.
- [ ] 2.9 Medir contagem v1 e v2 separadamente em `retail` e `kosarak`, que têm muitos itens, e em `chess`, que tem poucos itens muito populares e, portanto, mais contenção.

**Critérios de aceite**

- Suportes e CSR pré-processado idênticos ao sequencial em todos os datasets.
- Comparação v1 × v2 registrada em `NOTES.md`.

## Fase 3: construção da FP-tree por primitivas paralelas

Objetivo: construir a FP-tree inteira na GPU sem inserções concorrentes, armazenada como vetores `node_item[]`, `node_count[]`, `node_parent[]` e `node_depth[]`, mais uma tabela de cabeçalho. É a contribuição central do trabalho.

Notação: depois da Fase 2 há n transações ordenadas internamente. T\_i\[d\] é o item na profundidade d (0 = mais perto da raiz) e len\_i o tamanho. A raiz é implícita (pai = -1).

- [ ] 3.1 **Ordenação lexicográfica.** Ordenar os índices 0…n-1 das transações com `thrust::sort` e um comparador que compara os segmentos do CSR item a item (prefixo menor vem antes). Reescrever o CSR na nova ordem. Opcional depois: versão LSD por coluna, com `stable_sort_by_key` da última profundidade para a primeira e sentinela para transações curtas.
- [ ] 3.2 **LCP.** Kernel com uma thread por transação: LCP\[0\] = 0 e LCP\[i\] = tamanho do maior prefixo comum entre T\_{i-1} e T\_i.
- [ ] 3.3 **Nós novos e índices.** new\_i = len\_i - LCP\[i\]. node\_base = `exclusive_scan`(new). O total de nós é N = soma de new. O nó novo da transação i na profundidade d (d ≥ LCP\[i\]) recebe o id node\_base\[i\] + (d - LCP\[i\]).
- [ ] 3.4 **Item e profundidade.** Kernel por transação: para cada d ≥ LCP\[i\], escrever `node_item` = T\_i\[d\] e `node_depth` = d.
- [ ] 3.5 **Pais.** Para d > LCP\[i\], o pai é o nó anterior da mesma transação: node\_base\[i\] + (d - 1 - LCP\[i\]). Para d = LCP\[i\] > 0, o pai é um nó criado antes: seja k o maior índice menor que i com LCP\[k\] < LCP\[i\] (previous smaller value). O pai é node\_base\[k\] + (LCP\[i\] - 1 - LCP\[k\]). Para d = LCP\[i\] = 0, o pai é a raiz. Implementar o previous smaller value com uma sparse table de mínimos sobre LCP (O(n log n), construída em log n passos paralelos) e busca binária por thread.
- [ ] 3.6 **Contadores sem atômicos.** O nó criado em (k, d) é compartilhado por todas as transações seguintes cujo LCP seja maior que d. Seja m o menor índice maior que k com LCP\[m\] ≤ d (ou n, se não existir). Então node\_count = m - k. Usar a mesma sparse table e busca binária (next smaller or equal). Transações duplicadas são tratadas automaticamente porque têm LCP = len.
- [ ] 3.7 **Contadores, versão alternativa para validação.** Calcular para cada posição (i, d) o id do nó que ela percorre e somar com `reduce_by_key`. Manter como verificação cruzada da 3.6 no modo debug.
- [ ] 3.8 **Tabela de cabeçalho.** `stable_sort_by_key` dos ids de nó pela chave `node_item`, gerando `header_nodes[]`. Contar nós por item com histograma e fazer `exclusive_scan` para obter `header_offsets[]`. Guardar também o suporte total de cada item.
- [ ] 3.9 **Validação.** No modo debug, copiar os vetores para a CPU, reconstruir o caminho de cada nó subindo pelos pais e gerar o mesmo formato textual da tarefa 1.9. O resultado deve ser idêntico ao da árvore sequencial, inclusive no número de nós.
- [ ] 3.10 **Medição.** Tempo de cada subetapa (sort, LCP, scan, pais, contadores, cabeçalho) e da construção inteira, comparados com a construção sequencial. Registrar também o número de nós e a memória usada.

**Critérios de aceite**

- Árvore idêntica à sequencial em todos os datasets e em pelo menos 3 suportes por dataset.
- 3.6 e 3.7 produzem os mesmos contadores.
- Tabela de tempos por subetapa em `NOTES.md`, indicando qual primitiva domina o custo (a expectativa é que seja a ordenação lexicográfica).

**Nota para o agente:** os casos de borda que mais quebram são transações duplicadas, uma transação que é prefixo da seguinte, transações de tamanho 1 e um dataset em que todas as transações são iguais. Crie testes para cada um.

## Fase 4: mineração v1 (estática)

Objetivo: minerar todos os itemsets na GPU com um bloco de threads por item da tabela de cabeçalho global. Cada bloco resolve sozinho toda a recursão do seu item. Essa versão expõe o desbalanceamento de carga.

**Decisões de projeto**

- Manter a ordem global de ranks em todas as árvores condicionais (não reordenar por frequência condicional). O resultado continua correto, e os caminhos extraídos já saem ordenados, sem sort por caminho.
- Uma tarefa é o par (sufixo S, árvore linearizada). A tarefa inicial de cada bloco é (item x, FP-tree global restrita aos nós de x).
- A recursão vira uma pilha explícita por bloco, em memória global.
- A memória de trabalho vem de um pool global com alocador por bump atômico. Cada bloco reserva arenas do pool.

**Tarefas**

- [ ] 4.1 Generalizar a rotina da Fase 3 para transações com peso: o contador do nó (k, d) passa a ser W\[m\] - W\[k\], com W = `exclusive_scan` dos pesos. Revalidar a Fase 3 com todos os pesos iguais a 1.
- [ ] 4.2 **Extração da base condicional (por bloco).** Para o item y, as threads dividem os nós de y na tabela de cabeçalho. O caminho de um nó v tem tamanho `node_depth[v]`, então um scan de bloco das profundidades dá os offsets. Cada thread sobe pelos pais escrevendo os itens de trás para frente, já em ordem crescente de rank. O peso do caminho é `node_count[v]`.
- [ ] 4.3 **Itens frequentes condicionais.** Histograma ponderado por bloco sobre a base, seguido de filtro dos itens com suporte ≥ minsup e compactação dos caminhos (removendo vazios).
- [ ] 4.4 **Construção da árvore condicional (versão de bloco).** Reimplementar a Fase 3 em nível de bloco: ordenação lexicográfica dos caminhos com `cub::BlockMergeSort` sobre índices (ou merge sort cooperativo em memória global para bases grandes), LCP, scans com `cub::BlockScan`, pais, contadores e cabeçalho. Para pais e contadores, uma varredura com pilha feita por uma única thread é aceitável nesta versão, desde que validada; registrar em `NOTES.md` se essa parte vira gargalo.
- [ ] 4.5 **Caminho único.** Se a árvore condicional tem exatamente um nó por profundidade, gerar diretamente todas as combinações não vazias dos itens do caminho, cada uma com suporte igual ao contador do nó mais profundo escolhido. Não recursar.
- [ ] 4.6 **Laço da pilha.** Desempilhar tarefa; para cada item frequente y do cabeçalho: emitir S ∪ {y}, extrair a base, construir a árvore condicional e empilhar (S ∪ {y}, árvore) se não for vazia nem caminho único.
- [ ] 4.7 **Saída em duas passadas.** Passada 1: executar a mineração apenas contando itemsets e o total de itens emitidos, com contadores atômicos globais. Alocar os buffers exatos. Passada 2: executar de novo escrevendo, com reserva de posição por `atomicAdd`. Formato: `out_items[]`, `out_offsets[]`, `out_support[]`. A ordenação canônica é feita na CPU antes de escrever o arquivo.
- [ ] 4.8 **Estouro de memória.** Se a arena de um bloco acabar, o bloco grava a tarefa numa lista de estouro e segue. O host processa essa lista depois, com a rotina de dispositivo inteiro da Fase 3, ou, em último caso, com o código sequencial. Registrar quantas tarefas estouram.
- [ ] 4.9 **Instrumentação.** Gravar por bloco: tempo (`clock64`), número de tarefas processadas e número de itemsets emitidos. Esses dados são o insumo da análise de desbalanceamento.
- [ ] 4.10 Validar a saída completa contra o sequencial em todos os datasets e suportes.

**Critérios de aceite**

- Saída idêntica à do sequencial em todos os casos.
- Histograma do tempo por bloco salvo em `results/`, mostrando a distribuição de carga.

## Fase 5: mineração v2 (dinâmica)

Objetivo: substituir a posse fixa de item por bloco por uma fila global de tarefas consumida por blocos persistentes. Um bloco que termina pega a próxima tarefa disponível, em qualquer nível da recursão. Todo o resto (extração, construção, caminho único, saída) reaproveita o código da Fase 4.

- [ ] 5.1 **Fila global.** Vetor circular de descritores de tarefa em memória global (sufixo, ponteiro para a árvore no pool, tamanho), com índices `head` e `tail` atômicos. Capacidade configurável.
- [ ] 5.2 **Blocos persistentes.** Lançar um número de blocos igual ao máximo que cabe simultaneamente na GPU (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × número de SMs). Cada bloco repete: a thread 0 tenta desenfileirar, faz `__syncthreads`, e o bloco processa a tarefa.
- [ ] 5.3 **Inserção de filhos.** Cada árvore condicional gerada vira uma tarefa enfileirada, em vez de ir para uma pilha local.
- [ ] 5.4 **Granularidade.** Se a base condicional de um filho for menor que um limiar (parâmetro `--grain`, por exemplo 256 itens), o próprio bloco processa o filho em profundidade, sem enfileirar. Isso evita sobrecarga da fila com tarefas minúsculas. Testar ao menos 3 valores de limiar.
- [ ] 5.5 **Terminação.** Manter um contador global de tarefas pendentes: incrementado ao enfileirar, decrementado ao concluir. Os blocos encerram quando a fila está vazia e o contador chega a zero. Usar `__threadfence` antes de publicar uma tarefa.
- [ ] 5.6 **Fila cheia.** Se a fila estiver cheia, o bloco processa o filho localmente em profundidade (degrada para o comportamento da v1, sem perder corretude).
- [ ] 5.7 Reaproveitar a saída em duas passadas, a lista de estouro e a instrumentação por bloco da Fase 4.
- [ ] 5.8 Validar contra o sequencial em todos os datasets e suportes.

**Critérios de aceite**

- Saída idêntica à do sequencial.
- Nenhum deadlock em 20 execuções repetidas de cada configuração.
- Histograma do tempo por bloco comparável ao da Fase 4.

## Fase 6: experimentos e análise

Objetivo: responder duas perguntas com dados. (1) Quanto a construção da FP-tree por primitivas acelera em relação à construção sequencial? (2) O escalonamento dinâmico melhora a mineração em relação ao estático, e em quais condições?

**Configurações**

| Dataset | Tipo | Suportes mínimos sugeridos |
| --- | --- | --- |
| chess | denso, poucos itens | 0,8 · 0,7 · 0,6 · 0,5 |
| mushroom | denso | 0,3 · 0,2 · 0,1 · 0,05 |
| accidents | denso, grande | 0,8 · 0,6 · 0,4 · 0,3 |
| T10I4D100K | esparso sintético | 0,01 · 0,005 · 0,002 · 0,001 |
| retail | esparso real | 0,01 · 0,005 · 0,002 · 0,001 |
| kosarak | esparso, grande | 0,01 · 0,005 · 0,002 · 0,001 |

Os suportes são ponto de partida. Ajustar para que o sequencial rode entre cerca de 1 s e 10 min em cada dataset e registrar os valores finais.

**Tarefas**

- [ ] 6.1 `scripts/bench.py`: para cada dataset × suporte × implementação (`seq`, `gpu --mine=static`, `gpu --mine=dynamic`), fazer 1 execução de aquecimento e 5 medidas. Gravar todas em `results/raw.csv`.
- [ ] 6.2 Verificar a saída de cada execução contra o sequencial dentro do script. Uma execução com saída diferente invalida a medida.
- [ ] 6.3 Experimento de construção: tempo de construção GPU (com e sem a ordenação lexicográfica) contra construção sequencial, por dataset, e decomposição por subetapa.
- [ ] 6.4 Experimento de mineração: estática × dinâmica por suporte. Para a dinâmica, variar `--grain`.
- [ ] 6.5 Experimento de balanceamento: distribuição do tempo por bloco nas duas versões. Métrica: razão entre o tempo do bloco mais lento e o tempo médio.
- [ ] 6.6 Tempo total ponta a ponta, com e sem transferências CPU↔GPU.
- [ ] 6.7 Memória: pico de uso da GPU por configuração e número de tarefas em estouro.
- [ ] 6.8 `scripts/plots.py` gera os gráficos: (a) speedup de construção por dataset; (b) tempo total × suporte, uma linha por implementação, eixo y logarítmico; (c) barras empilhadas do tempo por etapa; (d) distribuição do tempo por bloco, estática × dinâmica; (e) efeito do `--grain`.
- [ ] 6.9 Escrever `results/RESULTS.md` com os gráficos, a mediana e o desvio padrão de cada medida, e um parágrafo de interpretação por experimento.

**Critérios de aceite**

- Todas as medidas válidas, reproduzíveis por um único comando (`scripts/run_all.sh`).
- Cada gráfico com eixos rotulados e unidades.

## Contingências e checklist final

As Fases 0 a 3 são o núcleo obrigatório. As Fases 4 e 5 são extensões e podem ser simplificadas sem comprometer a contribuição principal.

| Problema | Plano B |
| --- | --- |
| Ordenação lexicográfica com comparador é lenta demais | Implementar a versão LSD por coluna (tarefa 3.1) ou ordenar por hash de prefixo e resolver colisões |
| Sparse table usa memória demais em datasets grandes | Calcular previous/next smaller value por blocos com pilha local e passo de correção entre blocos |
| Construção em nível de bloco (4.4) fica complexa demais | Construir cada árvore condicional com a rotina de dispositivo inteiro da Fase 3, lançada do host, uma tarefa por vez |
| Mineração na GPU não funciona a tempo | Minerar o primeiro nível na GPU (extração das bases) e o restante na CPU; documentar como limitação |
| Fila dinâmica com deadlock | Trocar por escalonamento dinâmico só no primeiro nível: um contador atômico distribui itens da tabela de cabeçalho, sem filhos na fila |
| Saída grande demais para a memória da GPU | Escrever a saída em lotes, transferindo para a CPU quando o buffer encher |

**Checklist de entrega**

- [ ] `fpg_seq` e `fpg_gpu` compilam a partir de um clone limpo com os comandos do `README.md`.
- [ ] `tests` passa, incluindo os casos de borda da Fase 3.
- [ ] `scripts/run_all.sh` reproduz todos os resultados.
- [ ] `results/RESULTS.md` contém gráficos e interpretação.
- [ ] `NOTES.md` registra o hardware, as decisões e as limitações conhecidas.
