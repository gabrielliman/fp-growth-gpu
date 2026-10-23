# TODO

Lista do que falta, em ordem de prioridade. Cada item diz o estado atual, o que
fazer e como verificar. Os comandos estão detalhados em [TUTORIAL.md](TUTORIAL.md).

Legenda: **[exp]** experimento a rodar · **[bug]** correção · **[impl]**
implementação nova · **[doc]** documentação/artigo · **[perf]** otimização.

---

## 1. Pendências imediatas

### 1.1 [exp] Refazer o conjunto de suportes reduzidos com a correção da mineração

- **Estado:** os resultados do conjunto reduzido (`results/low_support/`)
  foram apagados porque a última rodada foi interrompida no meio. Só ficaram as
  medições antigas, anteriores à correção da preparação da mineração, em
  `results/pre_fix/raw_low.csv`.
- **Fazer:**
  ```bash
  STEPS="bench report" FPG_SET=low PY=<python> ./scripts/run_all.sh
  ```
  Leva ~1,5–2 h na RTX 4090; o mushroom 0,01 sozinho toma ~25 min, por causa
  da escrita de 90 M itemsets. Isso gera `results/low_support/` e
  `reports/relatorio_fases_suporte_reduzido.md`.
- **Depois:** comparar o novo relatório com `reports/relatorio_fases.md`, que
  foi escrito à mão com os números anteriores à correção. Decidir se ele é
  atualizado ou substituído pelo gerado.

### 1.2 [exp] Refazer o teste de estresse da dinâmica

- **Estado:** `results/stress_dynamic.txt` (420 execuções, 0 problemas) é de
  antes da correção da preparação da mineração. A correção só muda a alocação,
  mas o critério de aceite da Fase 5 pede o teste na versão final.
- **Fazer:** `STEPS="stress" ./scripts/run_all.sh` (~20–30 min).

### 1.3 [exp] kosarak sem o limite de 20 GB de RAM

- **Estado:** todos os processos rodam num cgroup com `MemoryMax=20G` (scripts
  `gpuutil.limited`, `validate_gpu.sh`, `stress_dynamic.sh`). kosarak abaixo de
  ~0,0008 estoura esse limite por causa do tamanho da saída (0,0008 já gera
  53 M itemsets).
- **Decidido:** usar um limite **dinâmico**: RAM disponível (`MemAvailable`)
  menos 8 GB de folga, ainda dentro de um cgroup. Assim, se estourar, só o
  nosso processo morre e o OOM killer não atinge processos de outros usuários.
  Não rodar sem cgroup.
- **Fazer:**
  1. em `scripts/gpuutil.py`, aceitar `FPG_MEM_LIMIT=auto` e calcular o limite
     no momento de cada execução (ler `/proc/meminfo`);
  2. rodar sequencial e GPU dinâmica em kosarak 0,0008 · 0,00075 · 0,0007 ·
     0,00065 · 0,0006 até onde couber, registrando onde cada um falha;
  3. lembrar que as saídas temporárias vão para `results/tmp/` (disco), não
     para `/tmp` (tmpfs, que consome RAM).

### 1.4 [bug] Eclat em GPU falha em 38 casos de teste

- **Estado:** `src/baselines/vertical.cu` (`mine_eclat_gpu`, estilo *Frontier
  Expansion* de Zhang, Zhang e Bakos 2013). O resultado difere do sequencial
  em casos como "duplicadas", "prefixo da seguinte" e várias bases aleatórias.
  O Apriori em GPU (mesmo arquivo) passa em todos os testes.
- **Reproduzir:** `./build/tests --with-eclat` (sem a flag os testes do Eclat
  são pulados, para a suíte padrão passar limpa).
- **Pistas:** a expansão parcial (classe filha aberta entre lotes, campo
  `open`) e a liberação das posições do pool (`to_free`, filhas com um membro)
  são os trechos mais novos e menos testados. Comparar itemsets faltando ou
  sobrando num caso pequeno (por exemplo "duplicadas minsup=1") contra o
  sequencial.

### 1.5 [impl] Comparação com Apriori e Eclat em GPU

- **Estado:** os binários `fpg_apriori_gpu` e `fpg_eclat_gpu` existem (mesmo
  leitor, pré-processamento e saída canônica dos outros), mas ainda não estão
  no benchmark. O Apriori segue o GPApriori (Zhang, Zhang e Bakos, CLUSTER
  2011); o Eclat segue o Frontier Expansion (J. Supercomputing 2013). Não
  existe código público desses trabalhos; as implementações são nossas, "no
  estilo de".
- **Fazer (depois do 1.4):**
  1. acrescentar `gpu_apriori` e `gpu_eclat` a `IMPLS` em `scripts/bench.py`,
     com timeout por execução (o Apriori pode levar horas em suporte baixo),
     registrando "não terminou" em vez de travar o benchmark;
  2. rodar com os suportes padrão (`FPG_SET=common`) e, se o tempo permitir,
     com os reduzidos;
  3. acrescentar a comparação a `plots.py` e a `relatorio_fases.py`.
---

## 2. Otimizações identificadas (headroom)

Medidas e justificativas em `reports/relatorio_fases_suporte_padrao.md`
(seções 8.5, 9.5, 12 e 13) e no artigo (seção "Headroom").

| # | Fase | Melhoria | Ganho esperado |
| --- | --- | --- | --- |
| 2.1 | mineração | dimensionar arenas e pool pela necessidade em vez de 6–9 GB fixos | a preparação custa 7–14 ms por execução, só em `cudaMalloc`/`cudaFree`; é o que ainda faz a GPU perder nas configurações minúsculas |
| 2.2 | mineração | saída em passada única (buffers otimistas que crescem e retomam) | até 2× na mineração; a passada 1 é ~47 % do tempo |
| 2.3 | mineração | dividir entre blocos (ou mandar para a rotina de grade inteira) a construção das árvores condicionais gigantes | até ~4× em accidents (eficiência de balanceamento de 24 %) |
| 2.4 | mineração | investigar `grain` 64 em mushroom 0,1, que fica 6× pior que `grain` 16 | — |
| 2.5 | construção | ordenação lexicográfica por radix híbrido (radix pelos primeiros ranks ou hash do prefixo + comparador nos empates) | construção até ~100–160× sobre o sequencial; deixa de perder em árvores pequenas |
| 2.6 | construção | escrita dos nós coalescida (warp por transação longa ou expansão por scan) | essa subetapa usa 10 % da banda |
| 2.7 | contagem, pré-proc., construção | alocador com cache para os temporários do Thrust/CUB; testar `CUDA_MODULE_LOADING=EAGER` | elimina ~0,5–3 ms fixos por fase |
| 2.8 | saída | montar o CSR de itemsets em bloco (sem `ItemsetList::add` por itemset) ou gerar o formato final na GPU | D2H + montagem chega a segundos com dezenas de milhões de itemsets |
| 2.9 | escrita | buffer único de texto + offsets ordenados (desenhado, nunca aplicado; ver `NOTES.md`) e ordenação paralela | metade da RAM por itemset; até ~12× na escrita |
| 2.10 | leitura | parsing paralelo nos 12 núcleos ou na GPU | leitura é a maior fase restante em kosarak e accidents |
| 2.11 | H2D | memória *pinned* e cópia sobreposta à contagem | ~1–3 ms |

## 3. Melhorias de método

- **[exp] Baseline multi-thread na CPU:** hoje o sequencial é single-thread.
  Uma versão com OpenMP (mineração paralela por item) tornaria a comparação
  mais justa.
- **[exp] Contadores de hardware:** o Nsight Compute exige permissão de
  administrador (`ERR_NVGPUCTRPERM`) e o Nsight Systems desta máquina está sem
  o importador. Com acesso, medir a banda e a ocupação reais de cada kernel no
  lugar das estimativas por tráfego mínimo.
- **[doc] Relatórios históricos:** `reports/fase0.md` … `fase6.md` descrevem
  o estado de cada fase quando ela foi concluída (números da época). Não
  atualizar; servem de histórico.

## 4. Feito recentemente (contexto)

- Suportes padrão (os mais usados na literatura FIMI) passaram a ser o
  conjunto principal (`results/`); os reduzidos viram experimento complementar
  (`FPG_SET=low`).
- Todas as medições usam uma única GPU, a do slot PCIe x16 (`FPG_GPU_BUS`);
  cada execução registra o barramento.
- Correção: arenas e pool da mineração eram zerados fora do cronômetro
  (10–34 ms por execução invisíveis no CSV). Agora a memória não é zerada e a
  preparação é medida (`mineracao_preparacao`).
- Apriori em GPU implementado e validado; Eclat em GPU implementado, com bug (1.4).
