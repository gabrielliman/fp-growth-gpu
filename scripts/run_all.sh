#!/usr/bin/env bash
# Reproduz os resultados de um conjunto de thresholds: build, dados, testes,
# validações, benchmarks, limites de hardware, gráficos, tabelas e relatório.
#
# Variáveis opcionais:
#   FPG_SET        conjunto de thresholds: common (padrão, results/) ou low
#                  (results/low_support/); ver scripts/gpuutil.py
#   STEPS          etapas a rodar, separadas por espaço (padrão: todas)
#                  build data tests reference validate stress bench micro report paper
#   PY             interpretador Python com pandas e matplotlib (padrão: python3)
#   BORGELT        binário fpgrowth de C. Borgelt (etapa "reference"; pulada se ausente)
#   TECTONIC       binário tectonic (etapa "paper"; pulada se ausente)
#   REPS           medidas por configuração (padrão 5, mais 1 de aquecimento)
#   FPG_MEM_LIMIT  limite de memória por processo (padrão 20G)
#   FPG_GPU_BUS    barramento PCI da GPU usada em todas as medições
#                  (padrão 00000000:01:00.0, o slot x16 da máquina original)
#
# Exemplos:
#   PY=~/.venv/bin/python3 ./scripts/run_all.sh
#   STEPS="bench report" FPG_SET=low PY=~/.venv/bin/python3 ./scripts/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PY:-python3}
export FPG_SET=${FPG_SET:-common}
STEPS=${STEPS:-build data tests reference validate stress bench micro report paper}
has() { [[ " $STEPS " == *" $1 "* ]]; }
DIR=results
[ "$FPG_SET" = low ] && DIR=results/low_support
mkdir -p "$DIR" results/micro

if has build; then
  echo "== build"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build build -j
fi

# A partir daqui tudo usa a GPU fixa (FPG_GPU_BUS).
source "$ROOT/scripts/gpu_env.sh"

if has data; then
  echo "== datasets"
  ./scripts/download_data.sh
fi

if has tests; then
  echo "== testes"
  systemd-run --user --scope -q -p MemoryMax=${FPG_MEM_LIMIT:-20G} ./build/tests
fi

if has reference; then
  if [ -n "${BORGELT:-}" ] && [ -x "$BORGELT" ]; then
    echo "== validação do sequencial contra Borgelt"
    ./scripts/check_reference.sh "$BORGELT"
  else
    echo "== validação contra Borgelt pulada (defina BORGELT)"
  fi
fi

if has validate; then
  echo "== validação GPU x sequencial (--debug)"
  ./scripts/validate_gpu.sh "--mine=static" chess:0.6 mushroom:0.1 retail:0.002 T10I4D100K:0.002 kosarak:0.002 accidents:0.4
  ./scripts/validate_gpu.sh "--mine=dynamic" chess:0.6 mushroom:0.1 retail:0.002 T10I4D100K:0.002 kosarak:0.002 accidents:0.4
fi

if has stress; then
  echo "== estresse da dinâmica (20 repetições por configuração)"
  ./scripts/stress_dynamic.sh > results/stress_dynamic.txt
  tail -1 results/stress_dynamic.txt
fi

if has bench; then
  echo "== benchmarks ($FPG_SET → $DIR)"
  rm -f "$DIR/raw.csv" "$DIR/runs.csv" "$DIR/grain.csv"
  rm -rf "$DIR/blocks"
  "$PY" scripts/bench_build.py
  "$PY" scripts/bench.py main
  "$PY" scripts/bench.py grain
  "$PY" scripts/bench.py blocks
fi

if has micro; then
  echo "== limites de hardware e estatísticas por configuração"
  for d in chess mushroom accidents T10I4D100K retail kosarak; do
    n=$(wc -l < data/$d.dat); w=$(wc -w < data/$d.dat)
    ./build/microbench "$n" "$w" | sed "s/^micro,/$d,$n,$w,/"
  done > results/micro/microbench.csv
  (echo "tag,dataset,kernel,ms"; for d in chess mushroom accidents T10I4D100K retail kosarak; do
    ./build/fpg_gpu data/$d.dat --count-bench 50 2>/dev/null; done) > results/micro/countbench.csv
  PY="$PY" ./scripts/config_stats.sh
fi

if has report; then
  echo "== gráficos, tabelas e relatório"
  (cd scripts && "$PY" plots.py)
  out=reports/relatorio_fases_suporte_padrao.md
  [ "$FPG_SET" = low ] && out=reports/relatorio_fases_suporte_reduzido.md
  "$PY" scripts/relatorio_fases.py "$out"
fi

if has paper; then
  if [ -n "${TECTONIC:-}" ] && [ "$FPG_SET" = low ] && [ -f results/low_support/raw.csv ]; then
    echo "== artigo"
    PY="$PY" TECTONIC="$TECTONIC" ./paper/build.sh
  else
    # O artigo ainda usa as macros do conjunto reduzido (ver TODO.md).
    echo "== artigo pulado (precisa de TECTONIC, FPG_SET=low e results/low_support/)"
  fi
fi
echo "pronto"
