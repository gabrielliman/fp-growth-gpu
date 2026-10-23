#!/usr/bin/env bash
# Critério da Fase 5: nenhum deadlock em 20 execuções repetidas de cada
# configuração. Cada execução tem timeout e a saída é comparada com a do sequencial.
source "$(dirname "$0")/gpu_env.sh"
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPS=${REPS:-20}
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
LIM=(systemd-run --user --scope -q -p MemoryMax=${FPG_MEM_LIMIT:-20G} -p MemorySwapMax=0)
total=0; bad=0
for c in chess:0.6 mushroom:0.1 retail:0.002 T10I4D100K:0.002 kosarak:0.002 accidents:0.4 chess:0.5; do
  d=${c%%:*}; s=${c##*:}
  "${LIM[@]}" "$ROOT/build/fpg_seq" "$ROOT/data/$d.dat" --minsup "$s" --out "$TMP/ref.txt" > /dev/null 2>&1
  for g in 0 256 4096; do
    ok=0; hang=0; diff=0
    for r in $(seq "$REPS"); do
      timeout 300 "${LIM[@]}" "$ROOT/build/fpg_gpu" "$ROOT/data/$d.dat" --minsup "$s" --mine=dynamic --grain "$g" \
          --out "$TMP/gpu.txt" > /dev/null 2>&1; rc=$?
      total=$((total+1))
      if [ $rc -eq 124 ]; then hang=$((hang+1)); bad=$((bad+1))
      elif [ $rc -ne 0 ] || ! cmp -s "$TMP/ref.txt" "$TMP/gpu.txt"; then diff=$((diff+1)); bad=$((bad+1))
      else ok=$((ok+1)); fi
    done
    echo "$c grain=$g ok=$ok timeout=$hang erro_ou_diferenca=$diff"
  done
done
echo "total=$total problemas=$bad"
[ $bad -eq 0 ]
