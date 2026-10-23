#!/usr/bin/env bash
# Valida fpg_gpu --debug (pré-processamento, árvore e itemsets x sequencial).
# uso: scripts/validate_gpu.sh "<flags extras>" dataset:minsup ...
source "$(dirname "$0")/gpu_env.sh"
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTRA="$1"; shift
fail=0
for c in "$@"; do
  d=${c%%:*}; s=${c##*:}
  f="$ROOT/data/$d.dat"; [ -f "$f" ] || f="$ROOT/tests/data/$d.dat"
  r=$(timeout 1200 systemd-run --user --scope -q -p MemoryMax=${FPG_MEM_LIMIT:-20G} -p MemorySwapMax=0 \
      "$ROOT/build/fpg_gpu" "$f" --minsup "$s" --debug $EXTRA 2>&1); rc=$?
  nok=$(echo "$r" | grep -c " OK$"); nf=$(echo "$r" | grep -c "FALHOU")
  info=$(echo "$r" | grep -oE "itemsets=[0-9]+|tarefas=[0-9]+|enfileiradas=[0-9]+|estouros=[0-9]+" | tr '\n' ' ')
  if [ $rc -ne 0 ] || [ "$nf" -ne 0 ]; then fail=1; echo "FAIL $c rc=$rc ok=$nok falhas=$nf $info"; echo "$r" | grep -E "FALHOU|erro" | head -3
  else echo "OK   $c ok=$nok $info"; fi
done
exit $fail
