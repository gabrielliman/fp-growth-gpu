#!/usr/bin/env bash
# Tarefa 1.8: compara fpg_seq com o fpgrowth de Christian Borgelt.
# uso: scripts/check_reference.sh <caminho do binário fpgrowth de Borgelt>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${1:?informe o binário de referência}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
check() {  # dataset minsup
    local f="$ROOT/data/$1.dat" s="$2"
    local n; n=$(wc -l < "$f")
    local abs; abs=$(python3 -c "import math;print(max(1,math.ceil($s*$n-1e-9)))")
    "$ROOT/build/fpg_seq" "$f" --minsup "$s" --out "$TMP/seq.txt" > /dev/null 2>&1
    "$REF" -ts -s-"$abs" -v" (%a)" "$f" "$TMP/ref_raw.txt" > /dev/null 2>&1
    python3 "$ROOT/scripts/canon.py" "$TMP/ref_raw.txt" "$TMP/ref.txt"
    if cmp -s "$TMP/seq.txt" "$TMP/ref.txt"; then
        echo "OK   $1 minsup=$s ($abs) itemsets=$(wc -l < "$TMP/seq.txt")"
    else
        echo "FAIL $1 minsup=$s ($abs)"; fail=1
    fi
}
for s in 0.9 0.8 0.7 0.6; do check chess $s; done
for s in 0.3 0.2 0.1 0.05; do check mushroom $s; done
for s in 0.01 0.005; do check retail $s; done
for s in 0.01 0.005; do check T10I4D100K $s; done
exit $fail
