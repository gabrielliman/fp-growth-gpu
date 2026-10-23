#!/usr/bin/env bash
# Estatísticas estruturais de cada configuração (itens frequentes, transações e
# itens após o pré-processamento, nós da FP-tree) para o conjunto FPG_SET.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SET=${FPG_SET:-common}
PY=${PY:-python3}
CONFIGS=$(cd "$ROOT/scripts" && FPG_SET=$SET "$PY" -c "import gpuutil; print(' '.join(f'{d}:{s}' for d, ss in gpuutil.CONFIGS.items() for s in ss))")
OUT="$ROOT/results/micro/config_stats_$SET.csv"
{
  echo "dataset,minsup,transacoes_orig,itens_orig,itens_distintos,tam_medio,tam_max,itens_freq,transacoes,itens_pre,nos"
  for c in $CONFIGS; do
    d=${c%%:*}; s=${c##*:}; f="$ROOT/data/$d.dat"
    st=$("$ROOT/build/fpg_seq" "$f" --stats)
    kv=$("$ROOT/build/fpg_seq" "$f" --minsup "$s" --no-mine 2>&1 >/dev/null)
    vals=$(echo "$kv" | grep -oE 'itens_freq=[0-9]+|transacoes=[0-9]+|itens_pre=[0-9]+|nos=[0-9]+' | cut -d= -f2 | paste -sd,)
    echo "$d,$s,$(echo "$st" | cut -d, -f2),$(wc -w < "$f" | tr -d ' '),$(echo "$st" | cut -d, -f3-5),$vals"
  done
} > "$OUT"
echo "gravado $OUT"
