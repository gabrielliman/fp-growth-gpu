#!/usr/bin/env bash
# Baixa os datasets FIMI para data/. O site original (fimi.uantwerpen.be/data)
# está fora do ar, então usamos o espelho da Wayback Machine.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DIR"
BASE="http://web.archive.org/web/2019id_/http://fimi.uantwerpen.be/data"
for f in chess mushroom retail T10I4D100K; do
  [ -s "$DIR/$f.dat" ] && continue
  echo "baixando $f"
  curl -fsSL --retry 3 --max-time 600 "$BASE/$f.dat" -o "$DIR/$f.dat.tmp" && mv "$DIR/$f.dat.tmp" "$DIR/$f.dat"
done
# kosarak e accidents: o .dat do espelho vem truncado em 1 MiB; usamos o .gz.
for f in kosarak accidents; do
  [ -s "$DIR/$f.dat" ] && continue
  echo "baixando $f.gz"
  curl -fsSL --retry 3 --max-time 600 "$BASE/$f.dat.gz" -o "$DIR/$f.dat.gz"
  gunzip -f "$DIR/$f.dat.gz"
done
wc -l "$DIR"/*.dat
