#!/usr/bin/env python3
"""Converte uma saída "itens (suporte)" (ex.: fpgrowth de Borgelt com -v" (%a)")
para o formato canônico: itens em ordem crescente, linhas ordenadas por bytes."""
import sys

lines = []
with open(sys.argv[1]) as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        head, sup = raw.rsplit("(", 1)
        items = sorted(int(x) for x in head.split())
        if not items:  # Borgelt pode emitir o conjunto vazio; não faz parte da saída
            continue
        lines.append(" ".join(map(str, items)) + " (" + sup.strip().rstrip(")") + ")")
lines.sort(key=lambda s: s.encode())
with open(sys.argv[2], "w") as f:
    f.write("".join(l + "\n" for l in lines))
