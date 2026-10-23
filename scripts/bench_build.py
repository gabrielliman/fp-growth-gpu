#!/usr/bin/env python3
"""Tarefa 3.10: tempo das subetapas da construção GPU × construção sequencial.

Gera results/build.csv (todas as medidas) e imprime a mediana por configuração.
"""
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import CONFIGS, RESULTS_DIR, ROOT, gpu_env, limited, parse_csv, parse_kv  # noqa: E402

SUB = ["construcao_sort", "construcao_lcp", "construcao_scan", "construcao_nos", "construcao_sparse",
       "construcao_pais", "construcao_contadores", "construcao_cabecalho", "construcao"]
REPS = int(os.environ.get("REPS", "5"))


def run(cmd, env=None):
    p = subprocess.run(limited(cmd), capture_output=True, text=True, env=env, check=True)
    return parse_csv(p.stdout), parse_kv(p.stderr)


def main():
    env = gpu_env()
    out_path = os.path.join(RESULTS_DIR, "build.csv")
    rows = []
    for ds, sups in CONFIGS.items():
        path = os.path.join(ROOT, "data", ds + ".dat")
        for s in sups:
            seq_build, gpu = [], {k: [] for k in SUB}
            nodes = mem = None
            for _ in range(REPS + 1):  # a primeira é aquecimento
                t, _kv = run([os.path.join(ROOT, "build", "fpg_seq"), path, "--minsup", str(s), "--no-mine"])
                seq_build.append(t["construcao"])
                t, kv = run([os.path.join(ROOT, "build", "fpg_gpu"), path, "--minsup", str(s), "--no-mine"], env)
                for k in SUB:
                    gpu[k].append(t.get(k, 0.0))
                nodes, mem = kv.get("nos"), kv.get("mem_construcao_MB")
            seq_build, gpu = seq_build[1:], {k: v[1:] for k, v in gpu.items()}
            for r in range(REPS):
                rows.append([ds, s, r, nodes, mem, seq_build[r]] + [gpu[k][r] for k in SUB])
            med = {k: statistics.median(v) for k, v in gpu.items()}
            ms = statistics.median(seq_build)
            print(f"{ds:11s} {s:<7g} nos={nodes:>8s} mem={mem:>7s}MB seq={ms:9.3f} gpu={med['construcao']:8.3f} "
                  f"speedup={ms / med['construcao']:6.1f}x | " +
                  " ".join(f"{k.replace('construcao_', '')}={med[k]:.3f}" for k in SUB[:-1]), flush=True)
    with open(out_path, "w") as f:
        f.write("dataset,minsup,rep,nos,mem_MB,seq_construcao_ms," + ",".join(SUB) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("gravado", out_path)


if __name__ == "__main__":
    main()
