#!/usr/bin/env python3
"""Fase 6: benchmarks.

Subcomandos:
Conjunto de thresholds: FPG_SET=common (padrão, results/) ou low
(results/low_support/), definidos em gpuutil.THRESHOLD_SETS.

  main    6.1/6.2/6.3/6.6/6.7  dataset × suporte × {seq, gpu estática, gpu dinâmica};
          1 aquecimento + REPS medidas; grava results/raw.csv e results/runs.csv.
  grain   6.4  dinâmica com vários --grain num suporte por dataset; results/grain.csv.
  blocks  6.5  instrumentação por bloco (estática e dinâmica); results/blocks/*.csv.

Toda execução grava a saída canônica e o hash é comparado com o do sequencial
na mesma configuração; uma execução com saída diferente é marcada como inválida.
Cada processo roda num cgroup com limite de memória (gpuutil.limited) e sempre
na mesma GPU (a do slot PCIe x16, gpuutil.GPU_BUS); se ela estiver ocupada por
outros usuários, o script espera.
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuutil import CONFIGS, MID, RESULTS_DIR, ROOT, limited, parse_csv, parse_kv, wait_gpu  # noqa: E402

REPS = int(os.environ.get("REPS", "5"))
GRAINS = [0, 16, 64, 256, 1024, 4096]
IMPLS = {
    "seq": ["fpg_seq"],
    "gpu_static": ["fpg_gpu", "--mine=static"],
    "gpu_dynamic": ["fpg_gpu", "--mine=dynamic"],
}
META = ["pci", "itemsets", "nos", "estouros", "enfileiradas", "tarefas", "desbalanceamento", "mem_construcao_MB",
        "mem_mineracao_MB", "mem_gpu_pico_MB", "blocos"]
TMP = os.path.join(ROOT, "results", "tmp")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_once(impl_cmd, ds, minsup, extra=()):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, "out.txt")
    cmd = [os.path.join(ROOT, "build", impl_cmd[0]), os.path.join(ROOT, "data", ds + ".dat"),
           "--minsup", str(minsup), "--out", out] + impl_cmd[1:] + list(extra)
    env = dict(os.environ)
    if impl_cmd[0] == "fpg_gpu":
        env["CUDA_VISIBLE_DEVICES"] = wait_gpu()
    t0 = time.time()
    p = subprocess.run(limited(cmd), capture_output=True, text=True, env=env)
    wall = time.time() - t0
    if p.returncode != 0:
        return None, {}, wall, f"rc={p.returncode}: {p.stderr.strip()[-200:]}"
    digest = sha(out)
    os.remove(out)
    return parse_csv(p.stdout), parse_kv(p.stderr), wall, digest


def open_csv(path, header):
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(header)
    return f, w


def done_keys(path, ncols):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {tuple(r[:ncols]) for r in csv.reader(f)}


def reference(ds, minsup):
    t, kv, wall, digest = run_once(IMPLS["seq"], ds, minsup)
    if t is None:
        raise RuntimeError(f"sequencial falhou em {ds} {minsup}: {digest}")
    return digest


def cmd_main():
    raw_path = os.path.join(RESULTS_DIR, "raw.csv")
    runs_path = os.path.join(RESULTS_DIR, "runs.csv")
    done = done_keys(runs_path, 3)
    fraw, wraw = open_csv(raw_path, ["impl", "dataset", "minsup", "rep", "etapa", "tempo_ms", "valido"])
    fruns, wruns = open_csv(runs_path, ["impl", "dataset", "minsup", "rep", "valido", "wall_s"] + META)
    for ds, sups in CONFIGS.items():
        for s in sups:
            ref = None
            for impl, icmd in IMPLS.items():
                if (impl, ds, str(s)) in done:
                    continue
                if ref is None:
                    ref = reference(ds, s)
                for rep in range(REPS + 1):  # rep 0 = aquecimento, descartada
                    t, kv, wall, digest = run_once(icmd, ds, s)
                    valid = int(t is not None and digest == ref)
                    if rep == 0:
                        continue
                    for etapa, ms in (t or {}).items():
                        wraw.writerow([impl, ds, s, rep, etapa, ms, valid])
                    wruns.writerow([impl, ds, s, rep, valid, f"{wall:.3f}"] + [kv.get(k, "") for k in META])
                    fraw.flush()
                    fruns.flush()
                    tot = (t or {}).get("total", float("nan"))
                    print(f"{impl:12s} {ds:11s} {s:<8g} rep={rep} total={tot:10.1f} ms "
                          f"{'OK' if valid else 'INVALIDO ' + str(digest)}", flush=True)


def cmd_grain():
    path = os.path.join(RESULTS_DIR, "grain.csv")
    done = done_keys(path, 3)
    f, w = open_csv(path, ["dataset", "minsup", "grain", "rep", "valido", "mineracao_ms", "total_ms", "enfileiradas",
                           "desbalanceamento"])
    for ds, s in MID.items():
        ref = None
        for g in GRAINS:
            if (ds, str(s), str(g)) in done:
                continue
            if ref is None:
                ref = reference(ds, s)
            for rep in range(REPS + 1):
                t, kv, wall, digest = run_once(IMPLS["gpu_dynamic"], ds, s, ["--grain", str(g)])
                valid = int(t is not None and digest == ref)
                if rep == 0:
                    continue
                t = t or {}
                w.writerow([ds, s, g, rep, valid, t.get("mineracao", ""), t.get("total", ""),
                            kv.get("enfileiradas", ""), kv.get("desbalanceamento", "")])
                f.flush()
                print(f"grain {ds:11s} {s:<8g} g={g:<5d} rep={rep} mine={t.get('mineracao', float('nan')):9.2f} ms "
                      f"{'OK' if valid else 'INVALIDO'}", flush=True)


def cmd_blocks():
    os.makedirs(os.path.join(RESULTS_DIR, "blocks"), exist_ok=True)
    for ds, s in MID.items():
        ref = reference(ds, s)
        for impl in ["gpu_static", "gpu_dynamic"]:
            path = os.path.join(RESULTS_DIR, "blocks", f"{impl}_{ds}.csv")
            t, kv, wall, digest = run_once(IMPLS[impl], ds, s, ["--block-stats", path])
            print(f"blocks {impl:12s} {ds:11s} {s:<8g} desbalanceamento={kv.get('desbalanceamento')} "
                  f"{'OK' if digest == ref else 'INVALIDO'}", flush=True)


if __name__ == "__main__":
    {"main": cmd_main, "grain": cmd_grain, "blocks": cmd_blocks}[sys.argv[1] if len(sys.argv) > 1 else "main"]()
