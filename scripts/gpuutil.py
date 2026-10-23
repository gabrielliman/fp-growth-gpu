"""Utilitários compartilhados pelos scripts de benchmark."""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Todas as medições usam uma única GPU fixa: a placa no slot PCIe x16. As duas
# RTX 4090 da máquina estão em links diferentes (x16 e x4), o que muda os tempos
# de transferência. A placa é identificada pelo barramento PCI (estável) e passada
# ao CUDA pelo UUID, que não depende da ordem de enumeração.
GPU_BUS = os.environ.get("FPG_GPU_BUS", "00000000:01:00.0")


def gpu_status():
    """(uuid, memória usada em MiB, utilização %, largura do link) da GPU fixa."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid,memory.used,utilization.gpu,pcie.link.width.current",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        bus, uuid, mem, util, width = [x.strip() for x in line.split(",")]
        if bus.lower() == GPU_BUS.lower():
            return uuid, int(mem), int(util), int(width)
    raise RuntimeError(f"GPU {GPU_BUS} não encontrada")


def other_compute_apps():
    """Processos de computação na GPU fixa que não são deste usuário."""
    uuid = gpu_status()[0]
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    me = os.getuid()
    pids = []
    for line in out.strip().splitlines():
        g, pid = [x.strip() for x in line.split(",")]
        if g != uuid:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid == me:
                continue
        except OSError:
            pass
        pids.append(pid)
    return pids


def wait_gpu():
    """Espera a GPU fixa ficar livre (máquina compartilhada: nunca interrompe
    processos de outros usuários) e devolve o UUID para CUDA_VISIBLE_DEVICES.

    A utilização reportada pelo nvidia-smi é a média do último intervalo de
    amostragem, então logo após uma execução nossa ela ainda aparece alta. Por
    isso o critério é haver processo de outro usuário na GPU ou memória ocupada
    por terceiros, e não a utilização.
    """
    while True:
        uuid, mem, util, _ = gpu_status()
        others = other_compute_apps()
        if not others and mem < 2000:
            return uuid
        print(f"  [espera] GPU {GPU_BUS} em uso por outros (pids {others}, {mem} MiB); "
              "nova verificação em 60 s", flush=True)
        time.sleep(60)


def gpu_env():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = wait_gpu()
    return env


def parse_csv(stdout):
    """Linhas impl,dataset,minsup,etapa,tempo_ms -> dict etapa -> ms."""
    times = {}
    for line in stdout.splitlines():
        parts = line.split(",")
        if len(parts) == 5:
            try:
                times[parts[3]] = float(parts[4])
            except ValueError:
                pass
    return times


def parse_kv(stderr):
    """Extrai pares chave=valor das linhas [seq]/[gpu] do stderr."""
    kv = {}
    for line in stderr.splitlines():
        if line.startswith("[seq]") or line.startswith("[gpu]"):
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
    return kv


MEM_LIMIT = os.environ.get("FPG_MEM_LIMIT", "20G")


def limited(cmd):
    """Roda o comando num cgroup com limite de memória. Se a execução estourar,
    só o nosso processo é morto, e o OOM killer do sistema não chega a escolher
    processos de outros usuários."""
    return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM_LIMIT}",
            "-p", "MemorySwapMax=0"] + list(cmd)


# Conjuntos de thresholds. "common" são os suportes mais usados com estes
# datasets na literatura FIMI (os da tabela do plano original) e formam o
# conjunto principal; "low" são suportes reduzidos para que o sequencial leve
# de 0,17 s a 80 s, mantidos como experimento complementar.
THRESHOLD_SETS = {
    "common": {
        "chess": [0.8, 0.7, 0.6, 0.5],
        "mushroom": [0.3, 0.2, 0.1, 0.05],
        "accidents": [0.8, 0.6, 0.4, 0.3],
        "T10I4D100K": [0.01, 0.005, 0.002, 0.001],
        "retail": [0.01, 0.005, 0.002, 0.001],
        "kosarak": [0.01, 0.005, 0.002, 0.001],
    },
    "low": {
        "chess": [0.5, 0.45, 0.4, 0.35],
        "mushroom": [0.05, 0.03, 0.02, 0.01],
        "accidents": [0.3, 0.2, 0.15, 0.1],
        "T10I4D100K": [0.0002, 0.0001, 0.00005, 0.00003],
        "retail": [0.0002, 0.0001, 0.00005, 0.00003],
        "kosarak": [0.002, 0.001, 0.0009, 0.00085],
    },
}
# Suporte intermediário (o segundo menor) usado nos experimentos de grain e de
# balanceamento por bloco.
MID_SETS = {
    "common": {"chess": 0.6, "mushroom": 0.1, "accidents": 0.4, "T10I4D100K": 0.002, "retail": 0.002,
               "kosarak": 0.002},
    "low": {"chess": 0.45, "mushroom": 0.03, "accidents": 0.2, "T10I4D100K": 0.0001, "retail": 0.0001,
            "kosarak": 0.001},
}
THRESHOLD_SET = os.environ.get("FPG_SET", "common")
CONFIGS = THRESHOLD_SETS[THRESHOLD_SET]
MID = MID_SETS[THRESHOLD_SET]
RESULTS_DIR = os.path.join(ROOT, "results") if THRESHOLD_SET == "common" else \
    os.path.join(ROOT, "results", "low_support")
