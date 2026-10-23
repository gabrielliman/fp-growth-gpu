"""Estilo comum dos gráficos (paleta categórica de referência, ordem fixa)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Cor segue a entidade: cada implementação tem sempre a mesma cor.
COLORS = {
    "seq": "#52514e",            # referência sequencial: cinza (texto secundário)
    "gpu_static": "#2a78d6",     # slot 1 azul
    "gpu_dynamic": "#eb6834",    # slot 2 laranja
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABELS = {"seq": "Sequencial (CPU)", "gpu_static": "GPU estática", "gpu_dynamic": "GPU dinâmica"}


def setup():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt
