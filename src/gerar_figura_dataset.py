import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

_BASE      = os.path.dirname(os.path.dirname(__file__))
IMAGENS_DIR = os.path.join(_BASE, "imagens do archive zip")
FIGURAS_DIR = os.path.join(_BASE, "paper", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# Ordem: sem bola primeiro, depois com bola
IMAGENS = [
    ("campo-normal.png",   False, "Iluminação uniforme"),
    ("campo-escuro.png",   False, "Iluminação reduzida"),
    ("campo-gol.png",      False, "Gol ao fundo"),
    ("campo-vazio.png",    False, "Campo vazio"),
    ("campo-1.png",        False, "Vista alternativa"),
    ("bola-boa.png",       True,  "Bola nítida"),
    ("bola-boa-2.png",     True,  "Bola nítida (2)"),
    ("bola-na-linha.png",  True,  "Bola na linha"),
    ("bola-com-robo.png",  True,  "Bola com robô"),
    ("bola-borrado.png",   True,  "Motion blur"),
    ("penalty-boad.png",   True,  "Pênalti"),
]

NCOLS = 4
NROWS = 3

fig, axs = plt.subplots(NROWS, NCOLS, figsize=(14, 9))
fig.suptitle(
    "Imagens de Entrada — RoboCup SPL (NaoDevils 2019)",
    fontsize=14, fontweight="bold", y=1.01
)

for idx, (nome, tem_bola, descricao) in enumerate(IMAGENS):
    ax = axs[idx // NCOLS][idx % NCOLS]
    caminho = os.path.join(IMAGENS_DIR, nome)
    img_bgr = cv2.imread(caminho)

    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
    else:
        ax.set_facecolor("#dddddd")
        ax.text(0.5, 0.5, "não encontrada", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=8)

    cor_borda = "#2ca02c" if tem_bola else "#1f77b4"

    for spine in ax.spines.values():
        spine.set_edgecolor(cor_borda)
        spine.set_linewidth(3)

    nome_curto = nome.replace(".png", "")
    ax.set_title(f"{nome_curto}\n{descricao}", fontsize=8, pad=4)
    ax.axis("off")

# Oculta o subplot vazio (11 imagens em grade 3×4 = 1 slot sobrando)
axs[NROWS - 1][NCOLS - 1].set_visible(False)

legenda = [
    mpatches.Patch(color="#1f77b4", label="Sem bola (avaliação do campo)"),
    mpatches.Patch(color="#2ca02c", label="Com bola (avaliação do pipeline completo)"),
]
fig.legend(handles=legenda, loc="lower center", ncol=2,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
saida = os.path.join(FIGURAS_DIR, "dataset_imagens_entrada.png")
plt.savefig(saida, dpi=150, bbox_inches="tight")
print(f"Figura salva em: {saida}")
