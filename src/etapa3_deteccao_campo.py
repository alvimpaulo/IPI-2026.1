# =============================================================================
# ETAPA 3 - Detecção do Campo (Field Detection)
# =============================================================================
# Com a máscara morfológica limpa (Etapa 2), agora precisamos identificar
# a região do campo de fato — ou seja, encontrar o maior contorno verde
# e determinar a "borda superior" do campo (field boundary).
#
# O paper descreve que a detecção do campo serve para:
#   1. Encontrar a fronteira superior do campo (horizon line)
#   2. Restringir a busca por objetos somente à área do campo
#      (isso reduz muito o número de candidatos falsos)
#
# A abordagem aqui usa:
#   - Encontrar contornos na máscara binária (cv2.findContours)
#   - Selecionar o maior contorno (que deve ser o campo)
#   - Calcular o convex hull do campo para uma estimativa mais suave da borda
#   - Determinar a linha do horizonte (y mínimo do campo em cada coluna)
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from etapa1_segmentacao_cor import segmentar_campo_hsv
from etapa2_morfologia import aplicar_morfologia

_BASE = os.path.dirname(os.path.dirname(__file__))
IMAGENS_DIR = os.path.join(_BASE, "imagens do archive zip")
FIGURAS_DIR = os.path.join(_BASE, "paper", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

imagens_teste = [
    "campo-normal.png",
    "bola-boa.png",
    "campo-escuro.png",
    "campo-sem-bola.png",
    "campo-gol.png",
]


def encontrar_contorno_campo(mascara_limpa):
    """
    Encontra o contorno principal do campo na máscara binária.

    Usa cv2.findContours para detectar todos os contornos e depois
    seleciona o maior deles (por área), que deve corresponder ao campo.

    Returns:
        contorno_campo: o maior contorno encontrado (ou None)
        hull: o convex hull do contorno (borda convexa)
        area: área do contorno em pixels²
    """
    # findContours retorna uma lista de contornos
    # RETR_EXTERNAL: só contornos externos (sem buracos internos)
    # CHAIN_APPROX_SIMPLE: comprime segmentos horizontais/verticais
    contornos, _ = cv2.findContours(
        mascara_limpa,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        print("  [AVISO] Nenhum contorno encontrado na máscara!")
        return None, None, 0

    # Pego o maior contorno por área
    contorno_campo = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno_campo)

    print(f"  Contornos encontrados: {len(contornos)}")
    print(f"  Maior contorno (campo): área = {area:.0f} pixels²")

    # O convex hull é a "casca convexa" do contorno
    # Isso suaviza irregularidades e dá uma estimativa mais limpa da borda do campo
    hull = cv2.convexHull(contorno_campo)

    return contorno_campo, hull, area


def calcular_linha_horizonte(mascara_limpa, largura):
    """
    Calcula a "linha do horizonte" — o ponto mais alto (menor y) do campo
    em cada coluna da imagem.

    Isso é importante porque objetos acima do horizonte provavelmente
    não estão no campo e podem ser ignorados na busca pela bola.

    Returns:
        horizonte: array com o y mínimo do campo em cada coluna x
    """
    altura, _ = mascara_limpa.shape
    horizonte = np.full(largura, altura - 1, dtype=np.int32)

    for x in range(largura):
        coluna = mascara_limpa[:, x]
        # Procuro o primeiro pixel verde (de cima pra baixo)
        pixels_verdes = np.where(coluna > 0)[0]
        if len(pixels_verdes) > 0:
            horizonte[x] = pixels_verdes[0]

    return horizonte


def detectar_campo(imagem_bgr):
    """
    Pipeline completo de detecção do campo:
    1. Segmentação por cor (etapa 1)
    2. Morfologia (etapa 2)
    3. Detecção do contorno e linha do horizonte

    Returns:
        dict com todos os resultados intermediários e finais
    """
    altura, largura = imagem_bgr.shape[:2]

    # Etapas anteriores
    _, mascara_bruta = segmentar_campo_hsv(imagem_bgr)
    resultados_morfo = aplicar_morfologia(mascara_bruta)
    mascara_limpa = resultados_morfo["mascara_final"]

    # Detecção do contorno
    contorno_campo, hull, area = encontrar_contorno_campo(mascara_limpa)

    # Linha do horizonte
    horizonte = calcular_linha_horizonte(mascara_limpa, largura)

    # Calculo a ROI (Region of Interest) = bounding box do campo
    roi = None
    if contorno_campo is not None:
        x, y, w, h = cv2.boundingRect(contorno_campo)
        roi = (x, y, w, h)
        print(f"  ROI do campo: x={x}, y={y}, w={w}, h={h}")
        print(f"  Campo ocupa {(area / (altura * largura) * 100):.1f}% da imagem")

    return {
        "mascara_limpa": mascara_limpa,
        "contorno_campo": contorno_campo,
        "hull": hull,
        "horizonte": horizonte,
        "roi": roi,
        "altura": altura,
        "largura": largura,
    }


def visualizar_deteccao_campo(imagem_bgr, resultado, titulo):
    """
    Visualiza os resultados da detecção do campo:
    - Contorno do campo sobre a imagem
    - Convex hull (borda suavizada)
    - Linha do horizonte
    - Região de interesse (ROI) destacada
    """
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    altura, largura = imagem_bgr.shape[:2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Etapa 3 - Detecção do Campo\n{titulo}", fontsize=14)

    # --- Plot 1: Máscara limpa com o contorno do campo ---
    img_contorno = imagem_rgb.copy()

    if resultado["contorno_campo"] is not None:
        # Desenho o contorno original em verde
        cv2.drawContours(img_contorno, [resultado["contorno_campo"]], -1, (0, 255, 0), 2)
        # Desenho o convex hull em amarelo
        cv2.drawContours(img_contorno, [resultado["hull"]], -1, (255, 255, 0), 3)

    axs[0].imshow(img_contorno)
    axs[0].set_title("Contorno do Campo\n(verde) e Convex Hull (amarelo)")
    axs[0].axis("off")

    # --- Plot 2: Linha do horizonte ---
    img_horizonte = imagem_rgb.copy()
    horizonte = resultado["horizonte"]

    # Suavizo a linha do horizonte com média móvel pra ficar mais apresentável
    janela = 30
    horizonte_suave = np.convolve(
        horizonte,
        np.ones(janela) / janela,
        mode="same"
    ).astype(np.int32)

    # Desenho a linha do horizonte pixel a pixel
    for x in range(largura - 1):
        y1 = int(horizonte_suave[x])
        y2 = int(horizonte_suave[x + 1])
        cv2.line(img_horizonte, (x, y1), (x + 1, y2), (255, 0, 0), 2)

    axs[1].imshow(img_horizonte)
    axs[1].set_title("Linha do Horizonte\n(azul = borda superior do campo)")
    axs[1].axis("off")

    # --- Plot 3: ROI do campo destacada ---
    img_roi = imagem_rgb.copy()

    if resultado["roi"] is not None:
        x, y, w, h = resultado["roi"]
        # Destaco a área FORA da ROI com uma sobreposição escura
        overlay = img_roi.copy()
        overlay[:y, :] = (overlay[:y, :] * 0.3).astype(np.uint8)
        img_roi = overlay
        # Desenho o retângulo da ROI
        cv2.rectangle(img_roi, (x, y), (x + w, y + h), (255, 100, 0), 3)

    axs[2].imshow(img_roi)
    axs[2].set_title("ROI do Campo\n(região para busca de objetos)")
    axs[2].axis("off")

    plt.tight_layout()
    nome_arquivo = os.path.join(FIGURAS_DIR, f"resultado_etapa3_{titulo.replace('.', '_')}.png")
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()
    print(f"  Resultado salvo em: {nome_arquivo}")


# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ETAPA 3: Detecção do Campo (Field Detection)")
    print("=" * 60)

    for nome_imagem in imagens_teste:
        caminho = os.path.join(IMAGENS_DIR, nome_imagem)
        print(f"\nProcessando: {nome_imagem}")

        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            print(f"  [ERRO] Não consegui carregar: {caminho}")
            continue

        resultado = detectar_campo(imagem_bgr)
        visualizar_deteccao_campo(imagem_bgr, resultado, nome_imagem)

    print("\nEtapa 3 concluída!")
    print("Próximo passo: etapa4_candidatos_bola.py")
