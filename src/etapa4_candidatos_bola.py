# =============================================================================
# ETAPA 4 - Detecção de Candidatos à Bola (Blob Detection)
# =============================================================================
# Após detectar o campo e sua fronteira (Etapa 3), o próximo passo do paper
# é identificar regiões dentro do campo que possam ser a bola — os chamados
# "candidatos".
#
# A bola no RoboCup é branca com padrão preto, portanto:
#   - Primeiro inverto a máscara do campo: o que NÃO é verde pode ser a bola
#   - Dentro dessa região, busco blobs (regiões conectadas) que tenham
#     características de tamanho e forma compatíveis com uma bola
#
# O paper menciona usar blob detector combinado com a máscara do campo para
# reduzir candidatos falsos. Aqui implemento isso com:
#   1. Subtração da máscara do campo (inverte: campo=0, resto=1)
#   2. Restrição à ROI do campo (ignora objetos fora do campo)
#   3. SimpleBlobDetector do OpenCV com parâmetros ajustados
#   4. Filtragem por área e circularidade (a bola é redonda!)
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from etapa1_segmentacao_cor import segmentar_campo_hsv
from etapa2_morfologia import aplicar_morfologia
from etapa3_deteccao_campo import detectar_campo

_BASE = os.path.dirname(os.path.dirname(__file__))
IMAGENS_DIR = os.path.join(_BASE, "imagens do archive zip")
FIGURAS_DIR = os.path.join(_BASE, "paper", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

imagens_teste = [
    "bola-boa.png",
    "bola-boa-2.png",
    "bola-na-linha.png",
    "bola-com-robo.png",
    "bola-borrado.png",
    "campo-normal.png",
]


def criar_mascara_nao_campo(mascara_campo, imagem_bgr):
    """
    Cria uma máscara para tudo que NÃO é campo dentro da ROI.

    A ideia é: o que não é verde dentro do campo pode ser a bola, linhas,
    gol ou outros objetos. Depois filtramos pelos candidatos mais prováveis.
    """
    # Inverto a máscara do campo: verde vira preto, resto vira branco
    mascara_invertida = cv2.bitwise_not(mascara_campo)

    # Converto pra cinza e aplico limiar pra isolar objetos claros (bola branca)
    imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

    # Aplico limiar de Otsu: separa automaticamente fundo e objetos
    _, mascara_bola = cv2.threshold(
        imagem_cinza, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Combino: quero pixels que são claros E não são campo verde
    mascara_candidatos = cv2.bitwise_and(mascara_bola, mascara_invertida)

    return mascara_candidatos


def filtrar_por_roi(mascara, roi):
    """
    Zera tudo fora da ROI do campo.
    Assim, só analiso candidatos que estão dentro do campo.
    """
    if roi is None:
        return mascara

    mascara_roi = np.zeros_like(mascara)
    x, y, w, h = roi
    mascara_roi[y:y+h, x:x+w] = mascara[y:y+h, x:x+w]
    return mascara_roi


def detectar_blobs(mascara_candidatos):
    """
    Usa o SimpleBlobDetector do OpenCV para encontrar regiões circulares
    na máscara de candidatos.

    Os parâmetros foram ajustados para detectar bolas de futebol:
    - Área entre 100 e 5000 pixels² (evita ruídos e objetos grandes)
    - Circularidade mínima de 0.5 (a bola é redonda, mas pode estar parcial)
    - Convexidade mínima de 0.7 (bordas relativamente suaves)
    """
    params = cv2.SimpleBlobDetector_Params()

    # Filtra por área (tamanho do blob)
    params.filterByArea = True
    params.minArea = 80       # mínimo ~9x9 pixels
    params.maxArea = 8000     # máximo evita detectar regiões grandes demais

    # Filtra por circularidade (1.0 = círculo perfeito)
    params.filterByCircularity = True
    params.minCircularity = 0.4   # bolas parcialmente vistas têm circularidade menor

    # Filtra por convexidade
    params.filterByConvexity = True
    params.minConvexity = 0.7

    # Filtra por inércia (razão dos eixos: 1.0 = círculo, 0 = linha)
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # O SimpleBlobDetector trabalha com imagem 8-bit
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mascara_candidatos)

    print(f"  Blobs detectados: {len(keypoints)}")
    for i, kp in enumerate(keypoints):
        print(f"    Candidato {i+1}: centro=({kp.pt[0]:.0f}, {kp.pt[1]:.0f}), "
              f"diâmetro={kp.size:.1f}px")

    return keypoints


def detectar_candidatos_por_contorno(mascara_candidatos):
    """
    Método alternativo ao blob detector: usa contornos direto.

    O SimpleBlobDetector às vezes não funciona bem com máscaras binárias
    invertidas. Usar findContours dá mais controle sobre a filtragem.
    """
    contornos, _ = cv2.findContours(
        mascara_candidatos,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        # Filtro por área mínima e máxima
        if area < 80 or area > 8000:
            continue

        # Calculo o perímetro e a circularidade
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue

        # Circularidade = 4π * área / perímetro²
        # Para um círculo perfeito, circularidade = 1.0
        circularidade = (4 * np.pi * area) / (perimetro ** 2)

        if circularidade < 0.35:
            continue

        # Centro e raio aproximado
        (cx, cy), raio = cv2.minEnclosingCircle(contorno)

        candidatos.append({
            "contorno": contorno,
            "area": area,
            "circularidade": circularidade,
            "centro": (int(cx), int(cy)),
            "raio": int(raio),
        })

    # Ordeno por circularidade (mais circular primeiro = mais provável ser bola)
    candidatos.sort(key=lambda c: c["circularidade"], reverse=True)

    print(f"  Candidatos (por contorno): {len(candidatos)}")
    for i, c in enumerate(candidatos[:5]):  # mostro só os 5 melhores
        print(f"    #{i+1}: centro={c['centro']}, raio={c['raio']}px, "
              f"área={c['area']:.0f}px², circ={c['circularidade']:.2f}")

    return candidatos


def pipeline_candidatos(imagem_bgr):
    """
    Pipeline completo para encontrar candidatos à bola:
    1. Segmentação e morfologia (etapas 1 e 2)
    2. Detecção do campo e ROI (etapa 3)
    3. Cria máscara de não-campo
    4. Filtra pela ROI
    5. Detecta candidatos por contorno com filtro de circularidade
    """
    # Etapas anteriores
    resultado_campo = detectar_campo(imagem_bgr)
    mascara_campo = resultado_campo["mascara_limpa"]
    roi = resultado_campo["roi"]

    # Máscara de regiões não-verdes dentro do campo
    mascara_candidatos_bruta = criar_mascara_nao_campo(mascara_campo, imagem_bgr)

    # Restringo à ROI do campo
    mascara_candidatos = filtrar_por_roi(mascara_candidatos_bruta, roi)

    # Limpeza morfológica leve para remover ruídos na máscara de candidatos
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_candidatos = cv2.morphologyEx(mascara_candidatos, cv2.MORPH_OPEN, elem)

    # Detecção dos candidatos
    candidatos = detectar_candidatos_por_contorno(mascara_candidatos)

    return {
        "mascara_campo": mascara_campo,
        "mascara_candidatos": mascara_candidatos,
        "candidatos": candidatos,
        "roi": roi,
    }


def visualizar_candidatos(imagem_bgr, resultado, titulo):
    """
    Visualiza o processo de detecção de candidatos à bola.
    """
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Etapa 4 - Detecção de Candidatos à Bola\n{titulo}", fontsize=14)

    # --- Plot 1: Máscara de candidatos ---
    axs[0].imshow(resultado["mascara_candidatos"], cmap="gray")
    axs[0].set_title("Máscara de Candidatos\n(não-verde dentro do campo)")
    axs[0].axis("off")

    # --- Plot 2: Candidatos marcados com bounding circle ---
    img_candidatos = imagem_rgb.copy()
    for c in resultado["candidatos"]:
        cx, cy = c["centro"]
        raio = c["raio"]
        circ = c["circularidade"]
        # Cor varia com a circularidade: mais verde = mais circular
        cor = (int(255 * (1 - circ)), int(255 * circ), 0)
        cv2.circle(img_candidatos, (cx, cy), raio, cor, 2)
        cv2.circle(img_candidatos, (cx, cy), 2, (255, 0, 0), -1)

    # Desenha a ROI do campo
    if resultado["roi"] is not None:
        x, y, w, h = resultado["roi"]
        cv2.rectangle(img_candidatos, (x, y), (x+w, y+h), (255, 165, 0), 1)

    axs[1].imshow(img_candidatos)
    axs[1].set_title(f"Candidatos Detectados ({len(resultado['candidatos'])})\n"
                     "verde=circular, vermelho=menos circular")
    axs[1].axis("off")

    # --- Plot 3: Top-3 candidatos ampliados ---
    img_top = imagem_rgb.copy()
    cores_top = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
    labels_top = ["1º", "2º", "3º"]

    for i, c in enumerate(resultado["candidatos"][:3]):
        cx, cy = c["centro"]
        raio = max(c["raio"], 10)
        cv2.circle(img_top, (cx, cy), raio + 4, cores_top[i], 3)
        cv2.putText(img_top, labels_top[i], (cx - 10, cy - raio - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cores_top[i], 2)

    axs[2].imshow(img_top)
    axs[2].set_title("Top-3 Candidatos\n(por circularidade)")
    axs[2].axis("off")

    plt.tight_layout()
    nome_arquivo = os.path.join(FIGURAS_DIR, f"resultado_etapa4_{titulo.replace('.', '_')}.png")
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()
    print(f"  Resultado salvo em: {nome_arquivo}")


# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ETAPA 4: Detecção de Candidatos à Bola (Blob Detection)")
    print("=" * 60)

    for nome_imagem in imagens_teste:
        caminho = os.path.join(IMAGENS_DIR, nome_imagem)
        print(f"\nProcessando: {nome_imagem}")

        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            print(f"  [ERRO] Não consegui carregar: {caminho}")
            continue

        resultado = pipeline_candidatos(imagem_bgr)
        visualizar_candidatos(imagem_bgr, resultado, nome_imagem)

    print("\nEtapa 4 concluída!")
    print("Próximo passo: etapa5_classificador_lbp.py")
