# =============================================================================
# ETAPA 1 - Segmentação por Cor (Espaço HSV)
# =============================================================================
# O paper "Improving Field and Ball Detector for Humanoid Robot Soccer EROS
# Platform" propõe usar segmentação por cor como primeiro passo do pipeline
# de detecção do campo e da bola. A ideia é bem simples: converter a imagem
# para o espaço de cor HSV (Hue, Saturation, Value) e criar uma máscara
# binária que destaca somente os pixels verdes — que correspondem ao gramado.
#
# O HSV é preferível ao RGB para isso porque separa a informação de cor (Hue)
# da luminosidade (Value), tornando a segmentação mais robusta a variações
# de iluminação que são comuns em ambientes de competição de robótica.
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Caminhos relativos à pasta src/
_BASE = os.path.dirname(os.path.dirname(__file__))
IMAGENS_DIR = os.path.join(_BASE, "imagens do archive zip")
FIGURAS_DIR = os.path.join(_BASE, "paper", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# Vou usar algumas imagens de exemplo do conjunto de dados
imagens_teste = [
    "campo-normal.png",
    "bola-boa.png",
    "campo-escuro.png",
    "bola-boa-2.png",
    "bola-na-linha.png",
    "bola-com-robo.png",
    "bola-borrado.png",
    "campo-vazio.png",

]


def segmentar_campo_hsv(imagem_bgr):
    """
    Converte a imagem para HSV e aplica uma máscara para isolar o campo verde.

    O espaço HSV facilita muito a segmentação por cor porque:
    - H (Hue): representa a cor em si (verde está em torno de 35-85 graus)
    - S (Saturation): quão "pura" é a cor (evita tons acinzentados)
    - V (Value): brilho da cor (não afeta tanto a detecção do verde)

    Parâmetros HSV para verde do gramado:
    - H: 35 a 85 (em OpenCV, H vai de 0 a 179, então corresponde a ~17-42)
    - S: mínimo de 40 para evitar branco/cinza
    - V: mínimo de 40 para evitar preto
    """

    # Primeiro converto de BGR (padrão OpenCV) para HSV
    imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    # Defino os limites inferior e superior do verde no HSV
    # Esses valores são os mesmos usados no paper: segmentação simples do verde
    verde_baixo = np.array([30, 40, 40])   # H=30, S=40, V=40
    verde_alto  = np.array([90, 255, 255]) # H=90, S=255, V=255

    # Cria a máscara binária: 255 onde é verde, 0 onde não é
    mascara_verde = cv2.inRange(imagem_hsv, verde_baixo, verde_alto)

    return imagem_hsv, mascara_verde


def processar_imagem(caminho_imagem):
    """
    Carrega a imagem e aplica a segmentação por cor.
    Retorna a imagem original, a versão HSV e a máscara do campo.
    """
    imagem_bgr = cv2.imread(caminho_imagem)

    if imagem_bgr is None:
        print(f"  [ERRO] Não consegui carregar: {caminho_imagem}")
        return None, None, None

    imagem_hsv, mascara = segmentar_campo_hsv(imagem_bgr)

    # Converto BGR -> RGB só para mostrar corretamente no matplotlib
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    # Calculo a porcentagem de pixels verdes na imagem
    total_pixels = mascara.size
    pixels_verdes = np.count_nonzero(mascara)
    porcentagem = (pixels_verdes / total_pixels) * 100
    print(f"  Pixels verdes: {pixels_verdes} / {total_pixels} ({porcentagem:.1f}%)")

    return imagem_rgb, imagem_hsv, mascara


def visualizar_resultados(imagem_rgb, imagem_hsv, mascara, titulo):
    """
    Mostra a imagem original, os canais HSV separados e a máscara resultante.
    Isso ajuda a entender o que cada canal contribui para a segmentação.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Etapa 1 - Segmentação por Cor HSV\n{titulo}", fontsize=14)

    # Imagem original
    axs[0, 0].imshow(imagem_rgb)
    axs[0, 0].set_title("Imagem Original (RGB)")
    axs[0, 0].axis("off")

    # Canal H (Hue) - onde fica a informação de cor
    axs[0, 1].imshow(imagem_hsv[:, :, 0], cmap="hsv")
    axs[0, 1].set_title("Canal H (Matiz/Hue)")
    axs[0, 1].axis("off")

    # Canal S (Saturação)
    axs[0, 2].imshow(imagem_hsv[:, :, 1], cmap="gray")
    axs[0, 2].set_title("Canal S (Saturação)")
    axs[0, 2].axis("off")

    # Canal V (Valor/Brilho)
    axs[1, 0].imshow(imagem_hsv[:, :, 2], cmap="gray")
    axs[1, 0].set_title("Canal V (Brilho/Value)")
    axs[1, 0].axis("off")

    # Máscara binária do campo verde
    axs[1, 1].imshow(mascara, cmap="gray")
    axs[1, 1].set_title("Máscara Verde (campo)")
    axs[1, 1].axis("off")

    # Imagem original com a máscara aplicada (só o campo aparece)
    campo_isolado = cv2.bitwise_and(
        cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR),
        mask=mascara
    )
    campo_isolado_rgb = cv2.cvtColor(campo_isolado, cv2.COLOR_BGR2RGB)
    axs[1, 2].imshow(campo_isolado_rgb)
    axs[1, 2].set_title("Campo Isolado")
    axs[1, 2].axis("off")

    plt.tight_layout()
    nome_arquivo = os.path.join(FIGURAS_DIR, f"resultado_etapa1_{titulo.replace('.', '_')}.png")
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()
    print(f"  Resultado salvo em: {nome_arquivo}")


# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ETAPA 1: Segmentação por Cor no Espaço HSV")
    print("=" * 60)

    for nome_imagem in imagens_teste:
        caminho = os.path.join(IMAGENS_DIR, nome_imagem)
        print(f"\nProcessando: {nome_imagem}")

        imagem_rgb, imagem_hsv, mascara = processar_imagem(caminho)

        if imagem_rgb is not None:
            visualizar_resultados(imagem_rgb, imagem_hsv, mascara, nome_imagem)

    print("\nEtapa 1 concluída!")
    print("Próximo passo: etapa2_morfologia.py")
