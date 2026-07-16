# =============================================================================
# ETAPA 2 - Morfologia Matemática
# =============================================================================
# Depois da segmentação por cor (Etapa 1), a máscara binária que obtemos
# costuma ter bastante "ruído": pequenos buracos dentro da região verde,
# pequenos pontos verdes espalhados fora do campo, bordas irregulares, etc.
#
# O paper usa operações morfológicas para limpar essa máscara antes de
# tentar detectar o campo e a bola. As operações utilizadas são:
#
#   - EROSÃO: "encolhe" as regiões brancas. Remove ruídos pequenos.
#   - DILATAÇÃO: "expande" as regiões brancas. Preenche buracos pequenos.
#   - ABERTURA (opening): erosão seguida de dilatação. Remove ruídos externos.
#   - FECHAMENTO (closing): dilatação seguida de erosão. Fecha buracos internos.
#
# O elemento estruturante define a "forma" da operação. Usarei elipses
# porque o campo tem bordas curvas e a bola é circular.
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Importo a função de segmentação que fiz na etapa anterior
# (pra não precisar repetir o código)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from etapa1_segmentacao_cor import segmentar_campo_hsv

_BASE = os.path.dirname(os.path.dirname(__file__))
IMAGENS_DIR = os.path.join(_BASE, "imagens do archive zip")
FIGURAS_DIR = os.path.join(_BASE, "paper", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

imagens_teste = [
    "campo-normal.png",
    "bola-boa.png",
    "campo-escuro.png",
]


def aplicar_morfologia(mascara_original):
    """
    Aplica operações morfológicas para limpar a máscara binária do campo.

    A sequência de operações segue o que o paper descreve:
    1. Abertura: remove pequenos ruídos/pontos verdes falsos
    2. Fechamento: fecha buracos internos no campo (linhas brancas, sombras)
    3. Uma erosão final pra deixar a borda do campo mais limpa

    Returns:
        dict com as máscaras em cada estágio (pra visualização)
    """
    resultados = {}
    resultados["original"] = mascara_original.copy()

    # Elemento estruturante: elipse de tamanho 5x5
    # Elipse é melhor que quadrado porque responde bem a formas circulares
    # Kernels maiores que 5x5 acabam distorcendo demais a máscara do campo
    elem_pequeno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Usando o mesmo tamanho 5x5 para todas as operações
    # (kernels maiores removem detalhes importantes da borda do campo)
    elem_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- PASSO 1: Abertura (remove ruídos pequenos fora do campo) ---
    # A abertura é erosão + dilatação.
    # Remove componentes conexas menores que o elemento estruturante.
    mascara_aberta = cv2.morphologyEx(mascara_original, cv2.MORPH_OPEN, elem_pequeno)
    resultados["apos_abertura"] = mascara_aberta

    # --- PASSO 2: Fechamento (fecha buracos dentro do campo) ---
    # O fechamento é dilatação + erosão.
    # Preenche buracos menores que o elemento estruturante.
    # Isso é importante porque as linhas brancas do campo criam "buracos"
    # na máscara verde.
    mascara_fechada = cv2.morphologyEx(mascara_aberta, cv2.MORPH_CLOSE, elem_grande)
    resultados["apos_fechamento"] = mascara_fechada

    # --- PASSO 3: Abertura final com elemento grande ---
    # Remove ilhas verdes pequenas que sobraram (ex: placas de propaganda verde)
    mascara_limpa = cv2.morphologyEx(mascara_fechada, cv2.MORPH_OPEN, elem_grande)
    resultados["mascara_final"] = mascara_limpa

    # Calculo quantos pixels foram removidos/adicionados em cada etapa
    n_original = np.count_nonzero(mascara_original)
    n_aberta   = np.count_nonzero(mascara_aberta)
    n_fechada  = np.count_nonzero(mascara_fechada)
    n_limpa    = np.count_nonzero(mascara_limpa)

    print(f"  Pixels verdes - Original: {n_original}")
    print(f"  Pixels verdes - Após abertura:    {n_aberta}  (removidos: {n_original - n_aberta})")
    print(f"  Pixels verdes - Após fechamento:  {n_fechada} (adicionados: {n_fechada - n_aberta})")
    print(f"  Pixels verdes - Máscara final:    {n_limpa}   (removidos: {n_fechada - n_limpa})")

    return resultados


def visualizar_morfologia(imagem_rgb, resultados, titulo):
    """
    Mostra as máscaras em cada etapa da operação morfológica lado a lado.
    Isso deixa bem claro o efeito de cada operação.
    """
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Etapa 2 - Morfologia Matemática\n{titulo}", fontsize=14)

    # Imagem original como referência
    axs[0, 0].imshow(imagem_rgb)
    axs[0, 0].set_title("Imagem Original")
    axs[0, 0].axis("off")

    # Máscara bruta (saída da segmentação)
    axs[0, 1].imshow(resultados["original"], cmap="gray")
    axs[0, 1].set_title("Máscara Bruta\n(saída da etapa 1)")
    axs[0, 1].axis("off")

    # Após abertura
    axs[0, 2].imshow(resultados["apos_abertura"], cmap="gray")
    axs[0, 2].set_title("Após ABERTURA\n(remove ruídos pequenos)")
    axs[0, 2].axis("off")

    # Após fechamento
    axs[1, 0].imshow(resultados["apos_fechamento"], cmap="gray")
    axs[1, 0].set_title("Após FECHAMENTO\n(fecha buracos internos)")
    axs[1, 0].axis("off")

    # Máscara final
    axs[1, 1].imshow(resultados["mascara_final"], cmap="gray")
    axs[1, 1].set_title("Máscara Final\n(após todas as operações)")
    axs[1, 1].axis("off")

    # Campo isolado com a máscara final
    img_bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)
    campo_isolado = cv2.bitwise_and(img_bgr, img_bgr, mask=resultados["mascara_final"])
    campo_rgb = cv2.cvtColor(campo_isolado, cv2.COLOR_BGR2RGB)
    axs[1, 2].imshow(campo_rgb)
    axs[1, 2].set_title("Campo Isolado\n(máscara final aplicada)")
    axs[1, 2].axis("off")

    plt.tight_layout()
    nome_arquivo = os.path.join(FIGURAS_DIR, f"resultado_etapa2_{titulo.replace('.', '_')}.png")
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()
    print(f"  Resultado salvo em: {nome_arquivo}")


def comparar_elementos_estruturantes(mascara_original, titulo):
    """
    Função extra: compara o efeito de diferentes tamanhos de
    elemento estruturante. Útil para entender como o tamanho
    do kernel influencia o resultado.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Comparação de Elementos Estruturantes (Fechamento)\n{titulo}", fontsize=12)

    axs[0].imshow(mascara_original, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")

    for i, tamanho in enumerate([5, 15, 25]):
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))
        resultado = cv2.morphologyEx(mascara_original, cv2.MORPH_CLOSE, elem)
        axs[i + 1].imshow(resultado, cmap="gray")
        axs[i + 1].set_title(f"Kernel {tamanho}x{tamanho}")
        axs[i + 1].axis("off")

    plt.tight_layout()
    nome_arquivo = os.path.join(FIGURAS_DIR, f"resultado_etapa2_comparacao_{titulo.replace('.', '_')}.png")
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()


# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ETAPA 2: Morfologia Matemática")
    print("=" * 60)

    for nome_imagem in imagens_teste:
        caminho = os.path.join(IMAGENS_DIR, nome_imagem)
        print(f"\nProcessando: {nome_imagem}")

        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            print(f"  [ERRO] Não consegui carregar: {caminho}")
            continue

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

        # Passo 1: segmentação por cor (etapa anterior)
        _, mascara_bruta = segmentar_campo_hsv(imagem_bgr)

        # Passo 2: limpeza morfológica
        resultados = aplicar_morfologia(mascara_bruta)

        # Visualização
        visualizar_morfologia(imagem_rgb, resultados, nome_imagem)

        # Comparação de diferentes tamanhos de kernel
        comparar_elementos_estruturantes(mascara_bruta, nome_imagem)

    print("\nEtapa 2 concluída!")
    print("Próximo passo: etapa3_deteccao_campo.py")
