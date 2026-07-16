# =============================================================================
# ETAPA 5 - Classificador em Cascata com Descritores LBP
# =============================================================================
# Esta é a etapa final do pipeline proposto no paper. Após encontrar os
# candidatos à bola (Etapa 4), precisamos verificar quais deles realmente
# são a bola e quais são falsos positivos (linhas, placas, robôs, etc.).
#
# O paper usa um Cascade Classifier treinado com features LBP (Local Binary
# Pattern). O LBP é escolhido por ser leve computacionalmente — ideal para
# hardware embarcado de robôs humanoides.
#
# Como NÃO temos o classificador treinado do paper original, esta etapa:
#   1. Demonstra o cálculo do descritor LBP (como o paper usa)
#   2. Implementa uma verificação heurística baseada em LBP para validar
#      os candidatos (simula o comportamento do cascade classifier)
#
# O LBP funciona assim:
#   - Para cada pixel, compara com seus 8 vizinhos ao redor
#   - Se o vizinho >= pixel central, coloca 1; senão 0
#   - O resultado é um número binário de 8 bits (0-255)
#   - O histograma desses valores descreve a textura local da imagem
#
# Bolas têm textura característica diferente de grama ou placas.
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from skimage.feature import local_binary_pattern

sys.path.insert(0, os.path.dirname(__file__))
from etapa4_candidatos_bola import pipeline_candidatos

_BASE = os.path.dirname(os.path.dirname(__file__))
IMAGENS_DIR = os.path.join(_BASE, "imagens do archive zip")
LABELS_DIR  = os.path.join(_BASE, "datasets", "imagems-campo-lateral", "labels", "train")
FIGURAS_DIR = os.path.join(_BASE, "paper", "figuras")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# Tenta carregar o modelo treinado pelo treinar_lbp.py
# Se não existir, usa o score heurístico como fallback
_MODELO_PATH = os.path.join(os.path.dirname(__file__), "modelo_lbp_bola.pkl")
_SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler_lbp_bola.pkl")

_THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "threshold_lbp_bola.pkl")

_clf       = None
_scaler    = None
_threshold = 0.45  # fallback padrão se não houver threshold salvo

if os.path.exists(_MODELO_PATH) and os.path.exists(_SCALER_PATH):
    with open(_MODELO_PATH, "rb") as f:
        _clf = pickle.load(f)
    with open(_SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
    if os.path.exists(_THRESHOLD_PATH):
        with open(_THRESHOLD_PATH, "rb") as f:
            _threshold = pickle.load(f)
    print(f"[Etapa 5] Modelo LBP treinado carregado (threshold={_threshold:.3f}).")
else:
    print("[Etapa 5] Modelo treinado não encontrado — usando classificador heurístico.")
    print("          Execute src/treinar_lbp.py para treinar o modelo.")

imagens_teste = [
    "bola-boa.png",
    "bola-boa-2.png",
    "bola-na-linha.png",
    "bola-com-robo.png",
    "bola-borrado.png",
    "campo-sem-bola.png",
]


def calcular_lbp(imagem_cinza, raio=1, n_pontos=8):
    """
    Calcula o mapa LBP (Local Binary Pattern) da imagem.

    Parâmetros:
        raio: raio da vizinhança circular (1 = 8 vizinhos diretos)
        n_pontos: número de pontos de amostragem na vizinhança

    O método 'uniform' usa apenas padrões "uniformes" (com no máximo
    2 transições 0->1 ou 1->0), que são os mais informativos.
    Isso reduz o histograma de 256 para n_pontos+2 bins.

    Returns:
        lbp: mapa LBP (mesmo tamanho da imagem)
        histograma: histograma normalizado dos valores LBP
    """
    lbp = local_binary_pattern(imagem_cinza, n_pontos, raio, method="uniform")

    # Histograma dos padrões LBP
    n_bins = n_pontos + 2  # para method="uniform"
    histograma, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True   # normalizo para comparação entre regiões de tamanhos diferentes
    )

    return lbp, histograma


def extrair_patch_candidato(imagem_cinza, candidato, tamanho_padrao=32):
    """
    Recorta e redimensiona a região do candidato para um tamanho padrão.

    O cascade classifier do paper trabalha com patches de tamanho fixo,
    então preciso normalizar o tamanho dos candidatos antes de classificar.
    """
    cx, cy = candidato["centro"]
    raio = max(candidato["raio"], 8)   # raio mínimo de 8 pixels

    # Margem extra ao redor do candidato (20%)
    margem = int(raio * 0.2)
    raio_patch = raio + margem

    altura, largura = imagem_cinza.shape
    x1 = max(0, cx - raio_patch)
    y1 = max(0, cy - raio_patch)
    x2 = min(largura, cx + raio_patch)
    y2 = min(altura, cy + raio_patch)

    patch = imagem_cinza[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    # Redimensiono para tamanho padrão (facilita comparação)
    patch_redimensionado = cv2.resize(patch, (tamanho_padrao, tamanho_padrao))

    return patch_redimensionado


def calcular_score_bola(patch_cinza):
    """
    Calcula um "score" de quão parecido com uma bola o patch é.

    O paper usa um cascade classifier com LBP treinado. Aqui implemento
    uma versão simplificada usando características que bolas costumam ter:

    1. Contraste: a bola tem padrão preto/branco -> alto contraste
    2. Simetria: bolas são circulares -> simetria radial
    3. Distribuição LBP: textura característica da bola

    Retorna um score entre 0 e 1 (1 = muito provável ser bola)
    """
    if patch_cinza is None or patch_cinza.size == 0:
        return 0.0

    # --- Feature 1: Contraste da região ---
    # Bola tem contraste moderado a alto (padrão preto e branco)
    desvio_padrao = np.std(patch_cinza.astype(float))
    score_contraste = min(desvio_padrao / 80.0, 1.0)

    # --- Feature 2: Simetria radial ---
    # Calculo a simetria espelhando horizontalmente e verticalmente
    patch_flip_h = cv2.flip(patch_cinza, 1)
    patch_flip_v = cv2.flip(patch_cinza, 0)
    simetria_h = 1 - (np.mean(np.abs(patch_cinza.astype(float) - patch_flip_h)) / 255)
    simetria_v = 1 - (np.mean(np.abs(patch_cinza.astype(float) - patch_flip_v)) / 255)
    score_simetria = (simetria_h + simetria_v) / 2

    # --- Feature 3: Histograma LBP ---
    _, hist_lbp = calcular_lbp(patch_cinza, raio=1, n_pontos=8)

    # A bola tem textura com padrões uniformes distribuídos
    # (não muito homogêneo como a grama, não muito caótico como ruído)
    entropia_lbp = -np.sum(hist_lbp * np.log(hist_lbp + 1e-10))
    # Normalizo a entropia (log(10) é o máximo teórico para 10 bins)
    score_lbp = min(entropia_lbp / np.log(10), 1.0)

    # --- Score final (média ponderada) ---
    # Dou mais peso ao contraste e LBP, que são os mais discriminativos
    score_final = (0.4 * score_contraste +
                   0.2 * score_simetria +
                   0.4 * score_lbp)

    return score_final


def classificar_com_modelo(patch_cinza):
    """
    Usa o modelo SVM treinado com LBP para classificar o patch.
    Retorna a probabilidade de ser bola (entre 0 e 1).
    As features devem ser idênticas às usadas no treinar_lbp.py:
    histograma LBP uniforme (P=8, R=1, 10 bins).
    """
    if patch_cinza is None:
        return 0.0
    patch_32 = cv2.resize(patch_cinza, (32, 32))
    lbp = local_binary_pattern(patch_32, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feat = _scaler.transform([hist])
    prob = _clf.predict_proba(feat)[0][1]
    return float(prob)


def classificar_candidatos(imagem_bgr, candidatos, limiar=0.45):
    """
    Classifica cada candidato como bola ou não-bola.

    Se o modelo treinado (treinar_lbp.py) estiver disponível, usa ele.
    Caso contrário, usa o score heurístico baseado em contraste, simetria
    e entropia LBP — equivalente ao que o paper descreve conceitualmente.

    Returns:
        lista de (candidato, score, eh_bola)
    """
    imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    resultados = []

    # Usa threshold otimizado do treino se disponível
    limiar_efetivo = _threshold if _clf is not None else limiar

    for candidato in candidatos:
        patch = extrair_patch_candidato(imagem_cinza, candidato)
        if _clf is not None:
            score = classificar_com_modelo(patch)
        else:
            score = calcular_score_bola(patch)
        eh_bola = score >= limiar_efetivo

        resultados.append({
            "candidato": candidato,
            "patch": patch,
            "score": score,
            "eh_bola": eh_bola,
        })

        status = "BOLA ✓" if eh_bola else "não-bola"
        print(f"  Candidato {candidato['centro']}: score={score:.3f} -> {status}")

    return resultados


def _desenhar_candidatos_na_imagem(imagem_rgb, resultado_classificacao):
    """
    Retorna uma cópia da imagem com todos os candidatos anotados:
      - círculo amarelo numerado = candidato (antes da classificação)
      - círculo verde espesso    = classificado como BOLA
      - círculo vermelho fino    = rejeitado
    """
    img = imagem_rgb.copy()
    for i, res in enumerate(resultado_classificacao):
        c = res["candidato"]
        cx, cy = c["centro"]
        raio = max(c["raio"], 8)
        cor = (0, 200, 0) if res["eh_bola"] else (220, 50, 50)
        espessura = 3 if res["eh_bola"] else 1
        cv2.circle(img, (cx, cy), raio + 4, cor, espessura)
        # Número do candidato
        cv2.putText(img, str(i + 1), (cx - 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
        # Score abaixo do círculo
        cv2.putText(img, f"{res['score']:.2f}", (cx - 14, cy + raio + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)
    return img


def _lbp_colorido(imagem_cinza):
    """Retorna o mapa LBP normalizado para exibição com colormap jet."""
    lbp, _ = calcular_lbp(imagem_cinza, raio=1, n_pontos=8)
    lbp_norm = (lbp / lbp.max() * 255).astype(np.uint8)
    return lbp_norm


def visualizar_visao_geral(imagem_bgr, resultado_classificacao, titulo):
    """
    Figura 1 por imagem — visão geral com 6 painéis:
      [1] Imagem original com nome do arquivo em destaque
      [2] Todos os candidatos numerados (antes da classificação)
      [3] Mapa LBP da imagem inteira
      [4] Resultado final (verde=bola, vermelho=rejeitado)
      [5] Histograma LBP global da imagem
      [6] Distribuição dos scores de todos os candidatos
    """
    imagem_rgb   = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Etapa 5 — Classificador LBP\nImagem: {titulo}",
                 fontsize=14, fontweight="bold")

    # [0,0] Original com título destacado
    axs[0, 0].imshow(imagem_rgb)
    axs[0, 0].set_title("(1) Imagem Original", fontsize=11)
    axs[0, 0].axis("off")

    # [0,1] Candidatos numerados (todos, antes da classificação)
    img_cands = imagem_rgb.copy()
    for i, res in enumerate(resultado_classificacao):
        c = res["candidato"]
        cx, cy = c["centro"]
        raio = max(c["raio"], 8)
        cv2.circle(img_cands, (cx, cy), raio + 4, (255, 200, 0), 2)
        cv2.putText(img_cands, str(i + 1), (cx - 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    axs[0, 1].imshow(img_cands)
    axs[0, 1].set_title(f"(2) Candidatos encontrados: {len(resultado_classificacao)}\n"
                        f"(amarelo = candidato numerado)", fontsize=11)
    axs[0, 1].axis("off")

    # [0,2] Mapa LBP com candidatos sobrepostos
    lbp_norm = _lbp_colorido(imagem_cinza)
    lbp_rgb  = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_JET)
    lbp_rgb  = cv2.cvtColor(lbp_rgb, cv2.COLOR_BGR2RGB)
    for res in resultado_classificacao:
        c = res["candidato"]
        cx, cy = c["centro"]
        raio = max(c["raio"], 8)
        cor = (0, 255, 0) if res["eh_bola"] else (255, 80, 80)
        cv2.circle(lbp_rgb, (cx, cy), raio + 4, cor, 2)
    axs[0, 2].imshow(lbp_rgb)
    axs[0, 2].set_title("(3) Mapa LBP + candidatos\n(verde=bola, vermelho=rejeitado)", fontsize=11)
    axs[0, 2].axis("off")

    # [1,0] Resultado final anotado
    img_result = _desenhar_candidatos_na_imagem(imagem_rgb, resultado_classificacao)
    n_bolas = sum(1 for r in resultado_classificacao if r["eh_bola"])
    axs[1, 0].imshow(img_result)
    axs[1, 0].set_title(f"(4) Resultado final\n"
                        f"Bolas: {n_bolas} / {len(resultado_classificacao)} candidatos",
                        fontsize=11)
    axs[1, 0].axis("off")

    # [1,1] Histograma LBP global
    _, hist_global = calcular_lbp(imagem_cinza)
    bins = range(len(hist_global))
    axs[1, 1].bar(bins, hist_global, color="steelblue", alpha=0.85)
    axs[1, 1].set_title("(5) Histograma LBP global da imagem\n"
                        "(P=8, R=1, método uniforme)", fontsize=11)
    axs[1, 1].set_xlabel("Padrão LBP uniforme")
    axs[1, 1].set_ylabel("Frequência normalizada")
    axs[1, 1].set_xticks(list(bins))
    axs[1, 1].grid(axis="y", alpha=0.3)

    # [1,2] Scores dos candidatos (barras horizontais coloridas)
    if resultado_classificacao:
        nomes  = [f"#{i+1} ({r['candidato']['centro'][0]},{r['candidato']['centro'][1]})"
                  for i, r in enumerate(resultado_classificacao)]
        scores = [r["score"] for r in resultado_classificacao]
        cores  = ["green" if r["eh_bola"] else "tomato" for r in resultado_classificacao]
        ypos   = range(len(nomes))
        axs[1, 2].barh(list(ypos), scores, color=cores, alpha=0.85)
        axs[1, 2].set_yticks(list(ypos))
        axs[1, 2].set_yticklabels(nomes, fontsize=8)
        axs[1, 2].axvline(_threshold if _clf is not None else 0.45,
                          color="black", ls="--", lw=1.5, label="limiar")
        axs[1, 2].set_xlabel("Score LBP")
        axs[1, 2].set_title("(6) Score de cada candidato\n"
                            "(verde=bola, vermelho=rejeitado, linha=limiar)", fontsize=11)
        axs[1, 2].legend(fontsize=8)
        axs[1, 2].set_xlim(0, 1)
        axs[1, 2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    nome_saida = os.path.join(
        FIGURAS_DIR,
        f"resultado_etapa5_{titulo.replace('.', '_')}.png"
    )
    plt.savefig(nome_saida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura geral salva: {nome_saida}")


def visualizar_patches_detalhados(imagem_bgr, resultado_classificacao, titulo):
    """
    Figura 2 por imagem — detalhe de cada candidato individualmente:
    Para cada candidato mostra patch cinza, mapa LBP do patch e histograma LBP.
    Organizado em linhas de 3 colunas por candidato.
    """
    if not resultado_classificacao:
        return

    imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    n = len(resultado_classificacao)
    n_cols = 3  # patch | lbp_patch | histograma
    n_rows = n

    fig, axs = plt.subplots(n_rows, n_cols,
                             figsize=(12, max(3 * n_rows, 4)),
                             squeeze=False)
    fig.suptitle(f"Etapa 5 — Detalhe LBP por candidato\nImagem: {titulo}",
                 fontsize=13, fontweight="bold")

    for i, res in enumerate(resultado_classificacao):
        patch = res["patch"]
        c     = res["candidato"]
        cx, cy = c["centro"]
        veredicto = "BOLA ✓" if res["eh_bola"] else "rejeitado ✗"
        cor_titulo = "darkgreen" if res["eh_bola"] else "crimson"

        linha_label = (f"Candidato #{i+1}  |  centro=({cx},{cy})  "
                       f"raio={c['raio']}px  |  score={res['score']:.3f}  →  {veredicto}")

        if patch is not None:
            lbp_patch, hist_patch = calcular_lbp(patch, raio=1, n_pontos=8)

            # Coluna 0: patch em escala de cinza
            axs[i, 0].imshow(patch, cmap="gray", vmin=0, vmax=255)
            axs[i, 0].set_title(linha_label, fontsize=8,
                                 color=cor_titulo, fontweight="bold")
            axs[i, 0].set_xlabel("Patch (32×32px)")
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])

            # Coluna 1: mapa LBP do patch
            axs[i, 1].imshow(lbp_patch, cmap="hot")
            axs[i, 1].set_title("Mapa LBP do patch", fontsize=8)
            axs[i, 1].set_xlabel("Valores LBP uniforme")
            axs[i, 1].set_xticks([])
            axs[i, 1].set_yticks([])

            # Coluna 2: histograma LBP do patch
            bins = range(len(hist_patch))
            cor_barra = "seagreen" if res["eh_bola"] else "tomato"
            axs[i, 2].bar(list(bins), hist_patch, color=cor_barra, alpha=0.85)
            axs[i, 2].set_title("Histograma LBP do patch", fontsize=8)
            axs[i, 2].set_xlabel("Padrão LBP")
            axs[i, 2].set_ylabel("Freq. norm.")
            axs[i, 2].set_xticks(list(bins))
            axs[i, 2].grid(axis="y", alpha=0.3)

            # Anotação do score dentro do histograma
            axs[i, 2].text(0.98, 0.95, f"score={res['score']:.3f}",
                           transform=axs[i, 2].transAxes,
                           ha="right", va="top", fontsize=8,
                           color=cor_titulo, fontweight="bold")
        else:
            for col in range(n_cols):
                axs[i, col].text(0.5, 0.5, "patch inválido",
                                 ha="center", va="center", color="gray")
                axs[i, col].axis("off")

    plt.tight_layout()
    nome_saida = os.path.join(
        FIGURAS_DIR,
        f"resultado_etapa5_patches_{titulo.replace('.', '_')}.png"
    )
    plt.savefig(nome_saida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura patches salva: {nome_saida}")


def visualizar_lbp_e_classificacao(imagem_bgr, resultado_candidatos,
                                    resultado_classificacao, titulo):
    """Ponto de entrada das visualizações: gera as duas figuras por imagem."""
    visualizar_visao_geral(imagem_bgr, resultado_classificacao, titulo)
    visualizar_patches_detalhados(imagem_bgr, resultado_classificacao, titulo)


def visualizar_resumo_comparativo(historico):
    """
    Figura final com resumo de todas as imagens processadas:
      - Quantos candidatos cada imagem tinha
      - Quantos foram classificados como bola
      - Score máximo e mínimo por imagem
    """
    if not historico:
        return

    nomes   = [h["nome"] for h in historico]
    n_cands = [h["n_candidatos"] for h in historico]
    n_bolas = [h["n_bolas"] for h in historico]
    scores_max = [h["score_max"] for h in historico]
    scores_med = [h["score_med"] for h in historico]

    x = np.arange(len(nomes))
    w = 0.35

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Etapa 5 — Resumo comparativo de todas as imagens", fontsize=13)

    # Candidatos vs bolas detectadas
    axs[0].bar(x - w/2, n_cands, w, label="Candidatos", color="steelblue", alpha=0.85)
    axs[0].bar(x + w/2, n_bolas, w, label="Classificados como bola", color="seagreen", alpha=0.85)
    axs[0].set_xticks(list(x))
    axs[0].set_xticklabels(nomes, rotation=20, ha="right", fontsize=9)
    axs[0].set_ylabel("Quantidade")
    axs[0].set_title("Candidatos × Bolas detectadas")
    axs[0].legend()
    axs[0].grid(axis="y", alpha=0.3)

    # Score máximo e médio por imagem
    axs[1].plot(list(x), scores_max, "o-", color="tomato",  label="Score máximo", lw=2)
    axs[1].plot(list(x), scores_med, "s--", color="orange", label="Score médio",  lw=2)
    thr = _threshold if _clf is not None else 0.45
    axs[1].axhline(thr, color="black", ls=":", lw=1.5, label=f"Limiar ({thr:.2f})")
    axs[1].set_xticks(list(x))
    axs[1].set_xticklabels(nomes, rotation=20, ha="right", fontsize=9)
    axs[1].set_ylabel("Score LBP")
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Score máximo e médio por imagem")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    nome_saida = os.path.join(FIGURAS_DIR, "resultado_etapa5_resumo_comparativo.png")
    plt.savefig(nome_saida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigura de resumo salva: {nome_saida}")


# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ETAPA 5: Classificador em Cascata com Descritores LBP")
    print("=" * 60)

    total_bolas_detectadas = 0
    total_imagens_com_bola = 0
    historico = []  # para o resumo comparativo final

    for nome_imagem in imagens_teste:
        caminho = os.path.join(IMAGENS_DIR, nome_imagem)
        print(f"\n{'='*50}")
        print(f"Processando: {nome_imagem}")
        print(f"{'='*50}")

        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            print(f"  [ERRO] Não consegui carregar: {caminho}")
            continue

        h, w = imagem_bgr.shape[:2]
        print(f"  Dimensões: {w}×{h} px")

        # Etapas 1-4: encontra candidatos
        resultado_candidatos = pipeline_candidatos(imagem_bgr)
        candidatos = resultado_candidatos["candidatos"]
        print(f"  Candidatos encontrados (Etapa 4): {len(candidatos)}")

        if len(candidatos) == 0:
            print("  Nenhum candidato para classificar — pulando visualização.")
            historico.append({
                "nome": nome_imagem,
                "n_candidatos": 0,
                "n_bolas": 0,
                "score_max": 0.0,
                "score_med": 0.0,
            })
            continue

        # Etapa 5: classifica com LBP
        resultado_classificacao = classificar_candidatos(imagem_bgr, candidatos)

        scores  = [r["score"] for r in resultado_classificacao]
        n_bolas = sum(1 for r in resultado_classificacao if r["eh_bola"])

        if n_bolas > 0:
            total_bolas_detectadas += 1
        if "sem-bola" not in nome_imagem and "vazio" not in nome_imagem:
            total_imagens_com_bola += 1

        historico.append({
            "nome": nome_imagem,
            "n_candidatos": len(candidatos),
            "n_bolas": n_bolas,
            "score_max": float(max(scores)) if scores else 0.0,
            "score_med": float(np.mean(scores)) if scores else 0.0,
        })

        print(f"  Resultado: {n_bolas} bola(s) confirmada(s) de {len(candidatos)} candidatos")

        visualizar_lbp_e_classificacao(
            imagem_bgr, resultado_candidatos, resultado_classificacao, nome_imagem
        )

    # Figura final: resumo comparativo de todas as imagens
    visualizar_resumo_comparativo(historico)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO CONCLUÍDO")
    print("=" * 60)
    print(f"Bolas detectadas em {total_bolas_detectadas} / "
          f"{total_imagens_com_bola} imagens com bola.")
    print("\nResumo das etapas:")
    print("  1. Segmentação HSV       -> identifica pixels verdes do campo")
    print("  2. Morfologia            -> limpa a máscara (remove ruídos)")
    print("  3. Detecção do campo     -> encontra fronteira e ROI do campo")
    print("  4. Candidatos à bola     -> blobs não-verdes dentro do campo")
    print("  5. Classificador LBP     -> valida candidatos pela textura LBP")
    print(f"\nFiguras salvas em: {FIGURAS_DIR}")
