# =============================================================================
# TREINAMENTO DO CLASSIFICADOR LBP PARA DETECÇÃO DE BOLA
# =============================================================================
# O paper "Improving Field and Ball Detector for Humanoid Robot Soccer EROS
# Platform" usa um Cascade Classifier treinado com descritores LBP para
# verificar se um candidato à bola é de fato a bola ou um falso positivo.
#
# Este script treina um classificador equivalente usando:
#   - Amostras POSITIVAS: patches recortados das bolas anotadas no dataset
#     (anotações em formato COCO do arquivo manual_train.json)
#   - Amostras NEGATIVAS: patches aleatórios de regiões SEM bola nas mesmas
#     imagens (garante que o classificador aprenda o contexto do campo)
#   - Descritor: LBP uniforme (P=8, R=1) -> histograma de 10 bins
#   - Classificador: SVM com kernel RBF — equivalente funcional ao cascade
#     classifier do paper, mas mais simples de treinar sem dados de XML
#
# Por que SVM ao invés do CascadeClassifier do OpenCV?
#   O cascade classifier do OpenCV requer um processo de treinamento externo
#   (opencv_traincascade) que precisa de amostras em formato específico e
#   leva horas. O SVM com LBP produz resultado equivalente em minutos e
#   é o que a maioria dos trabalhos acadêmicos usa como baseline.
#
# Tempo estimado: < 5 minutos num Intel i5 de 14ª geração.
# =============================================================================

import cv2
import numpy as np
import json
import os
import glob
import pickle
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =============================================================================
# Configurações
# =============================================================================
_BASE        = os.path.dirname(os.path.dirname(__file__))
ARCHIVE_DIR  = os.path.join(_BASE, "archive")
ANNOT_FILE   = os.path.join(ARCHIVE_DIR, "annotations", "manual_train.json")
FIGURAS_DIR  = os.path.join(_BASE, "paper", "figuras")
MODELO_DIR   = os.path.join(_BASE, "src")
os.makedirs(FIGURAS_DIR, exist_ok=True)

# Tamanho do patch normalizado (mesmo tamanho para todas as amostras)
PATCH_SIZE = 32

# Parâmetros do LBP — igual ao paper (P=8 pontos, raio=1, método uniforme)
LBP_P      = 8
LBP_R      = 1
LBP_METHOD = "uniform"

# Quantas amostras negativas por positiva (balanceia o dataset)
RATIO_NEG_POS = 3


# =============================================================================
# Funções auxiliares
# =============================================================================

def encontrar_imagem(nome_arquivo):
    """
    Procura a imagem nas subpastas do archive (upper_*).
    O JSON só guarda o nome do arquivo, sem o caminho completo.
    """
    for pasta in glob.glob(os.path.join(ARCHIVE_DIR, "upper_*")):
        caminho = os.path.join(pasta, nome_arquivo)
        if os.path.exists(caminho):
            return caminho
    for pasta in glob.glob(os.path.join(ARCHIVE_DIR, "backup_images", "*")):
        caminho = os.path.join(pasta, nome_arquivo)
        if os.path.exists(caminho):
            return caminho
    return None


def extrair_patch(imagem_cinza, bbox, tamanho=PATCH_SIZE):
    """
    Recorta e redimensiona o patch da bola para tamanho fixo.
    bbox formato: [x1, y1, x2, y2]
    Adiciona margem de 10% ao redor da bbox.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = imagem_cinza.shape

    margem_x = max(int((x2 - x1) * 0.1), 2)
    margem_y = max(int((y2 - y1) * 0.1), 2)
    x1 = max(0, x1 - margem_x)
    y1 = max(0, y1 - margem_y)
    x2 = min(w_img, x2 + margem_x)
    y2 = min(h_img, y2 + margem_y)

    patch = imagem_cinza[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return None

    return cv2.resize(patch, (tamanho, tamanho))


def calcular_descritor_lbp(patch):
    """
    Calcula o histograma LBP normalizado do patch.
    Este é o mesmo descritor usado no paper para caracterizar a textura
    da bola e diferenciá-la de outros objetos.
    """
    lbp = local_binary_pattern(patch, LBP_P, LBP_R, method=LBP_METHOD)
    n_bins = LBP_P + 2  # método uniform: P+2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist


def amostras_negativas_da_imagem(imagem_cinza, bboxes_positivas, n_amostras):
    """
    Extrai patches aleatórios de regiões que NÃO se sobrepõem com nenhuma
    bola anotada. Esses patches são as amostras negativas (não-bola).
    """
    h, w = imagem_cinza.shape
    negativos = []
    tentativas = 0
    tamanho = PATCH_SIZE

    while len(negativos) < n_amostras and tentativas < n_amostras * 20:
        tentativas += 1
        x = np.random.randint(0, w - tamanho)
        y = np.random.randint(0, h - tamanho)

        sobrepoe = False
        for x1, y1, x2, y2 in bboxes_positivas:
            if not (x + tamanho <= x1 or x >= x2 or
                    y + tamanho <= y1 or y >= y2):
                sobrepoe = True
                break

        if not sobrepoe:
            patch = imagem_cinza[y:y + tamanho, x:x + tamanho]
            negativos.append(patch)

    return negativos


# =============================================================================
# Carregamento das anotações e extração de features
# =============================================================================

def carregar_dataset():
    print("Carregando anotações...")
    with open(ANNOT_FILE) as f:
        dados = json.load(f)

    id_para_arquivo = {img["id"]: img["file_name"] for img in dados["images"]}
    annots_bola = [a for a in dados["annotations"] if a["category_id"] == 2]
    print(f"  Anotações de bola encontradas: {len(annots_bola)}")

    bboxes_por_imagem = {}
    for a in annots_bola:
        img_id = a["image_id"]
        if img_id not in bboxes_por_imagem:
            bboxes_por_imagem[img_id] = []
        bboxes_por_imagem[img_id].append(a["bbox"])

    X = []
    y = []

    imagens_processadas = 0
    imagens_nao_encontradas = 0

    for img_id, bboxes in bboxes_por_imagem.items():
        nome_arquivo = id_para_arquivo.get(img_id)
        if nome_arquivo is None:
            continue

        caminho = encontrar_imagem(nome_arquivo)
        if caminho is None:
            imagens_nao_encontradas += 1
            continue

        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            continue

        imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

        for bbox in bboxes:
            patch = extrair_patch(imagem_cinza, bbox)
            if patch is not None:
                feat = calcular_descritor_lbp(patch)
                X.append(feat)
                y.append(1)

        n_neg = len(bboxes) * RATIO_NEG_POS
        negativos = amostras_negativas_da_imagem(imagem_cinza, bboxes, n_neg)
        for patch in negativos:
            feat = calcular_descritor_lbp(patch)
            X.append(feat)
            y.append(0)

        imagens_processadas += 1
        if imagens_processadas % 10 == 0:
            print(f"  Imagens processadas: {imagens_processadas} | "
                  f"Amostras: {len(X)} (pos={sum(y)}, neg={len(y)-sum(y)})")

    print(f"\nTotal: {imagens_processadas} imagens processadas, "
          f"{imagens_nao_encontradas} não encontradas.")
    print(f"Dataset: {len(X)} amostras — "
          f"{sum(y)} positivas, {len(y)-sum(y)} negativas")

    return np.array(X), np.array(y)


# =============================================================================
# Treinamento e avaliação
# =============================================================================

def treinar_e_salvar():
    X, y = carregar_dataset()

    if len(X) == 0:
        print("[ERRO] Nenhuma amostra extraída. Verifique os caminhos.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("\nTreinando SVM com kernel RBF...")
    print("(Isso pode levar alguns minutos — aguarde)")

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Resultados no conjunto de teste ===")
    print(classification_report(y_test, y_pred,
                                 target_names=["não-bola", "bola"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão:")
    print(f"  VN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  VP={cm[1,1]}")

    modelo_path  = os.path.join(MODELO_DIR, "modelo_lbp_bola.pkl")
    scaler_path  = os.path.join(MODELO_DIR, "scaler_lbp_bola.pkl")
    with open(modelo_path, "wb") as f:
        pickle.dump(clf, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModelo salvo em: {modelo_path}")
    print(f"Scaler salvo em: {scaler_path}")

    visualizar_lbp_medio(X, y)

    return clf, scaler


def visualizar_lbp_medio(X, y):
    """
    Plota o histograma LBP médio das amostras positivas e negativas.
    Isso ilustra por que o LBP consegue discriminar bola de não-bola.
    """
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Histograma LBP Médio — Bola vs Não-Bola", fontsize=13)

    bins = np.arange(LBP_P + 2)
    axs[0].bar(bins, X_pos.mean(axis=0), color="green", alpha=0.8)
    axs[0].set_title(f"Bola (n={len(X_pos)})")
    axs[0].set_xlabel("Padrão LBP uniforme")
    axs[0].set_ylabel("Frequência média")
    axs[0].set_xticks(bins)

    axs[1].bar(bins, X_neg.mean(axis=0), color="red", alpha=0.8)
    axs[1].set_title(f"Não-Bola (n={len(X_neg)})")
    axs[1].set_xlabel("Padrão LBP uniforme")
    axs[1].set_ylabel("Frequência média")
    axs[1].set_xticks(bins)

    plt.tight_layout()
    caminho = os.path.join(FIGURAS_DIR, "lbp_histogramas_treino.png")
    plt.savefig(caminho, dpi=150)
    plt.show()
    print(f"Figura salva em: {caminho}")


# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TREINAMENTO: Classificador LBP para Bola (RoboCup)")
    print("=" * 60)
    treinar_e_salvar()
    print("\nTreinamento concluído!")
    print("Use modelo_lbp_bola.pkl e scaler_lbp_bola.pkl na etapa 5.")
