import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage import filters
from skimage.metrics import structural_similarity as ssim

# ============================================================
# 1. Função de convolução manual
# ============================================================
def convolve(img, kernel):
    m, n = kernel.shape
    pad = m // 2
    img_padded = np.pad(img, pad, mode='constant')
    result = np.zeros_like(img, dtype=float)

    for i in range(pad, img_padded.shape[0] - pad):
        for j in range(pad, img_padded.shape[1] - pad):
            region = img_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            result[i - pad, j - pad] = np.sum(region * kernel)
    return result


# ============================================================
# 2. Funções auxiliares
# ============================================================
def gradient_magnitude_direction(Gx, Gy):
    M = np.sqrt(Gx**2 + Gy**2)
    D = np.arctan2(Gy, Gx + 1e-8)  # evitar divisão por zero
    return M, D


# ============================================================
# 3. Supressão não-máxima
# ============================================================
def non_max_suppression(M, D):
    M_suppressed = np.zeros_like(M)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            q = 255
            r = 255

            # 0 graus
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = M[i, j + 1]
                r = M[i, j - 1]
            # 45 graus
            elif (22.5 <= angle[i, j] < 67.5):
                q = M[i + 1, j - 1]
                r = M[i - 1, j + 1]
            # 90 graus
            elif (67.5 <= angle[i, j] < 112.5):
                q = M[i + 1, j]
                r = M[i - 1, j]
            # 135 graus
            elif (112.5 <= angle[i, j] < 157.5):
                q = M[i - 1, j - 1]
                r = M[i + 1, j + 1]

            if (M[i, j] >= q) and (M[i, j] >= r):
                M_suppressed[i, j] = M[i, j]
            else:
                M_suppressed[i, j] = 0

    return M_suppressed


# ============================================================
# 4. Limiarização e histerese
# ============================================================
def threshold_hysteresis(M):
    med = np.median(M)
    TL = 0.5 * med
    TH = 1.5 * med

    strong = (M >= TH)
    weak = ((M <= TH) & (M >= TL))

    result = np.zeros_like(M)
    strong_i, strong_j = np.where(strong)
    weak_i, weak_j = np.where(weak)

    result[strong_i, strong_j] = 255
    result[weak_i, weak_j] = 75

    # Histerese - conectar bordas fracas a fortes
    for i in range(1, M.shape[0] - 1):
        for j in range(1, M.shape[1] - 1):
            if result[i, j] == 75:
                if np.any(result[i - 1:i + 2, j - 1:j + 2] == 255):
                    result[i, j] = 255
                else:
                    result[i, j] = 0

    return result


# ============================================================
# 5. Aplicação completa do operador gradiente
# ============================================================
def aplicar_operador(img, tipo="sobel"):
    # Pré-filtragem (Gaussiano)
    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)

    # Máscaras Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Máscaras Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    if tipo == "sobel":
        Gx = convolve(img_smooth, sobel_x)
        Gy = convolve(img_smooth, sobel_y)
    else:
        Gx = convolve(img_smooth, prewitt_x)
        Gy = convolve(img_smooth, prewitt_y)

    M, D = gradient_magnitude_direction(Gx, Gy)
    M_suppressed = non_max_suppression(M, D)
    M_final = threshold_hysteresis(M_suppressed)
    M_final = (M_final / M_final.max()) * 255  # normaliza para 0-255
    M_final = M_final.astype(np.uint8)

    return M_final


# ============================================================
# 6. Execução principal
# ============================================================
def main():
    imagens = ['moedas.png', 'Lua1_gray.jpg', 'chessboard_inv.png', 'img02.jpg']

    for nome in imagens:
        print(f"\nProcessando imagem: {nome}")
        img = cv2.imread(nome, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  Imagem {nome} não encontrada. Pulei.")
            continue

        grad_sobel = aplicar_operador(img, "sobel")
        grad_prewitt = aplicar_operador(img, "prewitt")

        # Comparação com filtros prontos do skimage
        sobel_cv = filters.sobel(img)
        prewitt_cv = filters.prewitt(img)

        # SSIM
        s_sobel = ssim(grad_sobel / 255.0, sobel_cv, data_range=1.0)
        s_prewitt = ssim(grad_prewitt / 255.0, prewitt_cv, data_range=1.0)


        print(f"SSIM Sobel: {s_sobel:.4f}")
        print(f"SSIM Prewitt: {s_prewitt:.4f}")

        # Exibir resultados
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].imshow(grad_sobel, cmap='gray'); axs[0, 0].set_title('Sobel Manual')
        axs[0, 1].imshow(grad_prewitt, cmap='gray'); axs[0, 1].set_title('Prewitt Manual')
        axs[1, 0].imshow(sobel_cv, cmap='gray'); axs[1, 0].set_title('Sobel (skimage)')
        axs[1, 1].imshow(prewitt_cv, cmap='gray'); axs[1, 1].set_title('Prewitt (skimage)')
        for ax in axs.ravel(): ax.axis('off')
        plt.suptitle(f"Comparativo {nome}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
