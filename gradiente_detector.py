import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage import filters
from skimage.metrics import structural_similarity as ssim

# ============================================================
# 1. Convolução manual
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
# 2. Magnitude e direção
# ============================================================
def gradient_magnitude_direction(Gx, Gy):
    M = np.sqrt(Gx ** 2 + Gy ** 2)
    D = np.arctan2(Gy, Gx + 1e-8)
    return M, D

# ============================================================
# 3. Supressão não-máxima simplificada
# ============================================================
def non_max_suppression(M):
    M_suppressed = (M / M.max()) * 255
    return M_suppressed.astype(np.uint8)

# ============================================================
# 4. Limiarização simples
# ============================================================
def threshold_hysteresis(M):
    med = np.median(M)
    TL = 0.3 * med
    TH = 1.2 * med

    strong = (M >= TH)
    weak = ((M <= TH) & (M >= TL))

    result = np.zeros_like(M)
    result[strong] = 255
    result[weak] = 100
    return result

# ============================================================
# 5. Operador Gradiente Completo (Sobel / Prewitt)
# ============================================================
def aplicar_operador(img, tipo="sobel"):
    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

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
    M_suppressed = non_max_suppression(M)
    M_final = threshold_hysteresis(M_suppressed)
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

        # Resultados com skimage
        sobel_cv = filters.sobel(img)
        prewitt_cv = filters.prewitt(img)

        # Comparação SSIM
        s_sobel = ssim(grad_sobel / 255.0, sobel_cv, data_range=1.0)
        s_prewitt = ssim(grad_prewitt / 255.0, prewitt_cv, data_range=1.0)

        print(f"SSIM Sobel: {s_sobel:.4f}")
        print(f"SSIM Prewitt: {s_prewitt:.4f}")

        # ====================================================
        # Exibição e salvamento das figuras
        # ====================================================
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title('Imagem Original')

        axs[0, 1].imshow(grad_sobel, cmap='gray')
        axs[0, 1].set_title('Sobel Manual')

        axs[0, 2].imshow(grad_prewitt, cmap='gray')
        axs[0, 2].set_title('Prewitt Manual')

        axs[1, 1].imshow(sobel_cv, cmap='gray')
        axs[1, 1].set_title('Sobel (skimage)')

        axs[1, 2].imshow(prewitt_cv, cmap='gray')
        axs[1, 2].set_title('Prewitt (skimage)')

        for ax in axs.ravel():
            ax.axis('off')

        plt.suptitle(f"Comparativo {nome}")
        plt.tight_layout()
        plt.savefig(f"comparativo_{nome}.png", bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
