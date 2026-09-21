"""JPEG restoration methods, self-contained (no imports from other files).

All 7 candidates are kept -- each one wins on SSIM for at least some
(image, quality) combination, including median_k3, which never wins on PSNR
but does win on SSIM.
"""

import numpy as np
import cv2


def restore_gaussian_blur(image, quality, factor=0.2, ksize=0):
    sigma_blur = factor * (100 - quality) / 10.0
    if sigma_blur < 0.1:
        return image.copy()
    k = ksize if ksize > 0 else 0
    return cv2.GaussianBlur(image, (k, k), sigma_blur)


def restore_median(image, ksize=3):
    return cv2.medianBlur(image, ksize)


def restore_bilateral(image, quality, d=9, sigma_space=None):
    sigma_color = float(100 - quality)
    if sigma_space is None:
        sigma_space = sigma_color
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def restore_nlm(image, quality, template_window=7, search_window=21):
    h = float(100 - quality) / 5.0
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, h=h, hColor=h,
            templateWindowSize=template_window, searchWindowSize=search_window,
        )
    return cv2.fastNlMeansDenoising(
        image, h=h, templateWindowSize=template_window, searchWindowSize=search_window,
    )


def restore_boundary_smooth(image, quality, kernel_size=3, alpha=None):
    if alpha is None:
        alpha = min(1.0, (100 - quality) / 80.0)
    if alpha < 1e-3:
        return image.copy()
    result = image.astype(np.float32).copy()
    H, W = result.shape[:2]
    half = kernel_size // 2
    for r in range(8, H, 8):
        r_lo, r_hi = max(0, r - half), min(H, r + half + 1)
        mean = result[r_lo:r_hi].mean(axis=0)
        result[r] = result[r] * (1.0 - alpha) + mean * alpha
    for c in range(8, W, 8):
        c_lo, c_hi = max(0, c - half), min(W, c + half + 1)
        mean = result[:, c_lo:c_hi].mean(axis=1, keepdims=True)
        result[:, c:c + 1] = result[:, c:c + 1] * (1.0 - alpha) + mean * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# (name, fn(image, quality) -> restored_image) -- matches
# experiment_jpeg_per_image.py's CONFIGS exactly, all 7 methods kept.
METHODS = [
    ("gauss_0.2x",   lambda img, q: restore_gaussian_blur(img, q, factor=0.2)),
    ("median_k3",    lambda img, q: restore_median(img, ksize=3)),
    ("bilateral_d3", lambda img, q: restore_bilateral(img, q, d=3)),
    ("bilateral_d5", lambda img, q: restore_bilateral(img, q, d=5)),
    ("bilateral_d9", lambda img, q: restore_bilateral(img, q, d=9)),
    ("nlm_tw5_sw11", lambda img, q: restore_nlm(img, q, template_window=5, search_window=11)),
    ("bndry_k3",     lambda img, q: restore_boundary_smooth(img, q, kernel_size=3)),
]
