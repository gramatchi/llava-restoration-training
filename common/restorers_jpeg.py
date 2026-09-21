# jpeg artifact restoration functions (quality is assumed known) quality: 1-100 (lower = heavier compression, more artifacts).

import numpy as np
import cv2


def restore_gaussian_blur(
    image: np.ndarray,
    quality: int,
    factor: float = 0.5,   # sigma_blur = factor * (100 - quality) / 10
    ksize: int = 0,         # 0 = auto from sigma_blur
) -> np.ndarray:
    sigma_blur = factor * (100 - quality) / 10.0
    if sigma_blur < 0.1:
        return image.copy()
    k = ksize if ksize > 0 else 0
    return cv2.GaussianBlur(image, (k, k), sigma_blur)


def restore_median(
    image: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def restore_bilateral(
    image: np.ndarray,
    quality: int,
    d: int = 9,
    sigma_space: float | None = None,  # defaults to sigma_color
) -> np.ndarray:
    sigma_color = float(100 - quality)
    if sigma_space is None:
        sigma_space = sigma_color
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def restore_nlm(
    image: np.ndarray,
    quality: int,
    template_window: int = 7,   # patch size for similarity comparison (odd)
    search_window: int = 21,    # area to search for similar patches (odd)
) -> np.ndarray:
    h = float(100 - quality) / 5.0  # filter strength proportional to artifact level
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image,
            h=h,
            hColor=h,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )
    return cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


def restore_boundary_smooth(
    image: np.ndarray,
    quality: int,
    kernel_size: int = 3,   # how many rows/cols around the boundary to average
    alpha: float | None = None,  # blend strength; None = derived from quality
) -> np.ndarray:
    # Smooths only at 8x8 JPEG block boundaries, leaves interior pixels untouched.
    if alpha is None:
        alpha = min(1.0, (100 - quality) / 80.0)
    if alpha < 1e-3:
        return image.copy()

    result = image.astype(np.float32).copy()
    H, W = result.shape[:2]
    half = kernel_size // 2

    # horizontal block boundaries → smooth vertically
    for r in range(8, H, 8):
        r_lo = max(0, r - half)
        r_hi = min(H, r + half + 1)
        mean = result[r_lo:r_hi].mean(axis=0)
        result[r] = result[r] * (1.0 - alpha) + mean * alpha

    # vertical block boundaries → smooth horizontally
    for c in range(8, W, 8):
        c_lo = max(0, c - half)
        c_hi = min(W, c + half + 1)
        mean = result[:, c_lo:c_hi].mean(axis=1, keepdims=True)
        result[:, c:c + 1] = result[:, c:c + 1] * (1.0 - alpha) + mean * alpha

    return np.clip(result, 0, 255).astype(np.uint8)
