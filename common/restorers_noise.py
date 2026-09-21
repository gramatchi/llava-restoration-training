# Gaussian noise restoration functions (sigma is assumed known).

import numpy as np
import cv2


def restore_gaussian_blur(
    image: np.ndarray,
    sigma_blur: float,
    ksize: int = 0,       # 0 = auto-computed from sigma_blur (capped at max_ksize)
    max_ksize: int = 31,  # cap so huge sigma doesn't blow up runtime (matches bilateral/NLM window scale)
) -> np.ndarray:
    if ksize > 0:
        k = ksize
    else:
        k = int(round(sigma_blur * 6 + 1)) | 1
        k = min(k, max_ksize)
    return cv2.GaussianBlur(image, (k, k), sigma_blur)


def restore_median(
    image: np.ndarray,
    ksize: int = 3,  # must be odd: 3, 5, 7, ...
) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def restore_bilateral(
    image: np.ndarray,
    sigma_color: float,        # range of colors to mix (set to noise sigma)
    d: int = 9,                # pixel neighbourhood diameter
    sigma_space: float | None = None,  # spatial extent; defaults to sigma_color
) -> np.ndarray:
    if sigma_space is None:
        sigma_space = sigma_color
    return cv2.bilateralFilter(image, d, float(sigma_color), float(sigma_space))


def restore_nlm(
    image: np.ndarray,
    h: float,                   # filter strength ≈ noise sigma
    template_window: int = 7,   # patch size for similarity (odd, e.g. 5, 7, 11)
    search_window: int = 21,    # area to search for similar patches (odd, e.g. 11, 21, 35)
) -> np.ndarray:
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image,
            h=float(h),
            hColor=float(h),
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )
    return cv2.fastNlMeansDenoising(
        image,
        h=float(h),
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )
