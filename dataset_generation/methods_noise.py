"""Gaussian noise restoration methods, self-contained.

All 12 candidates are kept -- each wins on SSIM for at least some
(image, sigma) combination.
"""

import cv2


def restore_median(image, ksize=3):
    return cv2.medianBlur(image, ksize)


def restore_bilateral(image, sigma_color, d=9, sigma_space=None):
    if sigma_space is None:
        sigma_space = sigma_color
    return cv2.bilateralFilter(image, d, float(sigma_color), float(sigma_space))


def restore_nlm(image, h, template_window=7, search_window=21):
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, h=float(h), hColor=float(h),
            templateWindowSize=template_window, searchWindowSize=search_window,
        )
    return cv2.fastNlMeansDenoising(
        image, h=float(h), templateWindowSize=template_window, searchWindowSize=search_window,
    )


# (name, fn(image, sigma) -> restored_image) -- matches
# experiment_gaussian_noise_per_image.py's CONFIGS exactly, all 12 kept.
METHODS = [
    ("median_k3",     lambda img, s: restore_median(img, ksize=3)),
    ("median_k5",     lambda img, s: restore_median(img, ksize=5)),
    ("median_k7",     lambda img, s: restore_median(img, ksize=7)),
    ("median_k9",     lambda img, s: restore_median(img, ksize=9)),
    ("median_k11",    lambda img, s: restore_median(img, ksize=11)),
    ("median_k15",    lambda img, s: restore_median(img, ksize=15)),
    ("bilateral_d3",  lambda img, s: restore_bilateral(img, sigma_color=s, d=3)),
    ("bilateral_d5",  lambda img, s: restore_bilateral(img, sigma_color=s, d=5)),
    ("bilateral_d9",  lambda img, s: restore_bilateral(img, sigma_color=s, d=9)),
    ("bilateral_d15", lambda img, s: restore_bilateral(img, sigma_color=s, d=15)),
    ("bilateral_d21", lambda img, s: restore_bilateral(img, sigma_color=s, d=21)),
    ("nlm_tw5_sw11",  lambda img, s: restore_nlm(img, h=s, template_window=5, search_window=11)),
]
