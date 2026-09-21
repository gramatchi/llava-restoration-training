"""Gamma/exposure restoration, self-contained.

Only 'lut' is a real candidate -- it's the exact inverse gamma curve, just
precomputed as a lookup table for speed. Kept as a single-entry list so the
interface matches methods_jpeg.py/methods_noise.py.
"""

import numpy as np
import cv2


def restore_lut(image, gamma):
    lut = np.array(
        [(i / 255.0) ** (1.0 / gamma) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


METHODS = [
    ("lut", lambda img, g: restore_lut(img, g)),
]
