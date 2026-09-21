# gamma restoration functions (we know gamma)

import numpy as np
import cv2


def restore_inverse_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    # mathematical inverse
    norm = image.astype(np.float32) / 255.0
    restored = np.power(np.clip(norm, 0.0, 1.0), 1.0 / gamma)
    return (restored * 255.0).astype(np.uint8)


def restore_lut(image: np.ndarray, gamma: float) -> np.ndarray:
    # Look-Up Table(faster than per-pixel, creates array of 256 values than pick one)
    lut = np.array(
        [(i / 255.0) ** (1.0 / gamma) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)
