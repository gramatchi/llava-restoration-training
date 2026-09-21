# Image corruption functions.

import numpy as np
import cv2


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float | None = None,
    mean: float = 0.0,
    clip: bool = True,
) -> tuple[np.ndarray, dict]:

    # sigma (0,255)
    # clip at high sigma, clipping shrinks apparent variance
    
    if sigma is None:
        sigma = float(np.random.uniform(1.0, 200.0))

    noise = np.zeros(image.shape, dtype=np.float32)
    cv2.randn(noise, mean, sigma) # cv noise func
    noisy = image.astype(np.float32) + noise
    # cut if out the border
    if clip:
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, {"sigma": sigma, "mean": mean}


def add_jpeg_compression(
    image: np.ndarray,
    quality: int | None = None,
) -> tuple[np.ndarray, dict]:
    
    # quality: 1–100 (lower = more compression/artifacts)

    if quality is None:
        quality = int(np.random.randint(10, 96))

    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]) #binary buffer
    
    #decoder
    flag = cv2.IMREAD_COLOR if image.ndim == 3 else cv2.IMREAD_GRAYSCALE
    compressed = cv2.imdecode(encoded, flag)
    return compressed, {"quality": quality}


def add_gamma_corruption(
    image: np.ndarray,
    gamma: float | None = None,
) -> tuple[np.ndarray, dict]:

    # gamma > 1 darkens (contraction), gamma < 1 brightens (expansion)

    if gamma is None:
        gamma = float(np.random.uniform(0.25, 4.0))
    #normalization 
    norm = image.astype(np.float32) / 255.0
    # on pixels
    result = (np.power(np.clip(norm, 0.0, 1.0), gamma) * 255.0).astype(np.uint8)
    return result, {"gamma": gamma}
