"""
gamma has no per-image search beacause always lut 

jpeg, gamma, noise - ten parameters 
"""

JPEG_QUALITY = [1, 5, 10, 15, 25, 40, 60, 70, 85, 95]
NOISE_SIGMA = [1, 10, 20, 40, 50, 80, 100, 150, 180, 260]
GAMMA = [0.25, 0.33, 0.44, 0.58, 0.76, 1.32, 1.74, 2.30, 3.03, 4.0]

SEVERITY_LEVELS = {"jpeg": JPEG_QUALITY, "noise": NOISE_SIGMA, "gamma": GAMMA}
