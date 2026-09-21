"""
one distinct descriptive phrase per severity level
(10 for jpeg/noise, 5 magnitude words x 2 directions for gamma), each with
several synonym variants for diversity (same principle as the answer-
template pools: don't repeat one rigid phrase every time)

Goal: someone who doesn't know
what the raw numbers mean should still understand roughly how bad the
distortion is from the wording alone. 

Order matches severity_levels.py exactly: index 0 = mildest, index 9 =
strongest

corrected for jpeg (low quality number = high severity, so index 0 there
corresponds to quality=95, the highest/mildest quality value).
"""

from severity_levels import JPEG_QUALITY, NOISE_SIGMA, GAMMA

# mildest -> strongest, 10 entries each, 2-3 synonym phrasings per entry
JPEG_WORDS = [
    ["barely noticeable — might not even be a real artifact", "hard to notice, may not be genuine", "so subtle it might not be real distortion"],
    ["very slight", "very mild", "quite faint"],
    ["slight", "faint", "light"],
    ["mild", "somewhat noticeable", "light-to-moderate"],
    ["noticeable", "clearly visible", "apparent"],
    ["moderate", "medium", "fairly noticeable"],
    ["fairly strong", "quite pronounced", "considerable"],
    ["strong", "heavy", "pronounced"],
    ["very strong", "very heavy", "intense"],
    ["severe", "extreme", "very severe"],
]
NOISE_WORDS = [
    ["barely noticeable — might not even be present", "hard to notice, may not be genuine", "so subtle it might not be real noise"],
    ["very slight", "very mild", "quite faint"],
    ["slight", "faint", "light"],
    ["mild", "somewhat noticeable", "light-to-moderate"],
    ["noticeable", "clearly visible", "apparent"],
    ["moderate", "medium", "fairly noticeable"],
    ["fairly strong", "quite pronounced", "considerable"],
    ["strong", "heavy", "pronounced"],
    ["very strong", "very heavy", "intense"],
    ["severe", "extreme", "very severe"],
]
# 5 magnitude words, mildest (closest to gamma=1.0) -> strongest, applied to
# either direction (bright or dark); the mildest entry has no direction word
# (not meaningful this close to 1.0), so it's phrased standalone.
GAMMA_MAGNITUDE_WORDS = [
    ["barely noticeable — exposure seems about right", "hard to tell, exposure looks close to normal", "so subtle the exposure may be fine"],
    ["slightly", "a bit", "faintly"],
    ["moderately", "noticeably", "fairly"],
    ["quite", "clearly", "distinctly"],
    ["very", "extremely", "severely"],
]

# from digit to word 
def jpeg_severity_word(quality, rng):
    idx_by_value = sorted(JPEG_QUALITY, reverse=True).index(quality)
    return rng.choice(JPEG_WORDS[idx_by_value])


def noise_severity_word(sigma, rng):
    idx_by_value = sorted(NOISE_SIGMA).index(sigma)
    return rng.choice(NOISE_WORDS[idx_by_value])


def gamma_severity_phrase(gamma, rng):
    direction = "dark" if gamma > 1.0 else "bright"
    darks = sorted(g for g in GAMMA if g > 1.0)
    brights = sorted((1.0 / g for g in GAMMA if g < 1.0))
    side = darks if gamma > 1.0 else brights
    side_ratio = gamma if gamma > 1.0 else 1.0 / gamma
    idx = min(range(len(side)), key=lambda i: abs(side[i] - side_ratio))
    magnitude = rng.choice(GAMMA_MAGNITUDE_WORDS[idx])
    if idx == 0:
        return magnitude
    return f"{magnitude} {direction}"
