# diff gamma levels with n trials restores two methods (inverse and lut), compute psnr, ssim and speed
# large-scale version: 100 trials per gamma level, 15 gamma levels


import json
import pathlib
import random
import subprocess
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from corruptors import add_gamma_corruption
from restorers_gamma import restore_inverse_gamma, restore_lut

_caffeinate = subprocess.Popen(["caffeinate", "-di"])

RAW_DIR = pathlib.Path("/Users/nichita/Documents/research/test_3/data/raw")
OUT_DIR = pathlib.Path("results/gamma_restoration_large")
OUT_FILE = OUT_DIR / "results_gamma_restoration_large.json"
SAMPLES_DIR = OUT_DIR / "samples"

GAMMA_VALUES = [0.1, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4]
N_TRIALS = 100
THUMB_W  = 400


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    channel_axis = 2 if a.ndim == 3 else None
    return float(ssim(a, b, data_range=255, channel_axis=channel_axis))

# one row: original | corrupted | inv_gamma | lut  — each panel THUMB_W wide with label
def _save_comparison(
    original: np.ndarray,
    corrupted: np.ndarray,
    restored_inv: np.ndarray,
    restored_lut: np.ndarray,
    true_gamma: float,
    psnr_inv: float,
    psnr_lut: float,
    ssim_inv: float,
    ssim_lut: float,
    path: pathlib.Path,
) -> None:
    def _thumb(img: np.ndarray, label: str) -> np.ndarray:
        h_orig, w_orig = img.shape[:2]
        h = int(THUMB_W * h_orig / w_orig)
        t = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (THUMB_W, h), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return t

    panels = [
        _thumb(original,     f"original  gamma={true_gamma:.2f}"),
        _thumb(corrupted,    "corrupted"),
        _thumb(restored_inv, f"inv_gamma  {psnr_inv:.2f}dB  {ssim_inv:.4f}"),
        _thumb(restored_lut, f"lut        {psnr_lut:.2f}dB  {ssim_lut:.4f}"),
    ]
    cv2.imwrite(str(path), np.hstack(panels))

# console output
def _print_summary(
    all_psnrs_inv: list[float],
    all_psnrs_lut: list[float],
    all_ssims_inv: list[float],
    all_ssims_lut: list[float],
    all_times_inv: list[float],
    all_times_lut: list[float],
) -> None:
    avg_psnr_inv = float(np.mean(all_psnrs_inv))
    avg_psnr_lut = float(np.mean(all_psnrs_lut))
    avg_ssim_inv = float(np.mean(all_ssims_inv))
    avg_ssim_lut = float(np.mean(all_ssims_lut))
    avg_ms_inv   = float(np.mean(all_times_inv)) * 1000
    avg_ms_lut   = float(np.mean(all_times_lut)) * 1000
    speedup = avg_ms_inv / avg_ms_lut if avg_ms_lut > 0 else float("inf")

    c0, c1, c2 = 22, 16, 16
    sep = f"+{'-' * c0}+{'-' * c1}+{'-' * c2}+"
    def row(label, v1, v2):
        return f"| {label:<{c0-2}} | {v1:^{c1-2}} | {v2:^{c2-2}} |"

    print("\n" + sep)
    print(row("", "inverse_gamma", "lut"))
    print(sep)
    print(row("avg PSNR (dB)", f"{avg_psnr_inv:.2f}", f"{avg_psnr_lut:.2f}"))
    print(row("avg SSIM", f"{avg_ssim_inv:.4f}", f"{avg_ssim_lut:.4f}"))
    print(row("avg time (ms)", f"{avg_ms_inv:.3f}", f"{avg_ms_lut:.3f}"))
    print(row("speedup", "1.0x", f"{speedup:.1f}x faster"))
    print(sep)


def run_experiment(max_images: int | None = None):
    image_paths = sorted(RAW_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {RAW_DIR}")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    trial_paths = random.choices(image_paths, k=N_TRIALS)

    print(f"Images: {len(image_paths)}, trials per gamma: {N_TRIALS}\n")
    print(
        f"{'gamma':>6}  "
        f"{'inv_psnr':>9}  {'inv_ssim':>8}  "
        f"{'lut_psnr':>9}  {'lut_ssim':>8}  "
        f"{'inv_ms':>8}  {'lut_ms':>8}  {'speedup':>9}"
    )
    print("-" * 92)

    results = {}
    all_psnrs_inv, all_psnrs_lut = [], []
    all_ssims_inv, all_ssims_lut = [], []
    all_times_inv, all_times_lut = [], []

    for g_idx, true_gamma in enumerate(GAMMA_VALUES):
        psnrs_inv, psnrs_lut = [], []
        ssims_inv, ssims_lut = [], []
        times_inv, times_lut = [], []
        sample_data = None
        sample_idx = g_idx % N_TRIALS

        for i, path in enumerate(trial_paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            corrupted, _ = add_gamma_corruption(image, gamma=true_gamma)

            t0 = time.perf_counter()
            r_inv = restore_inverse_gamma(corrupted, true_gamma)
            times_inv.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            r_lut = restore_lut(corrupted, true_gamma)
            times_lut.append(time.perf_counter() - t0)

            psnrs_inv.append(cv2.PSNR(image, r_inv))
            psnrs_lut.append(cv2.PSNR(image, r_lut))
            ssims_inv.append(_compute_ssim(image, r_inv))
            ssims_lut.append(_compute_ssim(image, r_lut))

            if i == sample_idx:
                sample_data = (image, corrupted, r_inv, r_lut)

        avg_psnr_inv = float(np.mean(psnrs_inv))
        avg_psnr_lut = float(np.mean(psnrs_lut))
        avg_ssim_inv = float(np.mean(ssims_inv))
        avg_ssim_lut = float(np.mean(ssims_lut))
        ms_inv  = float(np.mean(times_inv)) * 1000
        ms_lut  = float(np.mean(times_lut)) * 1000
        speedup = ms_inv / ms_lut if ms_lut > 0 else float("inf")

        all_psnrs_inv.extend(psnrs_inv);  all_psnrs_lut.extend(psnrs_lut)
        all_ssims_inv.extend(ssims_inv);  all_ssims_lut.extend(ssims_lut)
        all_times_inv.extend(times_inv);  all_times_lut.extend(times_lut)

        print(
            f"{true_gamma:>6.2f}  "
            f"{avg_psnr_inv:>9.2f}  {avg_ssim_inv:>8.4f}  "
            f"{avg_psnr_lut:>9.2f}  {avg_ssim_lut:>8.4f}  "
            f"{ms_inv:>7.3f}ms  {ms_lut:>7.3f}ms  {speedup:>8.1f}x"
        )

        if sample_data is not None:
            sample_path = SAMPLES_DIR / f"gamma_{int(true_gamma * 100):04d}.png"
            _save_comparison(
                *sample_data, true_gamma,
                avg_psnr_inv, avg_psnr_lut,
                avg_ssim_inv, avg_ssim_lut,
                sample_path,
            )

        results[str(true_gamma)] = {
            "true_gamma": true_gamma,
            "inv_gamma": {"mean_psnr": round(avg_psnr_inv, 2), "mean_ssim": round(avg_ssim_inv, 4), "mean_ms": round(ms_inv, 3)},
            "lut":       {"mean_psnr": round(avg_psnr_lut, 2), "mean_ssim": round(avg_ssim_lut, 4), "mean_ms": round(ms_lut, 3)},
        }

    _print_summary(
        all_psnrs_inv, all_psnrs_lut,
        all_ssims_inv, all_ssims_lut,
        all_times_inv, all_times_lut,
    )

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults   -> {OUT_FILE.resolve()}")
    print(f"Samples   -> {SAMPLES_DIR.resolve()}/")
    return results


if __name__ == "__main__":
    try:
        run_experiment()
    finally:
        _caffeinate.terminate()
