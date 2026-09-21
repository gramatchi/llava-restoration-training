"""Per-image best-restoration-method search for JPEG compression.

For each photo and each quality level, tries a handful of denoise/deblock
filters and records which one gives the best PSNR/SSIM. Used later to build
the ground-truth "correct method" labels for the dataset.

LIMIT_N caps how many images to process (handy for a quick timing test).
N_WORKERS controls parallelism (defaults to all CPUs).
"""
import json
import os
import pathlib
import time
import sys
from multiprocessing import Pool

import cv2
from skimage.metrics import structural_similarity as ssim

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from corruptors import add_jpeg_compression
from restorers_jpeg import (
    restore_gaussian_blur,
    restore_median,
    restore_bilateral,
    restore_nlm,
    restore_boundary_smooth,
)
from severity_levels import JPEG_QUALITY

RAW_DIRS = [
    pathlib.Path("/home/gramatchin/data/raw/extra_images/0002000"),
    pathlib.Path("/home/gramatchin/data/raw/extra_images/0003000"),
]
OUT_DIR = HERE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_VALUES = JPEG_QUALITY

CONFIGS = [
    ("gauss_0.2x",   lambda img, q: restore_gaussian_blur(img, q, factor=0.2)),
    ("median_k3",    lambda img, q: restore_median(img, ksize=3)),
    ("bilateral_d3", lambda img, q: restore_bilateral(img, q, d=3)),
    ("bilateral_d5", lambda img, q: restore_bilateral(img, q, d=5)),
    ("bilateral_d9", lambda img, q: restore_bilateral(img, q, d=9)),
    ("nlm_tw5_sw11", lambda img, q: restore_nlm(img, q, template_window=5, search_window=11)),
    ("bndry_k3",     lambda img, q: restore_boundary_smooth(img, q, kernel_size=3)),
]


def _ssim(a, b):
    return float(ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


def _init_worker():
    # cv2 spawns its own thread pool by default, which fights with
    # multiprocessing for cores -- pin each worker to 1 thread.
    cv2.setNumThreads(1)


def _process_path(path_str):
    # compress at each quality level, try every candidate restorer, keep
    # whichever wins on PSNR and whichever wins on SSIM (usually the same).
    path = pathlib.Path(path_str)
    image = cv2.imread(str(path))
    if image is None:
        return {"image": path.name, "error": "cannot read"}
    by_quality = {}
    for quality in QUALITY_VALUES:
        compressed, _ = add_jpeg_compression(image, quality=quality)
        scores = {}
        for name, fn in CONFIGS:
            restored = fn(compressed, quality)
            scores[name] = (
                round(float(cv2.PSNR(image, restored)), 4),
                round(_ssim(image, restored), 6),
            )
        best_psnr = max(scores, key=lambda n: scores[n][0])
        best_ssim = max(scores, key=lambda n: scores[n][1])
        by_quality[str(quality)] = {
            "best_psnr": {"method": best_psnr, "psnr": scores[best_psnr][0]},
            "best_ssim": {"method": best_ssim, "ssim": scores[best_ssim][1]},
        }
    return {"image": path.name, "by_quality": by_quality}


def run():
    # resumable: skips anything already in per_image.json, so a timed-out
    # job can just be resubmitted.
    image_paths = []
    for d in RAW_DIRS:
        image_paths.extend(sorted(d.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {RAW_DIRS}")

    limit_n = os.environ.get("LIMIT_N")
    if limit_n:
        image_paths = image_paths[: int(limit_n)]

    n_workers = int(os.environ.get("N_WORKERS", os.cpu_count() or 1))

    out_file = OUT_DIR / "new_2000_per_image_jpeg.json"
    SAVE_EVERY = 20

    if out_file.exists():
        results = json.loads(out_file.read_text())
        done_names = {r["image"] for r in results}
        image_paths = [p for p in image_paths if p.name not in done_names]
        print(f"Resuming: {len(results)} already done, {len(image_paths)} remaining\n")
    else:
        results = []

    n_images = len(image_paths)
    total_ops = n_images * len(QUALITY_VALUES) * len(CONFIGS)
    print(f"Images to process: {n_images}  |  quality levels: {len(QUALITY_VALUES)}  |  "
          f"configs: {len(CONFIGS)}  |  workers: {n_workers}")
    print(f"Total restoration runs: {total_ops:,}\n")

    if not image_paths:
        print("Nothing to do.")
        return

    t_start = time.perf_counter()
    total = len(results) + len(image_paths)
    path_strs = [str(p) for p in image_paths]

    with Pool(n_workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_path, path_strs, chunksize=1)):
            if "error" in res:
                print(f"  [skip] cannot read {res['image']}")
                continue
            results.append(res)
            done = len(results)
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta_s = (n_images - i - 1) / rate if rate > 0 else 0

            if i == 0:
                print(f"  Done 1 of {n_images} (this run) — {elapsed:.1f}s wall so far, "
                      f"effective rate {rate*n_workers:.3f} img/s/worker-equiv\n")
            if done % SAVE_EVERY == 0 or i == n_images - 1:
                out_file.write_text(json.dumps(results, indent=2))
                print(f"  Done {done} of {total}  |  remaining ~{eta_s/60:.1f} min  (saved)")

    out_file.write_text(json.dumps(results, indent=2))
    total_time = time.perf_counter() - t_start
    print(f"\nDone in {total_time/60:.1f} min")
    print(f"Results -> {out_file.resolve()}")


if __name__ == "__main__":
    run()
