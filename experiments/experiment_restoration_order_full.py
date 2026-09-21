"""Restoration-order experiment.

For every pair and the triple of {jpeg, noise, gamma}, at low/medium/high
severity per type and every corruption order, this searches all restoration
orders and, for each step, all candidate methods. It records which order gives
the best PSNR and whether that is the reverse of the corruption order. The
method search is included because the best method for a single distortion does
not necessarily stay the best once other distortions are stacked on top.

Sampled on 15 photos. Resumable: cases already in the output json are skipped.
MAX_NEW_CASES limits a run to a few new cases (quick timing check).
CPU only, run it with run_order_experiment.sh.
"""

import sys
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import random

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))
from corruptors import add_jpeg_compression, add_gaussian_noise, add_gamma_corruption
from restorers_jpeg import (
    restore_gaussian_blur as jpeg_restore_gaussian_blur,
    restore_median as jpeg_restore_median,
    restore_bilateral as jpeg_restore_bilateral,
    restore_nlm as jpeg_restore_nlm,
    restore_boundary_smooth as jpeg_restore_boundary,
)
from restorers_noise import (
    restore_median as noise_restore_median,
    restore_bilateral as noise_restore_bilateral,
    restore_nlm as noise_restore_nlm,
)
from restorers_gamma import restore_lut as gamma_restore_lut

RAW_DIR = pathlib.Path("/home/gramatchin/data/raw")
OUT_PATH = pathlib.Path(__file__).resolve().parent / "results_restoration_order_full.json"
N_SAMPLES = 15
SEED = 42
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
MAX_NEW_CASES = int(os.environ.get("MAX_NEW_CASES", "0"))  # 0 = no limit

TYPES = ["jpeg", "noise", "gamma"]

SEVERITY_LEVELS = {
    "jpeg": {"low": 70, "medium": 30, "high": 5},       # quality: higher = milder
    "noise": {"low": 15, "medium": 50, "high": 150},    # sigma: higher = stronger
    "gamma": {"low": 1.2, "medium": 1.8, "high": 3.0},  # darkening side only, kept simple
}

# candidate methods per type. gamma has only one (the exact inverse), so no search there.
JPEG_METHODS = {
    "gauss_0.2x": lambda img, q: jpeg_restore_gaussian_blur(img, q, factor=0.2),
    "median_k3": lambda img, q: jpeg_restore_median(img, ksize=3),
    "bilateral_d3": lambda img, q: jpeg_restore_bilateral(img, q, d=3),
    "bilateral_d5": lambda img, q: jpeg_restore_bilateral(img, q, d=5),
    "bilateral_d9": lambda img, q: jpeg_restore_bilateral(img, q, d=9),
    "nlm_tw5_sw11": lambda img, q: jpeg_restore_nlm(img, q, template_window=5, search_window=11),
    "bndry_k3": lambda img, q: jpeg_restore_boundary(img, q, kernel_size=3),
}
NOISE_METHODS = {
    "median_k3": lambda img, s: noise_restore_median(img, ksize=3),
    "median_k5": lambda img, s: noise_restore_median(img, ksize=5),
    "median_k7": lambda img, s: noise_restore_median(img, ksize=7),
    "bilateral_d3": lambda img, s: noise_restore_bilateral(img, sigma_color=s, d=3),
    "bilateral_d5": lambda img, s: noise_restore_bilateral(img, sigma_color=s, d=5),
    "bilateral_d9": lambda img, s: noise_restore_bilateral(img, sigma_color=s, d=9),
    "nlm_tw5_sw11": lambda img, s: noise_restore_nlm(img, h=s, template_window=5, search_window=11),
}
GAMMA_METHODS = {
    "lut": lambda img, g: gamma_restore_lut(img, gamma=g),
}
METHODS_BY_TYPE = {"jpeg": JPEG_METHODS, "noise": NOISE_METHODS, "gamma": GAMMA_METHODS}


def corrupt_one(image, dtype, severity_value):
    if dtype == "jpeg":
        out, _ = add_jpeg_compression(image, quality=severity_value)
    elif dtype == "noise":
        cv2.setRNGSeed(0)
        out, _ = add_gaussian_noise(image, sigma=severity_value)
    elif dtype == "gamma":
        out, _ = add_gamma_corruption(image, gamma=severity_value)
    else:
        raise ValueError(dtype)
    return out


def corrupt_sequence(image, order, severities):
    out = image
    for dtype in order:
        out = corrupt_one(out, dtype, severities[dtype])
    return out


def score(original, restored):
    psnr = cv2.PSNR(original, restored)
    s = ssim(original, restored, data_range=255, channel_axis=2 if original.ndim == 3 else None)
    return psnr, s


def evaluate_method_combo(task):
    # runs in a worker process; relies on fork so the method tables are inherited
    restore_order, method_combo, corr_order, severities, originals = task
    psnrs, ssims = [], []
    for original in originals:
        corrupted = corrupt_sequence(original, corr_order, severities)
        restored = corrupted
        for dtype, method_name in zip(restore_order, method_combo):
            fn = METHODS_BY_TYPE[dtype][method_name]
            restored = fn(restored, severities[dtype])
        p, s = score(original, restored)
        psnrs.append(p)
        ssims.append(s)
    return restore_order, method_combo, float(np.mean(psnrs)), float(np.mean(ssims))


class StopEarly(Exception):
    """Stops the nested loops once MAX_NEW_CASES is reached."""


def case_key(subset, severity_label, corr_order):
    return (tuple(subset), tuple(sorted(severity_label.items())), tuple(corr_order))


def load_existing_results():
    if not OUT_PATH.exists():
        return [], set()
    data = json.loads(OUT_PATH.read_text())
    existing = data.get("cases", [])
    done_keys = {
        case_key(c["subset"], c["severity"], c["corruption_order"]) for c in existing
    }
    print(f"Resuming: found {len(existing)} already-completed cases, skipping them.", flush=True)
    return existing, done_keys


def search_all_cases(pool, subsets, severity_names, results, done_keys):
    n_new_cases = 0
    for subset in subsets:
        corruption_orders = list(itertools.permutations(subset))
        restoration_orders = list(itertools.permutations(subset))
        severity_combos = list(itertools.product(severity_names, repeat=len(subset)))

        print(f"=== Subset {subset}: {len(severity_combos)} severity combos x "
              f"{len(corruption_orders)} corruption orders ===", flush=True)

        for sev_combo in severity_combos:
            severities = {dtype: SEVERITY_LEVELS[dtype][sev_name] for dtype, sev_name in zip(subset, sev_combo)}
            severity_label = {dtype: sev_name for dtype, sev_name in zip(subset, sev_combo)}

            for corr_order in corruption_orders:
                key = case_key(subset, severity_label, corr_order)
                if key in done_keys:
                    continue

                reverse_order = tuple(reversed(corr_order))

                tasks = []
                for restore_order in restoration_orders:
                    method_vocabs = [list(METHODS_BY_TYPE[dtype].keys()) for dtype in restore_order]
                    for method_combo in itertools.product(*method_vocabs):
                        tasks.append((restore_order, method_combo, corr_order, severities, originals_ref[0]))

                best_per_order = {}  # restore_order -> (mean_psnr, mean_ssim, method_combo)
                for restore_order, method_combo, mean_psnr, mean_ssim in pool.imap_unordered(evaluate_method_combo, tasks, chunksize=4):
                    cur = best_per_order.get(restore_order)
                    if cur is None or mean_psnr > cur[0]:
                        best_per_order[restore_order] = (mean_psnr, mean_ssim, method_combo)

                best_order = max(best_per_order, key=lambda ro: best_per_order[ro][0])
                reverse_is_best = best_order == reverse_order

                case = {
                    "subset": list(subset),
                    "severity": severity_label,
                    "corruption_order": list(corr_order),
                    "correct_reverse_restore_order": list(reverse_order),
                    "results_per_restore_order": {
                        "->".join(ro): {
                            "best_methods": dict(zip(ro, v[2])),
                            "mean_psnr": v[0], "mean_ssim": v[1],
                        } for ro, v in best_per_order.items()
                    },
                    "best_order_overall": list(best_order),
                    "reverse_is_best": bool(reverse_is_best),
                }
                results.append(case)
                done_keys.add(key)
                print(f"  sev={severity_label} corrupt={corr_order} reverse={reverse_order} "
                      f"best={best_order} reverse_is_best={reverse_is_best} "
                      f"(best PSNR={best_per_order[best_order][0]:.2f}dB)", flush=True)

                OUT_PATH.write_text(json.dumps({"n_samples": len(originals_ref[0]), "cases": results}, indent=2))

                n_new_cases += 1
                if MAX_NEW_CASES and n_new_cases >= MAX_NEW_CASES:
                    print(f"\nMAX_NEW_CASES={MAX_NEW_CASES} reached, stopping.", flush=True)
                    raise StopEarly()


originals_ref = [None]  # filled in run(), read in search_all_cases()


def run():
    image_paths = sorted(RAW_DIR.glob("*.png"))
    rng = random.Random(SEED)
    sample_paths = rng.sample(image_paths, min(N_SAMPLES, len(image_paths)))
    originals = [cv2.imread(str(p)) for p in sample_paths]
    originals = [o for o in originals if o is not None]
    originals_ref[0] = originals
    print(f"Testing on {len(originals)} images, {N_WORKERS} worker processes\n", flush=True)

    subsets = list(itertools.combinations(TYPES, 2)) + [tuple(TYPES)]
    severity_names = list(next(iter(SEVERITY_LEVELS.values())).keys())  # ["low","medium","high"]

    results, done_keys = load_existing_results()

    with mp.Pool(N_WORKERS) as pool:
        try:
            search_all_cases(pool, subsets, severity_names, results, done_keys)
        except StopEarly:
            pass

    n_cases = len(results)
    n_reverse_best = sum(r["reverse_is_best"] for r in results)
    print(f"\n=== SUMMARY: reverse order was best in {n_reverse_best}/{n_cases} cases overall ===", flush=True)

    print("\n=== breakdown by whether severities were uniform vs mixed ===", flush=True)
    uniform = [r for r in results if len(set(r["severity"].values())) == 1]
    mixed = [r for r in results if len(set(r["severity"].values())) > 1]
    if uniform:
        print(f"  uniform severity cases: reverse best in {sum(r['reverse_is_best'] for r in uniform)}/{len(uniform)}", flush=True)
    if mixed:
        print(f"  mixed severity cases:   reverse best in {sum(r['reverse_is_best'] for r in mixed)}/{len(mixed)}", flush=True)

    print(f"\nResults saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    run()
