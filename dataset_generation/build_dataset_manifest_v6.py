"""Builds the training manifest: 3000 photos x 16 examples (4 per category).

Reads data/method_lookup_{jpeg,noise}.json, whose keys are the image paths
exactly as they appear in the manifest.
"""

import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from methods_jpeg import METHODS as JPEG_METHOD_LIST
from methods_noise import METHODS as NOISE_METHOD_LIST
from methods_gamma import METHODS as GAMMA_METHOD_LIST
from severity_levels import SEVERITY_LEVELS
from severity_words import jpeg_severity_word, noise_severity_word, gamma_severity_phrase

_DATA = os.path.join(os.path.dirname(__file__), "data")
JPEG_LOOKUP = os.path.join(_DATA, "method_lookup_jpeg.json")
NOISE_LOOKUP = os.path.join(_DATA, "method_lookup_noise.json")
OUT_PATH = os.path.join(_DATA, "llava_multi_v6_manifest.json")

SEED = 42
N_PER_PHOTO = {"none": 4, "single": 4, "pair": 4, "triple": 4}  # 16 per photo, 48000 total

TYPES = ["jpeg", "noise", "gamma"]
SUBSETS_PAIR = [("jpeg", "noise"), ("jpeg", "gamma"), ("noise", "gamma")]
SUBSET_TRIPLE = ("jpeg", "noise", "gamma")
TYPE_LABEL = {"jpeg": "JPEG compression", "noise": "Gaussian noise", "gamma": "exposure shift"}


def step_description(dtype, value, rng):
    if dtype == "jpeg":
        return f"{jpeg_severity_word(value, rng)} JPEG compression (quality {value})"
    elif dtype == "noise":
        return f"{noise_severity_word(value, rng)} Gaussian noise (sigma≈{value})"
    elif dtype == "gamma":
        return f"{gamma_severity_phrase(value, rng)} exposure shift (gamma≈{value})"
    raise ValueError(dtype)


def load_per_image_lookup():
    jpeg_data = json.load(open(JPEG_LOOKUP))
    noise_data = json.load(open(NOISE_LOOKUP))
    return jpeg_data, noise_data


def best_method(image_name, dtype, value, jpeg_data, noise_data):
    if dtype == "jpeg":
        return jpeg_data[image_name][str(value)]
    elif dtype == "noise":
        return noise_data[image_name][str(value)]
    elif dtype == "gamma":
        return "lut"
    raise ValueError(dtype)


QUESTIONS = [
    "<image>\nAnalyze this image. Identify any distortions present and give a step-by-step restoration plan.",
    "<image>\nThis image may have more than one problem. Diagnose it and describe the fix, in order.",
    "<image>\nWhat's wrong with this photo, and in what order should I fix it?",
    "<image>\nInspect this image for artifacts. List every issue found and the restoration pipeline to correct them.",
    "<image>\nEvaluate this image for combined distortions and recommend a full restoration sequence.",
    "<image>\nThis photo looks degraded in more than one way — what's going on, and how do I fix all of it?",
    "<image>\nDetermine which distortions affect this image and the correct order to restore it.",
    "<image>\nDiagnose this picture — could be one issue or several — and tell me the fix pipeline.",
    "<image>\nAssess this image for degradation and lay out the restoration steps in order.",
    "<image>\nIdentify all distortions present and propose an ordered restoration pipeline.",
    "<image>\nWhat's wrong with this photo?",
    "<image>\nFix this image.",
    "<image>\nAnything off with this pic? How do I fix it?",
    "<image>\nCheck this photo for problems.",
    "<image>\nHow would you clean this image up?",
    "<image>\nSpot the issues here and tell me how to fix them, step by step.",
    "<image>\nGive me a diagnosis and a repair plan for this photo.",
    "<image>\nWalk me through fixing this image.",
    "<image>\nIs this photo degraded? If so, how many ways, and what's the fix?",
    "<image>\nTell me what's damaged in this photo and how to restore it properly.",
]

DETECTED_TEMPLATES = [
    "Detected distortions ({n}): {types}.",
    "I find {n} distortion{s} present: {types}.",
    "This image has {n} distortion{s}: {types}.",
    "{n} issue{s} detected: {types}.",
    "Distortions found ({n}): {types}.",
    "Analysis shows {n} distortion{s}: {types}.",
    "I can identify {n} distortion{s} here: {types}.",
]
ORDER_TEMPLATES = [
    "Corruption order: {corr}. Restoration order (reverse): {rest}.",
    "Likely applied in this order: {corr}. To restore, reverse it: {rest}.",
    "These were introduced as {corr}; undo them in the opposite order: {rest}.",
    "Corruption sequence: {corr} — so the fix sequence, reversed, is: {rest}.",
    "The distortions were most likely applied as {corr}, so restoration should go {rest}.",
    "Assuming corruption order {corr}, the correct fix order is the reverse: {rest}.",
]
INTROS = [
    "This image shows {n} combined issues: {issues}.",
    "I detect {n} overlapping problems here: {issues}.",
    "This photo has {n} distortions at once: {issues}.",
    "Multiple issues found ({n}): {issues}.",
    "Here's what's affecting this photo ({n}): {issues}.",
    "Breaking it down, there are {n} problems: {issues}.",
]
PIPELINE_INTROS = [
    "Restoration pipeline, in order:",
    "Fix these in this order:",
    "To restore it, follow these steps in order:",
    "Apply this sequence:",
    "Recommended repair sequence:",
    "Here's the step-by-step fix:",
]
STEP_TEMPLATES = [
    "({i}) fix the {desc} using '{method}'",
    "({i}) apply '{method}' to correct the {desc}",
    "({i}) restore from the {desc} with '{method}'",
    "({i}) address the {desc} via '{method}'",
    "({i}) use '{method}' to handle the {desc}",
]
SINGLE_FIX_TEMPLATES = [
    "This photo shows {desc}. Fix it using '{method}'.",
    "This photo shows {desc}. Apply '{method}' to correct it.",
    "This photo shows {desc}. Restore using '{method}'.",
    "Issue: {desc}. Recommended fix: '{method}'.",
    "There's {desc} here. Best fix: '{method}'.",
    "This photo has {desc} — use '{method}' to restore it.",
]
NONE_TEMPLATES = [
    "Detected distortions (0): none.",
    "I find no distortions present.",
    "This image has no distortions.",
    "0 issues detected: none.",
    "No distortions found.",
    "This photo appears clean.",
]
NONE_FIX_TEMPLATES = [
    "No restoration needed.",
    "The image looks clean — no fix required.",
    "No issues to address.",
    "This photo requires no correction.",
    "Nothing to fix here.",
    "It's fine as-is.",
]


def build_multi_answer(rng, corr_order, restore_order, steps):
    n = len(steps)
    types_str = ", ".join(TYPE_LABEL[t] for t in sorted(corr_order))
    detected = rng.choice(DETECTED_TEMPLATES).format(n=n, s="s" if n != 1 else "", types=types_str)
    corr_str = " → ".join(TYPE_LABEL[t] for t in corr_order)
    rest_str = " → ".join(TYPE_LABEL[t] for t in restore_order)
    order_line = rng.choice(ORDER_TEMPLATES).format(corr=corr_str, rest=rest_str)
    issue_descs = [s["desc"] for s in steps]
    intro = rng.choice(INTROS).format(n=n, issues=", ".join(issue_descs))
    pipeline_intro = rng.choice(PIPELINE_INTROS)
    step_strs = [
        rng.choice(STEP_TEMPLATES).format(i=i + 1, desc=s["desc"], method=s["method"])
        for i, s in enumerate(steps)
    ]
    pipeline = f"{intro} {pipeline_intro} " + "; ".join(step_strs) + "."
    return f"{detected} {order_line} {pipeline}"


def build():
    rng = random.Random(SEED)
    jpeg_data, noise_data = load_per_image_lookup()
    image_names = sorted(jpeg_data.keys())
    manifest = []

    # every (severity, order) combination for each pair of types
    pair_candidates = []
    for subset in SUBSETS_PAIR:
        levels = [SEVERITY_LEVELS[t] for t in subset]
        for sev_combo in itertools.product(*levels):
            for order in itertools.permutations(subset):
                pair_candidates.append((subset, sev_combo, order))

    triple_levels = [SEVERITY_LEVELS[t] for t in SUBSET_TRIPLE]
    triple_candidates = []
    for sev_combo in itertools.product(*triple_levels):
        for order in itertools.permutations(SUBSET_TRIPLE):
            triple_candidates.append((SUBSET_TRIPLE, sev_combo, order))

    single_candidates = [(t, v) for t in TYPES for v in SEVERITY_LEVELS[t]]

    for image_name in image_names:
        # id is the filename stem, so photos in subfolders get plain ids too
        base_id = os.path.splitext(os.path.basename(image_name))[0]

        # none
        for _ in range(N_PER_PHOTO["none"]):
            answer = f"{rng.choice(NONE_TEMPLATES)} {rng.choice(NONE_FIX_TEMPLATES)}"
            manifest.append({
                "id": f"{base_id}_v6_none_{rng.randrange(10**6)}",
                "base_id": base_id, "image": image_name,
                "distortions": [],
                "ground_truth": {"distortion_types": [], "corruption_order": [], "restore_order": [], "steps": []},
                "conversations": [
                    {"from": "human", "value": rng.choice(QUESTIONS)},
                    {"from": "gpt", "value": answer},
                ],
            })

        # singles
        chosen_singles = rng.sample(single_candidates, N_PER_PHOTO["single"])
        for dtype, value in chosen_singles:
            method = best_method(image_name, dtype, value, jpeg_data, noise_data)
            desc = step_description(dtype, value, rng)
            answer = f"{rng.choice(DETECTED_TEMPLATES).format(n=1, s='', types=TYPE_LABEL[dtype])} " \
                     f"{rng.choice(SINGLE_FIX_TEMPLATES).format(desc=desc, method=method)}"
            manifest.append({
                "id": f"{base_id}_v6_{dtype}_{value}_{rng.randrange(10**6)}",
                "base_id": base_id, "image": image_name,
                "distortions": [{"type": dtype, "params": {("quality" if dtype == "jpeg" else "sigma" if dtype == "noise" else "gamma"): value}}],
                "ground_truth": {
                    "distortion_types": [dtype], "corruption_order": [dtype], "restore_order": [dtype],
                    "steps": [{"type": dtype, "value": value, "method": method}],
                },
                "conversations": [
                    {"from": "human", "value": rng.choice(QUESTIONS)},
                    {"from": "gpt", "value": answer},
                ],
            })

        # pairs
        chosen_pairs = rng.sample(pair_candidates, N_PER_PHOTO["pair"])
        for subset, sev_combo, corr_order in chosen_pairs:
            manifest.append(_build_combo_entry(rng, image_name, base_id, subset, sev_combo, corr_order, jpeg_data, noise_data, "pair"))

        # triples
        chosen_triples = rng.sample(triple_candidates, N_PER_PHOTO["triple"])
        for subset, sev_combo, corr_order in chosen_triples:
            manifest.append(_build_combo_entry(rng, image_name, base_id, subset, sev_combo, corr_order, jpeg_data, noise_data, "triple"))

    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    by_n = {}
    for e in manifest:
        n = len(e["ground_truth"]["distortion_types"])
        by_n[n] = by_n.get(n, 0) + 1
    print(f"Images: {len(image_names)}")
    print(f"Total entries: {len(manifest)}")
    for n in sorted(by_n):
        print(f"  {n} distortions: {by_n[n]} ({by_n[n]/len(manifest):.1%})")
    print(f"Saved -> {OUT_PATH}")

def _build_combo_entry(rng, image_name, base_id, subset, sev_combo, corr_order, jpeg_data, noise_data, tag):
    severity_label = dict(zip(subset, sev_combo))
    restore_order = tuple(reversed(corr_order))

    distortions = [
        {"type": dtype, "params": {("quality" if dtype == "jpeg" else "sigma" if dtype == "noise" else "gamma"): severity_label[dtype]}}
        for dtype in corr_order
    ]
    steps = []
    for dtype in restore_order:
        value = severity_label[dtype]
        method = best_method(image_name, dtype, value, jpeg_data, noise_data)
        steps.append({"dtype": dtype, "value": value, "desc": step_description(dtype, value, rng), "method": method})

    answer = build_multi_answer(rng, corr_order, restore_order, steps)
    subset_tag = "".join(d[0] for d in subset)
    order_tag = "".join(d[0] for d in corr_order)

    return {
        "id": f"{base_id}_v6_{tag}_{subset_tag}_{order_tag}_{rng.randrange(10**6)}",
        "base_id": base_id, "image": image_name,
        "distortions": distortions,
        "ground_truth": {
            "distortion_types": sorted(subset),
            "corruption_order": list(corr_order),
            "restore_order": list(restore_order),
            "steps": [{"type": s["dtype"], "value": s["value"], "method": s["method"]} for s in steps],
        },
        "conversations": [
            {"from": "human", "value": rng.choice(QUESTIONS)},
            {"from": "gpt", "value": answer},
        ],
    }


if __name__ == "__main__":
    build()
