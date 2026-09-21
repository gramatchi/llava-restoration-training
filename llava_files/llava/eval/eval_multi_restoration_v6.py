"""Evaluates the model on the multi-distortion task.

Parses each answer (detected types, restoration order, methods, values) and scores
it against the ground truth stored in the manifest.
"""

import argparse
import json
import math
import os
import re
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

sys.path.insert(0, "/nfsd/lttm4/tesisti/gramatchi/v6")
from severity_levels import JPEG_QUALITY, NOISE_SIGMA, GAMMA

JPEG_METHODS = ["gauss_0.2x", "median_k3", "bilateral_d3", "bilateral_d5", "bilateral_d9", "nlm_tw5_sw11", "bndry_k3"]
NOISE_METHODS = [
    "median_k3", "median_k5", "median_k7", "median_k9", "median_k11", "median_k15",
    "bilateral_d3", "bilateral_d5", "bilateral_d9", "bilateral_d15", "bilateral_d21",
    "nlm_tw5_sw11",
]
GAMMA_METHODS = ["lut"]
ALL_METHODS = sorted(set(JPEG_METHODS) | set(NOISE_METHODS) | set(GAMMA_METHODS), key=len, reverse=True)
METHOD_RE = re.compile(r'\b(' + '|'.join(re.escape(m) for m in ALL_METHODS) + r')\b') # looking for a method

# numbers written next to quality / sigma / gamma
QUALITY_RE = re.compile(r'quality\D{0,10}?(\d+)', re.IGNORECASE)
SIGMA_RE = re.compile(r'sigma\D{0,10}?([0-9]*\.?[0-9]+)', re.IGNORECASE)
GAMMA_RE = re.compile(r'gamma\D{0,5}?([0-9]*\.?[0-9]+)', re.IGNORECASE)
VALUE_RE_BY_TYPE = {"jpeg": QUALITY_RE, "noise": SIGMA_RE, "gamma": GAMMA_RE}

TYPE_RE_BY_TYPE = {
    "jpeg": re.compile(r'\bJPEG\b', re.IGNORECASE),
    "noise": re.compile(r'\bnoise\b', re.IGNORECASE),
    "gamma": re.compile(r'\bgamma\b', re.IGNORECASE),
}

PHRASE_TO_TYPE = {"JPEG compression": "jpeg", "Gaussian noise": "noise", "exposure shift": "gamma"}
TYPE_PHRASE_ALT = "|".join(re.escape(p) for p in PHRASE_TO_TYPE)
DETECTED_RE = re.compile(rf':\s*((?:{TYPE_PHRASE_ALT})(?:\s*,\s*(?:{TYPE_PHRASE_ALT}))*)\s*\.') # e.g. Detected distortions (N): jpeg, noise, gamma.
CHAIN_RE = re.compile(rf'(?:{TYPE_PHRASE_ALT})(?:\s*→\s*(?:{TYPE_PHRASE_ALT})){{1,2}}') # order X → Y → Z

# family = name before the first underscore, e.g. bilateral_d3 -> bilateral
def method_family(name):
    return name.split('_')[0] if name else None

# jpeg qualities [95,85,...,1] and noise sigmas map to indices 0-9
_JPEG_ORDER = sorted(JPEG_QUALITY, reverse=True)   # index 0 = mildest (highest quality)
_NOISE_ORDER = sorted(NOISE_SIGMA)                  # index 0 = mildest (lowest sigma)


# gamma index: distance from 1.0 on its own side (darkening or brightening)
def _gamma_sev_idx(g):
    darks = sorted(x for x in GAMMA if x > 1.0)
    brights = sorted((1.0 / x for x in GAMMA if x < 1.0))
    side = darks if g > 1.0 else brights
    ratio = g if g > 1.0 else 1.0 / g
    return min(range(len(side)), key=lambda i: abs(side[i] - ratio))

# snaps a value to the nearest defined level and returns its index
def severity_index(dtype, value):
    if dtype == "jpeg":
        return min(range(len(_JPEG_ORDER)), key=lambda i: abs(_JPEG_ORDER[i] - value))
    if dtype == "noise":
        return min(range(len(_NOISE_ORDER)), key=lambda i: abs(_NOISE_ORDER[i] - value))
    if dtype == "gamma":
        return _gamma_sev_idx(value)
    raise ValueError(dtype)

# distance in levels between predicted and true value
def _value_index_distance(dtype, pred, true):
    if pred is None or (dtype == "gamma" and pred <= 0):
        return None
    return abs(severity_index(dtype, pred) - severity_index(dtype, true))

# value is correct if it lands on the exact level / within one level (t=t binds the type)
VALUE_CORRECT_BY_TYPE = {
    t: (lambda pred, true, t=t: (_value_index_distance(t, pred, true) == 0))
    for t in ("jpeg", "noise", "gamma")
}
VALUE_CORRECT_WITHIN1_BY_TYPE = {
    t: (lambda pred, true, t=t: (lambda d: d is not None and d <= 1)(_value_index_distance(t, pred, true)))
    for t in ("jpeg", "noise", "gamma")
}

# if there is no clear "X → Y → Z" chain, take the order in which the type words first appear
def _order_of_appearance_fallback(text):
    type_hits = []
    for t, regex in TYPE_RE_BY_TYPE.items():
        m = regex.search(text)
        if m:
            type_hits.append((m.start(), t))
    type_hits.sort()
    return [t for _, t in type_hits]


def parse_pipeline_answer(text):
    # detected types from the "Detected distortions (N): ..." line
    m = DETECTED_RE.search(text)
    pred_types_detected = None
    if m:
        phrases = [p.strip() for p in m.group(1).split(',')]
        pred_types_detected = [PHRASE_TO_TYPE[p] for p in phrases if p in PHRASE_TO_TYPE]


    # order chains "A → B → C": normally two (corruption, then restoration), we want the second
    chains = CHAIN_RE.findall(text)
    pred_order = None
    if len(chains) >= 2:
        pred_order = [PHRASE_TO_TYPE[p.strip()] for p in chains[1].split('→') if p.strip() in PHRASE_TO_TYPE]
    elif len(chains) == 1:
        pred_order = [PHRASE_TO_TYPE[p.strip()] for p in chains[0].split('→') if p.strip() in PHRASE_TO_TYPE]

    # fallbacks when the answer is malformed
    if not pred_order:
        pred_order = _order_of_appearance_fallback(text)
    if not pred_types_detected:
        pred_types_detected = pred_order

    # does the "Detected distortions" line agree with the order chain?
    detected_order_agreement = set(pred_types_detected) == set(pred_order)

    # methods are listed in restoration order, so match them to pred_order by position
    method_mentions = [m.group(1) for m in METHOD_RE.finditer(text)]
    pred_method_by_type = dict(zip(pred_order, method_mentions))

    # numeric values (quality / sigma / gamma)
    pred_value_by_type = {}
    for t, regex in VALUE_RE_BY_TYPE.items():
        m = regex.search(text)
        if m:
            pred_value_by_type[t] = float(m.group(1))

    return pred_order, pred_types_detected, pred_method_by_type, pred_value_by_type, detected_order_agreement

# same on-the-fly corruption as apply_distortions() in train.py, in corruption order
def corrupt_for_eval(image, entry):
    sys.path.insert(0, os.environ.get("RESTORATION_COMMON_DIR", "/nfsd/lttm4/tesisti/gramatchi/training_release/common"))
    import zlib
    import numpy as np
    import cv2
    from corruptors import add_jpeg_compression, add_gaussian_noise, add_gamma_corruption

    image_np = np.ascontiguousarray(np.array(image)[:, :, ::-1])
    for d in entry["distortions"]:
        dtype, params = d["type"], d["params"]
        if dtype == "jpeg":
            image_np, _ = add_jpeg_compression(image_np, quality=params["quality"])
        elif dtype == "noise":
            seed = zlib.crc32(f"{entry['base_id']}_{params['sigma']}".encode()) & 0x7fffffff
            cv2.setRNGSeed(seed)
            image_np, _ = add_gaussian_noise(image_np, sigma=params["sigma"])
        elif dtype == "gamma":
            image_np, _ = add_gamma_corruption(image_np, gamma=params["gamma"])
    return Image.fromarray(np.ascontiguousarray(image_np[:, :, ::-1]))

# builds the prompt, distorts the photo and prepares the model inputs
class PipelineDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        entry = self.questions[index]
        qs = entry["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, entry["image"])).convert('RGB')
        image = corrupt_for_eval(image, entry)
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor, image.size

# batch_size is 1 in practice
def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    model.eval()

    questions = json.load(open(args.question_file))
    if args.limit is not None:
        questions = questions[:args.limit]

    # resume: skip questions that are already in the answers file and append to it
    already_answered = []
    if os.path.exists(args.answers_file):
        with open(args.answers_file) as f:
            already_answered = [json.loads(line) for line in f if line.strip()]
    n_done = len(already_answered)
    if n_done > 0:
        print(f"Resuming: {n_done} answers already on disk, skipping to question {n_done}.")
    questions_remaining = questions[n_done:]

    dataset = PipelineDataset(questions_remaining, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "a" if n_done > 0 else "w")

    rows = list(already_answered)

    for (input_ids, image_tensor, image_sizes), entry in tqdm(zip(loader, questions_remaining), total=len(questions_remaining)):
        gt = entry["ground_truth"]
        true_types = set(gt["distortion_types"])
        true_order = gt["restore_order"]
        true_steps = {s["type"]: s for s in gt["steps"]}

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=model.dtype, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        pred_order, pred_types_list, pred_method_by_type, pred_value_by_type, detected_order_agreement = parse_pipeline_answer(output_text)
        pred_types = set(pred_types_list)

        type_set_correct = pred_types == true_types
        order_correct = type_set_correct and pred_order == true_order

        # scored only for the types that are actually present
        step_results = {}
        for t, step in true_steps.items():
            pred_method = pred_method_by_type.get(t)
            pred_value = pred_value_by_type.get(t)
            method_correct = pred_method == step["method"]
            method_family_correct = method_family(pred_method) == method_family(step["method"])
            value_correct = VALUE_CORRECT_BY_TYPE[t](pred_value, step["value"])
            value_correct_within1 = VALUE_CORRECT_WITHIN1_BY_TYPE[t](pred_value, step["value"])
            step_results[t] = {
                "true_method": step["method"], "pred_method": pred_method,
                "method_correct": method_correct, "method_family_correct": method_family_correct,
                "true_value": step["value"], "pred_value": pred_value, "value_correct": value_correct,
                "value_correct_within1": value_correct_within1,
            }

        row = {
            "id": entry["id"], "output": output_text,
            "true_types": sorted(true_types), "pred_types": sorted(pred_types),
            "type_set_correct": type_set_correct,
            "true_order": true_order, "pred_order": pred_order, "order_correct": order_correct,
            "detected_order_agreement": detected_order_agreement,
            "steps": step_results,
        }
        rows.append(row)
        ans_file.write(json.dumps(row) + "\n")

    ans_file.close()

    n_total = len(rows)
    n_type_correct = sum(r["type_set_correct"] for r in rows)
    n_order_correct = sum(r["order_correct"] for r in rows)
    n_agreement = sum(r["detected_order_agreement"] for r in rows)

    method_correct = {"jpeg": [], "noise": [], "gamma": []}
    method_family_correct = {"jpeg": [], "noise": [], "gamma": []}
    value_correct = {"jpeg": [], "noise": [], "gamma": []}
    value_correct_within1 = {"jpeg": [], "noise": [], "gamma": []}
    by_n_correct = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        n = len(r["true_types"])
        by_n_correct[n].append(r["type_set_correct"])
        for t, s in r["steps"].items():
            method_correct[t].append(s["method_correct"])
            method_family_correct[t].append(s["method_family_correct"])
            value_correct[t].append(s["value_correct"])
            value_correct_within1[t].append(s["value_correct_within1"])

    def rate(lst):
        return sum(lst) / len(lst) if lst else None


    # overall metrics, conditional ones (e.g. order given correct type set), and
    # breakdowns by number of distortions (none / single / pair / triple)
    stats = {
        "model_path": args.model_path, "model_base": args.model_base, "n_total": n_total,
        "type_set_accuracy": n_type_correct / n_total if n_total else None,
        "order_accuracy": n_order_correct / n_total if n_total else None,
        "order_accuracy_given_type_correct": (n_order_correct / n_type_correct) if n_type_correct else None,
        "detected_order_agreement_rate": n_agreement / n_total if n_total else None,
    }
    for n in [0, 1, 2, 3]:
        stats[f"type_set_accuracy_n{n}"] = rate(by_n_correct[n])
    for t in ["jpeg", "noise", "gamma"]:
        stats[f"{t}_method_accuracy"] = rate(method_correct[t])
        stats[f"{t}_method_family_accuracy"] = rate(method_family_correct[t])
        stats[f"{t}_value_accuracy"] = rate(value_correct[t])
        stats[f"{t}_value_accuracy_within1"] = rate(value_correct_within1[t])

    print(json.dumps(stats, indent=2))
    if args.stats_file:
        os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
        json.dump(stats, open(args.stats_file, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--stats-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    eval_model(args)
