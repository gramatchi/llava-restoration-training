"""Builds a loadable model directory from checkpoint-28800 (last checkpoint).

Intermediate checkpoints only hold the LoRA adapter, so the fine-tuned
mm_projector weights (non_lora_trainables.bin) are missing. They still exist in
the raw DeepSpeed shards (global_step28800/), so we consolidate those and split
the state dict into LoRA / non-LoRA parts the same way train.py does at the end
of training.
"""
import os
import shutil
import sys

import torch

CKPT_DIR = "/nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6/checkpoint-28800"
ROOT_DIR = "/nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6"
OUT_DIR = "/nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6-final-28800"

# DeepSpeed drops zero_to_fp32.py into every checkpoint folder
sys.path.insert(0, CKPT_DIR)
from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

os.makedirs(OUT_DIR, exist_ok=True)

print("Consolidating ZeRO shards (loads the full fp32 state dict, needs memory)...")
state_dict = get_fp32_state_dict_from_zero_checkpoint(CKPT_DIR)
print(f"Total keys: {len(state_dict)}")


def strip_default_adapter_name(k):
    # PEFT keeps "lora_A.default.weight" internally, but the saved adapter format
    # is "lora_A.weight". With the ".default." left in, no keys match on load and
    # the model silently behaves like the base model (no error is raised).
    return k.replace(".default.weight", ".weight") if ".lora_" in k else k


lora_keys = {strip_default_adapter_name(k): v for k, v in state_dict.items() if "lora_" in k}
non_lora_keys = {k: v for k, v in state_dict.items() if "lora_" not in k}
print(f"lora keys: {len(lora_keys)}, non-lora keys: {len(non_lora_keys)}")
print("sample lora key:", next(iter(lora_keys)))

torch.save(lora_keys, os.path.join(OUT_DIR, "adapter_model.bin"))
torch.save(non_lora_keys, os.path.join(OUT_DIR, "non_lora_trainables.bin"))

# small config files that load_pretrained_model needs
for fname in ["adapter_config.json", "config.json"]:
    for src_dir in [ROOT_DIR, CKPT_DIR]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT_DIR, fname))
            break

print(f"Done -> {OUT_DIR}")
print(os.listdir(OUT_DIR))
