# LLaVA restoration planner: data, training and model

Code for fine-tuning LLaVA-1.5-7B (LoRA) to look at a photo and say which
distortions it has (JPEG compression, Gaussian noise, exposure/gamma; zero to
three at once), how strong they are, in which order to undo them, and which
restoration method to use for each step.

This repo covers the final version of the project (v6) and the checkpoint used
for all reported results (checkpoint-28800). The evaluation on 1000 held-out
photos is in a separate repo: https://github.com/gramatchi/final-test-llava

The trained LoRA adapter (693 MB) is too big for GitHub and is on Hugging Face:
https://huggingface.co/gramatchi/llava-1.5-7b-restoration-lora (see `model/README.md`).

## Folder layout

| Folder | Content |
|---|---|
| `common/` | `corruptors.py` (how distortions are applied) and the classical restoration functions |
| `dataset_generation/` | best-method search per photo, manifest generator, train/val/test split, and the generated data (`data/`) |
| `experiments/` | the restoration-order experiment (does reverse order matter?) and its results |
| `llava_files/` | our changes to the LLaVA repo (training script, eval script, DeepSpeed config) and the sbatch scripts for training and evaluation |
| `checkpoint_tools/` | rebuilds a loadable model dir from a raw DeepSpeed checkpoint |
| `model/` | model card and the small config files of the adapter (weights are on Hugging Face) |

## How the pieces fit

Distortions are not saved to disk. The manifest lists, for each example, the
clean photo, the distortions to apply (type, strength, order) and the answer
text. `train.py` applies the distortions on the fly when a sample is loaded.
The answer names the detected types, the corruption order, the restoration order
(the reverse) and, for each step, the best method found by exhaustive search on
that exact photo.

## Reproducing

1. **Ground-truth methods** (CPU jobs, resumable):
   `sbatch dataset_generation/sbatch/run_find_best_methods_jpeg_v6.sh`, same for `_noise_`.
   For the 2000 photos added in v6. The result for the first 1000 is already in
   `dataset_generation/data/original_1000_method_lookup_*.json`.
2. **Dataset**:
   ```
   python3 dataset_generation/build_method_lookup_v6.py
   python3 dataset_generation/build_dataset_manifest_v6.py
   python3 dataset_generation/split_dataset_v6.py
   ```
   The train/val/test files this produces are already in `dataset_generation/data/`
   (the full manifest is not, it can be regenerated).
3. **Set up LLaVA**: clone https://github.com/haotian-liu/LLaVA at commit
   `c121f0432da27facab705978f83c4ada465e46fd`, then copy our files over it:
   ```
   cp -r llava_files/llava llava_files/scripts <LLaVA repo>/
   ```
   Changes to stock LLaVA: `train.py` (on-the-fly distortion, eval data, early
   stopping, non-strict DeepSpeed resume), `train_mem_sdpa.py` (sdpa attention
   instead of flash-attn), `scripts/zero2.json` (added `gradient_clipping`, without
   it the loss blew up once), and `eval_multi_restoration_v6.py`.
   `train.py` and the eval script import `corruptors.py` from `common/` and
   `severity_levels.py` from `dataset_generation/`; set `RESTORATION_COMMON_DIR` and
   `RESTORATION_DATASET_DIR` if they are not at the default locations.
4. **Train**: `sbatch llava_files/sbatch/run_lora_full_multi_v6.sh` (3 epochs,
   about 12 h on one L40S, resumable). Hyperparameters are LLaVA's LoRA recipe,
   with `lora_r=64`, and evaluation and checkpointing every 3600 steps with early
   stopping on validation loss.
5. **Final checkpoint**: intermediate checkpoints only contain the LoRA weights, so
   checkpoint-28800 is rebuilt with `sbatch checkpoint_tools/run_rescue_checkpoint_28800.sh`.
6. **Evaluate**: `sbatch llava_files/sbatch/run_eval_v6_ckpt28800.sh`.

## Trying the model on a photo

Download the adapter from Hugging Face and run `demo.py` from inside the LLaVA repo
(set up as in step 3). It prints the model's answer for one photo:

```
cd <LLaVA repo>
python <this repo>/demo.py \
    --model-path <folder with the adapter, name must contain "lora"> \
    --image photo.png
```

A GPU with about 16 GB is needed.

Create `logs/` first (`mkdir -p logs`), the sbatch scripts write there.
Paths in the scripts (`/nfsd/lttm4/...`, `/home/gramatchin/...`) are the ones on
our cluster, change them for your setup. The photos themselves are not included.

The order experiment (`experiments/`) was run once on 15 photos and is included
with its results. `experiment_gamma_restoration_large.py` was written on a Mac
(`caffeinate`, local paths) and needs those edited before it runs elsewhere.

Environment notes: PyTorch 2.1.2, transformers 4.37.2, deepspeed 0.12.6, peft
0.4.0 (the versions LLaVA pins for training; newer peft and deepspeed break it).
