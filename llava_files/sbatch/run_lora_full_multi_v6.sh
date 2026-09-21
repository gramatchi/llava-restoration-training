#!/bin/bash
#SBATCH -J lora_v6
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/full_multi_v6_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/full_multi_v6_err_%j.txt
#SBATCH -t 28:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem 32G
#SBATCH --gres=gpu:l40s:1

# Full LoRA training of LLaVA-1.5-7B on the v6 dataset (3 epochs, ~12 h on one L40S).
# Run from inside the LLaVA repo after copying llava_files/ over it.
# Resumable: if it hits the time limit, submit it again and train.py picks up
# the last checkpoint in output_dir.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export HF_HOME=/nfsd/lttm4/tesisti/gramatchi/.cache/huggingface
export PIP_CACHE_DIR=/nfsd/lttm4/tesisti/gramatchi/.cache/pip

cd /nfsd/lttm4/tesisti/gramatchi/LLaVA

DATA=/nfsd/lttm4/tesisti/gramatchi/training_release/dataset_generation/data

# (inline comments are only possible because the args are in an array)
ARGS=(
    # LoRA
    --lora_enable True
    --lora_r 64                       # half of LLaVA's 13B recipe (128), our task is much narrower
    --lora_alpha 128
    --mm_projector_lr 2e-5            # smaller lr for the projector, as in LLaVA's LoRA recipe

    --deepspeed ./scripts/zero2.json  # LLaVA's ZeRO-2 config, plus gradient_clipping (see llava_files/scripts)

    # base model
    --model_name_or_path liuhaotian/llava-v1.5-7b
    --version v1

    # data. image_folder holds the clean photos, train.py distorts them on the fly
    --data_path $DATA/llava_multi_v6_train.json
    --eval_data_path $DATA/llava_multi_v6_val.json
    --image_folder /home/gramatchin/data/raw

    # vision side, same as LLaVA's finetune_lora.sh
    --vision_tower openai/clip-vit-large-patch14-336
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True

    --bf16 True
    --output_dir /nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6

    # schedule
    --num_train_epochs 3
    --per_device_train_batch_size 4   # largest that fit on one GPU
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 1

    # eval, checkpointing, early stopping
    --evaluation_strategy "steps"
    --eval_steps 3600                 # has to match save_steps for load_best_model_at_end
    --save_strategy "steps"
    --save_steps 3600
    --save_total_limit 3
    --load_best_model_at_end True     # the best-eval_loss checkpoint is kept, not just the last one
    --metric_for_best_model eval_loss
    --greater_is_better False
    --early_stopping_patience 2       # our own flag, added in train.py

    # optimizer, same as LLaVA's LoRA recipe
    --learning_rate 2e-4
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"

    --logging_steps 10
    --tf32 True
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True            # needed, the on-the-fly distortion happens in the lazy dataset
    --report_to none
)

# train_mem_sdpa.py is train() with sdpa attention, so flash-attn doesn't have to be built
deepspeed --master_port=$((10000 + SLURM_JOB_ID % 50000)) llava/train/train_mem_sdpa.py "${ARGS[@]}"
