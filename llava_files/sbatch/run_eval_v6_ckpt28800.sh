#!/bin/bash
#SBATCH -J eval_v6_28800
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/eval_v6_28800_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/eval_v6_28800_err_%j.txt
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:l40s:1

# Evaluates the final model (checkpoint-28800) on the v6 test split (4800 examples).
# Run from inside the LLaVA repo after copying llava_files/ over it.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export HF_HOME=/nfsd/lttm4/tesisti/gramatchi/.cache/huggingface
export PIP_CACHE_DIR=/nfsd/lttm4/tesisti/gramatchi/.cache/pip

cd /nfsd/lttm4/tesisti/gramatchi/LLaVA

DATA_DIR=/nfsd/lttm4/tesisti/gramatchi/training_release/dataset_generation/data
RESULTS_DIR=/nfsd/lttm4/tesisti/gramatchi/eval_results_multi_v6
mkdir -p $RESULTS_DIR

# the answers file is resumable: delete it first if you want a genuine rerun
python -m llava.eval.eval_multi_restoration_v6 \
    --model-path /nfsd/lttm4/tesisti/gramatchi/checkpoints/llava-multi-lora-v6-final-28800 \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder /home/gramatchin/data/raw \
    --question-file $DATA_DIR/llava_multi_v6_test.json \
    --answers-file $RESULTS_DIR/v6_ckpt28800_pipeline_answers.jsonl \
    --stats-file $RESULTS_DIR/v6_ckpt28800_pipeline_stats.json

echo "=== done ==="
