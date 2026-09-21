#!/bin/bash
#SBATCH -J demo_check
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/demo_check_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/demo_check_err_%j.txt
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:l40s:1

# checks that demo.py works with the adapter downloaded from Hugging Face,
# on one of the final-test photos (which the model has never seen)

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export HF_HOME=/nfsd/lttm4/tesisti/gramatchi/.cache/huggingface

# the adapter is downloaded into a folder whose name contains "lora"
MODEL_DIR=/nfsd/lttm4/tesisti/gramatchi/.cache/hf_check/llava-1.5-7b-restoration-lora
hf download gramatchi/llava-1.5-7b-restoration-lora --local-dir $MODEL_DIR

cd /nfsd/lttm4/tesisti/gramatchi/LLaVA
python /nfsd/lttm4/tesisti/gramatchi/training_release/demo.py \
    --model-path $MODEL_DIR \
    --image /home/gramatchin/data/raw/extra_images/0004000/0003001.png
