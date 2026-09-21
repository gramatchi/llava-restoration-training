#!/bin/bash
#SBATCH -J hf_upload
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/hf_upload_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/hf_upload_err_%j.txt
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -p allgroups
#SBATCH --mem 16G

# uploads the final adapter and its model card to Hugging Face.
# Needs a prior `hf auth login` (the token is read from ~/.cache/huggingface).
# It runs as a job because the upload gets killed on the login node.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi

REPO=gramatchi/llava-1.5-7b-restoration-lora

hf upload $REPO checkpoints/llava-multi-lora-v6-final-28800 . --commit-message "LoRA adapter, checkpoint-28800"
hf upload $REPO training_release/model/README.md README.md --commit-message "Add model card"
