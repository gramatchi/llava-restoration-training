#!/bin/bash
#SBATCH -J order_exp
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/order_exp_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/order_exp_err_%j.txt
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi/training_release

python3 experiments/experiment_restoration_order_full.py
