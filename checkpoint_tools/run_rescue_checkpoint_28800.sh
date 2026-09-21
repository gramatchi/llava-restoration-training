#!/bin/bash
#SBATCH -J rescue_28800
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/rescue_28800_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/rescue_28800_err_%j.txt
#SBATCH -t 00:15:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -p allgroups
#SBATCH --mem 16G

# turns checkpoint-28800 into a loadable model dir (see rescue_checkpoint_28800.py)
# needs a proper memory allocation, it runs out of memory in an interactive shell

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

cd /nfsd/lttm4/tesisti/gramatchi/training_release
python3 checkpoint_tools/rescue_checkpoint_28800.py
