#!/bin/bash
#SBATCH -J find_noise_v6
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/find_noise_v6_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/find_noise_v6_err_%j.txt
#SBATCH -t 02:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G

# same as the jpeg search but for Gaussian noise (10 sigma levels, ~57 s/image
# on one thread, roughly an hour at 32 workers). Resumable.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /nfsd/lttm4/tesisti/gramatchi/training_release

N_WORKERS=32 python3 -u dataset_generation/find_best_methods_noise_v6.py
