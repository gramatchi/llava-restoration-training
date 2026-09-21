#!/bin/bash
#SBATCH -J find_jpeg_v6
#SBATCH -o /nfsd/lttm4/tesisti/gramatchi/training_release/logs/find_jpeg_v6_%j.txt
#SBATCH -e /nfsd/lttm4/tesisti/gramatchi/training_release/logs/find_jpeg_v6_err_%j.txt
#SBATCH -t 01:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 16G

# best-method search (jpeg) for the 2000 photos added in v6 (extra_images/0002000
# and 0003000), 10 quality levels. About 30 s/image on one thread, so roughly
# 30 min at 32 workers; the time limit leaves plenty of margin.
# Resumable: already-processed images are skipped, so a resubmit after a
# timeout is fine.

source /nfsd/lttm4/tesisti/gramatchi/miniconda3/bin/activate
conda activate llava

# one thread per worker process, otherwise numpy/BLAS oversubscribes the cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /nfsd/lttm4/tesisti/gramatchi/training_release

N_WORKERS=32 python3 -u dataset_generation/find_best_methods_jpeg_v6.py
