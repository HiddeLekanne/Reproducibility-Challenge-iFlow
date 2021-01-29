#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=iFlow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=18:00:00
#SBATCH --mem=10000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

source activate iFlow-cuda
for seed in $(seq 1 100)
do
    python main.py \
        -x 1000_5_5_5_3_$seed'_'gauss_xtanh_u_f \
        -i iVAE \
        -ft RQNSF_AG \
        -npa Softplus \
        -fl 10 \
        -lr_df 0.25 \
        -lr_pn 10 \
        -b 64 \
        -e 20 \
        -l 1e-3 \
        -s 1 \
        -u 0 \
        -c 
done
