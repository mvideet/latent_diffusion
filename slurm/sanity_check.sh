#!/bin/bash
#SBATCH -J mcq_sav
#SBATCH -o /data/sls/u/urop/mvideet/diffusion_reasoning/slurm/out/sanity_check%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/diffusion_reasoning/slurm/err/sanity_check%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6
##SBATCH --partition=a5,a6
#SBATCH --exclude sls-a6-5
#SBATCH --mem=30G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=diffusion_env
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate diffusion_env

cd /data/sls/u/urop/mvideet/diffusion_reasoning/Latent_Injection

python -u sanity_check.py
