#!/bin/bash
#SBATCH -J load_model
#SBATCH -o /data/sls/u/urop/mvideet/diffusion_reasoning/slurm/out/load_model%A_%a.out
#SBATCH -e /data/sls/u/urop/mvideet/diffusion_reasoning/slurm/err/load_model%A_%a.err
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=a6,a5
##SBATCH --partition=a5,a6,2080
#SBATCH --exclude sls-a6-5
#SBATCH --mem=22G
#SBATCH --ntasks-per-node=1

# PYTHON_VIRTUAL_ENVIRONMENT=diffusion_env
# source /data/sls/scratch/mvideet/anaconda3/etc/profile.d/conda.sh
# conda activate diffusion_env

MODEL_NAME="llada-8b"

cd /data/sls/u/urop/mvideet/diffusion_reasoning

# 2. Run your module with “-m src.run_vggsound” instead of calling the .py directly
python -u load_diffusion.py
