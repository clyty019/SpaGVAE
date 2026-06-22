#!/bin/bash
#SBATCH --job-name=hvg_probe
#SBATCH --output=/home/zihend1/StaR/outputs/log_hvg_%a.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_hvg_%a.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-11
cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python scripts/experiments/star_v2/ablation_hvg.py --task $SLURM_ARRAY_TASK_ID
