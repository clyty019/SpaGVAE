#!/bin/bash
#SBATCH --job-name=k_robust
#SBATCH --output=/home/zihend1/StaR/outputs/log_krobust_%a.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_krobust_%a.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-11
cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python scripts/experiments/star_v2/run_k_robustness.py --task $SLURM_ARRAY_TASK_ID
