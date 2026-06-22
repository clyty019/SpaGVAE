#!/bin/bash
#SBATCH --job-name=beta05
#SBATCH --output=/home/zihend1/StaR/outputs/log_beta05_%a.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_beta05_%a.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python scripts/experiments/star_v2/run_beta_alpha05.py --gpuidx $SLURM_ARRAY_TASK_ID
