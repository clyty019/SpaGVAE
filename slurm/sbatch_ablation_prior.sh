#!/bin/bash
#SBATCH --job-name=prior_only
#SBATCH --output=/home/zihend1/StaR/outputs/log_prior_only.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_prior_only.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python scripts/experiments/star_v2/ablation_prior_only.py
