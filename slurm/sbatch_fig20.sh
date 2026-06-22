#!/bin/bash
#SBATCH --job-name=fig20
#SBATCH --output=/home/zihend1/StaR/outputs/log_fig20.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_fig20.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python scripts/experiments/star_v2/run_hemibrain_fig20.py
