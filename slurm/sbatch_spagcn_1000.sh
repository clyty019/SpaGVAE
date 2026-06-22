#!/bin/bash
#SBATCH --job-name=spagcn_1k
#SBATCH --output=/home/zihend1/StaR/outputs/log_spagcn_1000_%a.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_spagcn_1000_%a.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-47

cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python \
    scripts/experiments/star_v2/run_spagcn_1000.py --task ${SLURM_ARRAY_TASK_ID}
