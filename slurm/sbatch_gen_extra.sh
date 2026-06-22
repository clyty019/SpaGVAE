#!/bin/bash
#SBATCH --job-name=gen_extra
#SBATCH --output=/home/zihend1/StaR/outputs/log_genextra_%a.txt
#SBATCH --error=/home/zihend1/StaR/outputs/log_genextra_%a.err
#SBATCH --exclude=voyager
#SBATCH --partition=zhanglab.p
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-3
cd /home/zihend1/StaR
/home/zihend1/.conda/envs/py38/bin/python scripts/experiments/star_v2/run_gen_extra_baselines.py --task $SLURM_ARRAY_TASK_ID
