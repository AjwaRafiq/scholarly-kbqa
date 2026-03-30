#!/bin/bash
#SBATCH --job-name=t5_gen
#SBATCH --partition=gpujobs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/t5_%j.log

cd /home/f247810/scholarly-kbqa
source activate mkb_kbqa

python src/generation/train_t5.py
