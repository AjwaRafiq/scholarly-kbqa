#!/bin/bash
#SBATCH --job-name=kbqa_eval
#SBATCH --partition=gpujobs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/eval_%j.log

cd /home/f247810/scholarly-kbqa
source activate mkb_kbqa

python src/evaluation/evaluate.py
