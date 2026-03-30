#!/bin/bash
#SBATCH --job-name=bert_ranker
#SBATCH --partition=gpujobs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/ranker_%j.log

cd /home/f247810/scholarly-kbqa
source activate mkb_kbqa

python src/ranking/train_ranker.py
