#!/bin/bash
#SBATCH --job-name=kbqa_pipeline
#SBATCH --partition=gpujobs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/pipeline_%j.log

cd /home/f247810/scholarly-kbqa
source activate mkb_kbqa

python src/pipeline/kbqa_pipeline.py
