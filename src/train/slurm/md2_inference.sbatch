#!/bin/bash
#SBATCH -J inference_md2
#SBATCH -N 1 
#SBATCH --mem-per-cpu=65536
#SBATCH --gres=gpu:H100:1          # Try --gres instead of -G
#SBATCH --time=02:00:00
#SBATCH -o inference-md2-%j.out
#SBATCH -c 2

module load anaconda3 cuda/12.6.1
conda activate vlm_firewatch_env

cd /home/hice1/smanoli3/VLM-FireWatch/src/train

python inference_moondream2.py --batch_size 1 --num_workers 2 --plot_cm