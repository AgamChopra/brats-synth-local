#!/bin/bash

#SBATCH --job-name=brats-agam
#SBATCH --mail-user=achopra4@uw.edu
#SBATCH --mail-type=ALL

#SBATCH --gres=gpu:1
#SBATCH --account=kurtlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

#SBATCH --chdir=.
#SBATCH --export=all
#SBATCH --output=/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log/output.txt
#SBATCH --error=/gscratch/kurtlab/brats2024/repos/agam/brats-synth-local/log/error.txt

source ~/.bashrc
source activate agam_1

python3 train-gans.py --gui false --hyak true --identity gans
