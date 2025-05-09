#!/bin/bash

#SBATCH --partition=mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name="CS 601.471/671 porject"
#SBATCH --output=val_out2
#SBATCH --mem=32G
#SBATCH --nodelist=gpuz01
module load anaconda
conda activate project

python run_validation.py
