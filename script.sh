#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=train
#SBATCH --mail-type=END
#SBATCH --mail-user=db3484@nyu.edu
#SBATCH --output=slurm_data.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module purge
module load python/intel/2.7.12
module load pytorch/0.1.12_2
module load torchvision/0.1.8
module load scikit-image/intel/0.12.3

cd /scratch/db3484/Melanoma/mybackprop/
python run.py
