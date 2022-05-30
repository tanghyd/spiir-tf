#!/bin/bash
#
#SBATCH --job-name=keras_dist
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12g
#SBATCH --gres=gpu:2
#SBATCH --output logs_dist/logs_dist_%j.out
#SBATCH --error logs_dist/logs_dist_%j.err

# load environment modules
module purge

module load gcc/9.2.0 openmpi/4.0.2
module load cudnn/8.1.0-cuda-11.2.0
module load git/2.18.0
module load python/3.7.4

# activate virtual environment
source /fred/oz016/dtang/spiir-tf/venv/bin/activate

mkdir -p ./logs_dist/
srun python train_dist.py --n-epochs 60 --debug --distribute