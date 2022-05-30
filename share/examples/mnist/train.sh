#!/bin/bash
#
#SBATCH --job-name=keras
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12g
#SBATCH --gres=gpu:2
#SBATCH --output logs/logs_%j.out
#SBATCH --error logs/logs_%j.err
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=daniel.tang@uwa.edu.au

# load environment modules
module purge

module load gcc/9.2.0 openmpi/4.0.2
module load cudnn/8.1.0-cuda-11.2.0
module load git/2.18.0
module load python/3.7.4

# activate virtual environment
source /fred/oz016/dtang/spiir-tf/venv/bin/activate

mkdir -p ./logs/
srun python train.py --debug --distribute