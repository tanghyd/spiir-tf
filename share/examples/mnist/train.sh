#!/bin/bash
#
#SBATCH --job-name=tf-dist-test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12g
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --output ./logs.out
#SBATCH --error ./logs.err
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=daniel.tang@uwa.edu.au

# load environment modules
module purge

module load gcc/9.2.0 openmpi/4.0.2
module load cudnn/8.0.4-cuda-11.0.2
module load nccl/2.9.6
module load git/2.18.0
module load python3/7.4

# activate virtual environment
source /fred/oz016/dtang/spiir-tf/venv/bin/activate

srun python train.py
