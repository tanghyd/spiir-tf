#!/bin/bash
SBATCH=1  # SUBMIT TO SLURM

if [ $SBATCH -eq 1 ]
then
  sbatch train.sh
else
  ./train.sh
fi 