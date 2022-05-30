# MNIST TensorFlow Training Script for OzStar

Training code is in `train.py` and it is submitted to the SLURM scheduler via `sbatch train.sh`.

Requires a valid virtual environment to be installed in the root directory as `venv/`.

<!-- 
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=daniel.tang@uwa.edu.au 
-->