#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gpus-per-node=a100_3g.40gb:1
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"  
conda activate redenv 
python testCUDA.py