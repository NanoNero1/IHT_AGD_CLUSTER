#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gpus-per-node=a100_7g.80gb:2
#SBATCH --time=60:00:00          # total run time limit (HH:MM:SS)

module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"  
conda activate redenv 
##python cluster_iht_agd.py
python workIMAGENET.py