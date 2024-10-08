#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=2               # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem=512G
#SBATCH --time=60
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=a100_7g.80gb:4

module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"  
conda activate redenv 
##python cluster_iht_agd.py
python workIMAGENET.py