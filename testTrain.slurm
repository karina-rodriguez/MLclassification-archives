#!/bin/bash
#
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

#SBATCH --account=ec285
#SBATCH --job-name=visualcode-job

# Load the required modules

module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
source imageclasslocal/bin/activate
export TORCH_HOME=/work/ec285/ec285/krodrig/.cache/torch
#export HOME=/work/ec285/ec285/krodrig
python trainDinov2.py

