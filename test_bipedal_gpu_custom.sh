#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=dads_test_script
#SBATCH --time=00:05:00
#SBATCH --output=dads_test_output
#SBATCH --error=dads_test_error

# Modules to load for dads script
module load libs/cudnn/10.1-cuda-10.0
module load libs/cuda/11.0-gcc-5.4.0-2.26
module load CUDA/8.0.44
module load tools/git/2.18.0
module load libGLU/9.0.0-foss-2016a-Mesa-11.2.1
module load apps/ffmpeg/4.3
module load libxml2/2.9.3-foss-2016a

# Start dads conda env
conda activate dads-env

python unsupervised_skill_learning/dads_OOP.py --logdir=log_dir --flagfile=configs/bipedal_custom_test_offpolicy.txt

