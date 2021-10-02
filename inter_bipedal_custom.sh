#!/bin/bash
#SBATCH --ntasks 8 --cpus-per-task 28
#SBATCH --partition=cpu
#SBATCH --job-name=roel_script
#SBATCH --time=24:00:00
#SBATCH --output=roel_main_output
#SBATCH --error=roel_main_error

# Modules to load for dads script
module load libs/cudnn/10.1-cuda-10.0
module load libs/cuda/11.0-gcc-5.4.0-2.26
module load CUDA/8.0.44
module load tools/git/2.18.0
module load libGLU/9.0.0-foss-2016a-Mesa-11.2.1
module load apps/ffmpeg/4.3
module load libxml2/2.9.3-foss-2016a

# Start dads conda env
. ~/.bashrc
conda activate dads-env

# Slurm setupe info
cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Dir is "$(pwd)"
echo Slurm job id is "${SLURM_JOBID}"
echo This job runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

python unsupervised_skill_learning/dads_OOP.py --logdir=log_dir --flagfile=configs/bipedal_inter_custom_offpolicy.txt

