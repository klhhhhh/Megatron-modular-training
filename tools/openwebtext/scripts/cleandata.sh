#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
source /pscratch/sd/k/klhhhhh/envs/megatron/bin/activate
module load cudatoolkit/12.4
srun hostname
srun python /global/homes/k/klhhhhh/Megatron-modular-training/tools/openwebtext/cleanup_dataset.py /pscratch/sd/k/klhhhhh/openwebtext_data/merged_output.json /pscratch/sd/k/klhhhhh/openwebtext_data/cleaned_up.json | tee $SCRATCH/openwebtext_data/output_parallel.txt
srun python /global/homes/k/klhhhhh/Megatron-modular-training/tools/openwebtext/merge_files.py | tee $SCRATCH/openwebtext_data/write.txt