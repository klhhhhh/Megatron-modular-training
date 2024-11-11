#!/bin/bash
#SBATCH -A m4410_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -c 32
#SBATCH --gpus-per-task=1

source /pscratch/sd/k/klhhhhh/envs/megatron/bin/activate
module load cudatoolkit/12.4
cd /global/homes/k/klhhhhh/Megatron-modular-training
bash examples/meta-training/gpt/train_gpt3_857m.sh

