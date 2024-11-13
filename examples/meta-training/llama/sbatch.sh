#!/bin/bash
#SBATCH -A m4410_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 20:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

source /pscratch/sd/k/klhhhhh/envs/megatron/bin/activate
module load cudatoolkit/12.4
cd /global/homes/k/klhhhhh/Megatron-modular-training
bash /global/homes/k/klhhhhh/Megatron-modular-training/examples/meta-training/llama/llama3.1-8b_pretrain.sh

