export CUDNN_PATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CPATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12/include:$CPATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin:$PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/lib64:$LD_LIBRARY_PATH
export CUDACXX=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/nvcc
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin/nvcc:$PATH
export MAX_JOBS=4

export PATH=$HOME/gcc-12.3/bin:$PATH
export LD_LIBRARY_PATH=$HOME/gcc-12.3/lib64:$LD_LIBRARY_PATH