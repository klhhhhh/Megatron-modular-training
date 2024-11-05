
pip install flash-attn
export CUDNN_PATH=/path/to/cudnn
export CUDNN_ROOT=
export CUDNN_HOME=
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CPATH=/global/common/software/nersc9/cudnn/8.9.3-cuda12/include:$CPATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/bin:$PATH
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin/nvcc:$PATH
export MAX_JOBS=1

pip install cmake==3.24.3
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
CXXFLAGS="-std=c++17" pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

https://github.com/NVIDIA/TransformerEngine/issues/355
https://github.com/NVIDIA/TransformerEngine/issues/383
https://github.com/NVIDIA/TransformerEngine/issues/918
https://github.com/NVIDIA/TransformerEngine/issues/803
https://github.com/NVIDIA/TransformerEngine/issues/459
