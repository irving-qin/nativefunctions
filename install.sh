
pip install ninja
pip show torch

cuda_version=`python -c 'import torch; print(torch.version.cuda)'`
echo "CUDA version building current torch: $cuda_version"

export CUDA_HOME=/usr/local/cuda-$cuda_version
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

python setup.py develop
python custom_layer_norm.py
