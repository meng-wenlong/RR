conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers
pip install trl
pip install accelerate
pip install datasets
pip install evaluate
pip install vllm

pip install jsonlines

# Install pii
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/src" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh