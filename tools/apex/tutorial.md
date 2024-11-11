pip uninstall apex 
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.04-dev
pip install -r requirements.txt
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./