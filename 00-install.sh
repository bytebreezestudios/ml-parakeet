#!/bin/bash

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade tiktoken
python3 -m pip install --upgrade bitsandbytes

# Example to install a pre-release version of PyTorch (May be out of date, check the latest version):
# - During development (c. 2020), there was an error with the current version of PyTorch resulting in `-inf` in LLM attention going to NaN.
#
# !python3 -m -y pip uninstall torch torchaudio torchvision torchviz
# !python3 -m -y pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
#
# References:
# - https://discuss.pytorch.org/t/pytorch-for-cuda-12/169447/40?page=2
# - https://stackoverflow.com/questions/5189199/bypass-confirmation-prompt-for-pip-uninstall

python3 -m pip install torch

python3 -m pip freeze > requirements.txt
