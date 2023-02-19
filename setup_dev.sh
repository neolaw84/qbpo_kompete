#!/bin/bash

# pre-requisites: 
# install anaconda
# create a conda environment
# activate the created conda environment

conda install ipykernel jupyter numpy scipy pandas scikit-learn
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

pip install -r requirements.txt