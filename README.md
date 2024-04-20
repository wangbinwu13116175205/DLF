# Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective

# Requirements
conda env create -f DLF.yaml

# Data 

Please send an email to wbw1995@mail.ustc.edu.cn for data.

# Training 

python main.py --device cuda:4 --dataset SD --years 2019 --model_name DLF --seed 3028 --bs 64
