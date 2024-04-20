# Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective

# TODO

The clean code will be released around mid-May.

# Introduction

The decoupled learning framework (DLF) is proposed in this paper, which consists of a spatial-temporal graph learning network (DSTG) with a specialized decoupling training strategy. Incorporating inductive biases of time-series structures, DSTG can interpret time dependencies into latent trend and seasonal terms. To enable prompt adaptation to the evolving distribution of the dynamic graph, our decoupling training strategy is devised to iteratively update these two types of patterns. Specifically, for learning seasonal patterns, we conduct thorough training for the model using a long time series (eg, three months of data). To enhance the learning ability of the model, we also introduce the masked auto-encoding mechanism. During this period, we frequently update trend patterns to expand new information from dynamic graphs. Considering both effectiveness and efficiency, we develop a subnet sampling strategy to select a few representative nodes for fine-tuning the weights of the model. These sampled nodes cover unseen patterns and previously learned patterns.

# Requirements
conda env create -f DLF.yaml

# Data 

Please send an email to wbw1995@mail.ustc.edu.cn for data. Unzip it to the data/.

# Training 

python main.py --device cuda:4--model_name DLF --seed 3028 --bs 64

# Citation

If you find it helpful or use the data, please cite it:

@inproceedings{wang2024towards,
  title={Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective},
  author={Wang, Binwu and Wang, Pengkun and Zhang, Yudong and Wang, Xu and Zhou, Zhengyang and Bai, Lei and Wang, Yang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={8},
  pages={9089--9097},
  year={2024}
}
