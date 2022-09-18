# BCDNet

This is the implementation of our ECCV2022 paper "Towards-Accurate-Binary-Neural-Networks-via-Modeling-Contextual-Dependencies". https://arxiv.org/abs/2209.01404

Existing Binary Neural Networks (BNNs) mainly operate on local convolutions with binarization function. However, such simple bit operations lack the ability of modeling contextual dependencies, which is critical for learning discriminative deep representations in vision models. In this work, we tackle this issue by presenting new designs of binary neural modules, which enables BNNs to learn effective contextual dependencies. First, we propose a binary multi-layer perceptron (MLP) block as an alternative to binary convolution blocks to directly model contextual dependencies. Both short-range and long-range feature dependencies are modeled by binary MLPs, where the former provides local inductive bias and the latter breaks limited receptive field in binary convolutions. A sparse contextual interaction is achieved with the long-short range binary MLP block. Second, we compute the contextual dynamic embeddings to determine the binarization thresholds in general binary convolutional blocks. Armed with our binary MLP blocks and improved binary convolution, we build the BNNs with explicit Contextual Dependency modeling, termed as BCDNet. On the standard ImageNet-1K classification benchmark, the BCDNet achieves 72.3\% Top-1 accuracy and outperforms leading binary methods by a large margin. In particular, the proposed BCDNet exceeds the state-of-the-art ReActNet-A by 2.9\% Top-1 accuracy with similar operations, demonstrating the effectiveness of our binary designs.

<div align=center>
<img width=60% src="https://github.com/Sense-GVT/BCDNet/blob/main/images/fig4.jpg"/>
</div>

The performance of BCDNets on ImageNet-1K is:

<div align=center>
<img width=60% src="https://github.com/Sense-GVT/BCDNet/blob/main/images/fig2.jpg"/>
</div>

## Run

### 1. Requirements:
* slurm server
* python3, pytorch, torchvision ...
    
### 2. Data:
* Download ImageNet dataset

### 3. Steps to run:
(1) Step1:  binarizing activations
* Change directory `cd model_zoo_exp/BCDNet/a_3f`
* run `bash run.sh`

(2) Step2:  binarizing weights and activations
* Change directory `cd model_zoo_exp/BCDNet/a_3`
* run `bash run.sh`

## Pretrained model
* Step1: [BCDNet-A](https://drive.google.com/file/d/103GiQUx422DpwGuTR0bhxfIJvVoR00M5/view?usp=sharing)
* Step2: [BCDNet-A](https://drive.google.com/file/d/12jL6SzBppXJry0fLq0AzhY1dYFVGyfp5/view?usp=sharing)

## Contact

Xingrun Xing, BUAA (sy2002215 at buaa.edu.cn)
Yalong Jiang, BUAA (AllenYLJiang at outlook.com)
