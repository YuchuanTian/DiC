# DiC: Rethinking Conv3x3 Designs in Diffusion Models

<p align="left">
<a href="https://arxiv.org/abs/2501.00603" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2501.00603-b31b1b.svg?style=flat" /></a>
    <a href="https://github.com/YuchuanTian/DiC/blob/main/imgs/poster_cvpr.pdf" alt="arXiv">
    <img src="https://img.shields.io/badge/Poster-b31b1b.svg?style=flat" /></a>
</p>

**_ConvUNets have been overlooked... but they outperform Diffusion Transformers!_**

## News

**6/11/2025:** We have released the codes of DiC! 🔥🔥🔥 Weights, SiT, and REPA versions are coming very soon.

**3/3/2025:** Codes & Weights are at the final stage of inspection. We will have them released ASAP.

**2/27/2025:** DiC is accepted by CVPR 2025! 🎉🎉

![effect](imgs/demo.jpg)


🤔 In this work, we intend to build a diffusion model with Conv3x3 that is simple but efficient.

🔧 We re-design architectures & blocks of the model to tap the potential of Conv3x3 to the full.

🚀 The proposed DiC ConvUNets are more powerful than DiTs, and **much much faster**!

## Repo Outline

This repo is mostly based on the official repo of DiT. Weights, SiT and REPA versions will be opensourced very soon.

Torch model script: **dic_models.py**

## Preparation

Please run command ```pip install -r requirements.txt``` to install the supporting packages.

(Optional) Please download the VAE from this [link](https://huggingface.co/stabilityai/sd-vae-ft-ema). The VAE could be automatically downloaded as well.

## Training

Here we provide two ways to train a DiC model: 1. train on the original ImageNet dataset; 2. train on preprocessed VAE features (Recommended).

**Training Data Preparation**
Use the original ImageNet dataset + VAE encoder. Firstly, download ImageNet as follows:


```
imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

Then run the following command:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --data-path={path to imagenet/train} --image-size=256 --model={model name} --epochs={iteration//5000} # fp32 Training

accelerate launch --mixed_precision fp16 train_accelerate.py --data-path {path to imagenet/train} --image-size=256 --model={model name} --epochs={iteration//5000} # fp16 Training
```

**Training Feature Preparation (RECOMMENDED)**

Following Fast-DiT, it is recommended to load VAE features directly for faster training. You don't need to download the enormous ImageNet dataset (> 100G); instead, a much smaller "VAE feature" dataset (~21G for ImageNet 256x256) is available here on [HuggingFace](https://huggingface.co/datasets/yuchuantian/imagenet_vae_256) and [MindScope](https://www.modelscope.cn/models/YuchuanTian/imagenet_vae_256/). Please do the following steps:

1. Download [imagenet_feature.tar](https://huggingface.co/datasets/yuchuantian/imagenet_vae_256/blob/main/imagenet_feature.tar)

2. Unzip the tar ball by running ```tar -xf imagenet_feature.tar```

```
imagenet_feature/
├── imagenet256_features/ # VAE features
└── imagenet256_labels/ # labels
```

3. Append parser ```--feature-path={path to imagenet_feature}``` to the training command.



## Inference

#### Weights

Coming soon. Please keep tuned!

#### Sampling

Run the following command for parallel sampling:

```bash
torch --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt={path to checkpoint} --image-size=256 --model={model name} --cfg-scale={cfg scale}
```

## BibTex Formatted Citation

If you find this repo useful, please cite:
```
@article{tian2025dic,
  author       = {Yuchuan Tian and
                  Jing Han and
                  Chengcheng Wang and
                  Yuchen Liang and
                  Chao Xu and
                  Hanting Chen},
  title        = {DiC: Rethinking Conv3x3 Designs in Diffusion Models},
  journal      = {CoRR},
  volume       = {abs/2501.00603},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2501.00603},
  doi          = {10.48550/ARXIV.2501.00603},
  eprinttype    = {arXiv},
  eprint       = {2501.00603},
  timestamp    = {Mon, 10 Feb 2025 21:52:20 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2501-00603.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



## Acknowledgement

We acknowledge the authors of the following repos:

https://github.com/facebookresearch/DiT (Codebase)

https://github.com/YuchuanTian/U-DiT (Codebase)

https://github.com/chuanyangjin/fast-DiT (FP16 training; Training on features)

https://github.com/openai/guided-diffusion (Metric evalutation)

https://huggingface.co/stabilityai/sd-vae-ft-ema (VAE)
