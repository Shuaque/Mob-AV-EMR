# Mob-AV-EMR:  A Mobile-Ready AV LLM and Data Recipe for EMR Generation

This repository contains the PyTorch implementation of the following paper:
> **Mob-AV-EMR:  A Mobile-Ready AV LLM and Data Recipe for EMR Generation**<be>
><br>
> Authors: Shuak Bakytnur<br>
> **Paper Link**: [http://arxiv.org/abs/xxx](http://arxiv.org/abs/xxx)

## Introduction
Mob-AV-EMR, a lightweight audio–video
multimodal LLM for mobile health.
<!-- <div align="center"><img width="90%" src="image.png?raw=true" /></div> -->

### Step 0. Environment Setup
- Python version >= 3.10
```bash
conda create -n mob-av-emr python=3.10 -y
conda activate mob-av-emr
git clone https://github.com/Shuaque/Mob-AV-EMR.git
cd Mob-AV-EMR
```
```bash
pip install transformers

cd fairseq
pip install --editable ./
```

### Step 1. Preparation
```bash

```
#### Pretrained Models
1. Download the ```AV-HuBERT Base model``` from this [link](https://github.com/facebookresearch/av_hubert) 
2. Download the ```Qwen2.5-3B-Instruct model``` from this [link](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
3. 2. Download the ```Whisper-medium.en``` from this [link](https://huggingface.co/openai/whisper-medium.en) 

After downloading, make sure to place the models in the correct directories:
- The `base_vox_iter5.pt(AV-HuBERT)` model should be placed in the `pretrained_models/avhubert` folder.

The required datasets are:
- LMRC
- LWR-1000


### Step 2. Training
```bash

```


### Step 3. Evaluation
```bash

```


### Citation
If you find this work useful in your research, please cite the paper:

```bibtex
@article{xxxx,
  title={Mob-AV-EMR:  A Mobile-Ready AV LLM and Data Recipe for EMR Generation},
  author={xxx},
  journal={arXiv preprint arXiv:xxx},
  year={2025}
}
```

### Acknowledgement
This project is based on the [avhubert](https://github.com/facebookresearch/av_hubert), [auto-avsr](https://github.com/mpc001/auto_avsr), and [fairseq](https://github.com/facebookresearch/fairseq) code.Also and [MMS-LLaMA](https://github.com/JeongHun0716/MMS-LLaMA.git) We would like to thank the developers of these projects for their contributions and the open-source community for making this work possible.