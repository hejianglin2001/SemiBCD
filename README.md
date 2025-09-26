# SemiBCD

This repository contains the testing code and pretrained models for our paper:

> **Paper under review**  
> Authors: [Anonymous for review]  
> Under Review, 2025  

---

## 🔧 Environment Setup

We recommend using **conda**:

```bash
conda create -n semibcd python=3.9 -y
conda activate semibcd
pip install -r requirements.txt
```
## 📥 Download Backbone Pretrained Weights
SemiBCD uses a ResNet-50 backbone. Download the pretrained checkpoint:
👉 [ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) 


## 📥 Download Dataset
### 1. LEVIR-CD-256
👉 [LEVIR-CD-256 Dataset](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)  
下载后解压到 `data/LEVIR-CD-256/` 文件夹。

### 2. WHU-CD-256
👉 [WHU-CD-256 Dataset](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)  
下载后解压到 `data/WHU-CD-256/` 文件夹。

## 🚀 Run Testing

### 1. **Download pretrained experiment weights**  
   👉 [SemiBCD Experiment Weights](YOUR_EXPERIMENT_LINK_HERE)  
   

### 2. **Run testing**  
   例如运行 LEVIR 测试：
```bash
python test.py --config configs/levir_test.yaml --checkpoint checkpoints/sembcd_best.pth
```




## Acknowledgements
SemiBCD is based on [SemiCD-VL](https://github.com/likyoo/SemiCD-VL), [SemiVL](https://github.com/google-research/semivl), [UniMatch](https://github.com/LiheYoung/UniMatch), [APE](https://github.com/shenyunhang/APE), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank their authors for making the source code publicly available.

