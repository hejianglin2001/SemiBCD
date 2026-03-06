# SemiBCD：Semi-Supervised Building Change Detection From Bitemporal Remote Sensing Images Leveraging Visual–Language Models and Consistency Learning

**Code for SemiBCD paper: [Semi-Supervised Building Change Detection From Bitemporal Remote Sensing Images Leveraging Visual–Language Models and Consistency Learning.](https://ieeexplore.ieee.org/abstract/document/11414160)**

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
Extract the downloaded file to the `data/LEVIR-CD-256/` folder.

### 2. WHU-CD-256
👉 [WHU-CD-256 Dataset](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)  
Extract the downloaded file to the `data/WHU-CD-256/` folder.

## 🚀 Run Testing

### 1. **Download pretrained experiment weights**  
   👉 [SemiBCD Experiment Weights](https://pan.baidu.com/s/13MppHfYZyVdxRfkEw9tsTg?pwd=4zvg)
   

### 2. **Run testing**  
   Run WHU test:
   
```bash
python eval.py --config configs/eval_whu_config.yaml --checkpoint ./best.pth

```
   Run LEVIR test:
```bash
python eval.py --config configs/eval_levir_config.yaml --checkpoint ./best.pth

```


## 🚀 Run Training


   Train on WHU-CD:
   
```bash
python experiments.py --exp 48 --run RUN_ID

# e.g. RUN_ID=0 for SemiBCD on WHU-CD with 5% labels
# RUN_ID controls the labeled data ratio:
# 0 → 5% labels
# 1 → 10% labels
# 2 → 20% labels
# 3 → 40% labels
```
   Train on LEVIR-CD:
```bash
python experiments.py --exp 47 --run RUN_ID

# e.g. RUN_ID=0 for SemiBCD on LEVIR-CD with 5% labels
# RUN_ID controls the labeled data ratio:
# 0 → 5% labels
# 1 → 10% labels
# 2 → 20% labels
# 3 → 40% labels
```
## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@inproceedings{APE,
  title={Aligning and Prompting Everything All at Once for Universal Visual Perception},
  author={Shen, Yunhang and Fu, Chaoyou and Chen, Peixian and Zhang, Mengdan and Li, Ke and Sun, Xing and Wu, Yunsheng and Lin, Shaohui and Ji, Rongrong},
  journal={CVPR},
  year={2024}
}
```

## Acknowledgements
SemiBCD is based on [SemiCD-VL](https://github.com/likyoo/SemiCD-VL), [SemiVL](https://github.com/google-research/semivl), [UniMatch](https://github.com/LiheYoung/UniMatch), [APE](https://github.com/shenyunhang/APE), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank their authors for making the source code publicly available.

