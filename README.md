# 🦄 PoseAnything: Universal Pose-guided Video Generation with Part-aware Temporal Coherence

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

## 📝 Introduction

PoseAnything is a universal pose-guided video generation framework. It enables high-quality video generation for both human and non-human characters from arbitrary skeletal inputs

---

## 🛠️ Installation Guide

### 1. 📂 Clone Repository

```bash
git clone https://github.com/Ryan-w2024/PoseAnything.git
cd PoseAnything
```
### 2. 🐍 Environment Setup
Install with conda
```Bash
conda create -n poseanything python=3.10
conda activate poseanything
pip install -r requirements.txt
```

### 3. 💾 Model Weights Download
Use the VAE and tokenizer of the wan2.2-TI2V-5B model.
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./models/Wan2.2-TI2V-5B
huggingface-cli download Ryan241005/PoseAnything --local-dir ./models/Pony
```

After downloading, the weights files should be organized as:
```bash
PoseAnything/
├── models/
│   │── Wan2.2-TI2V-5B/
│   │   └── ...
│   └── Pony/
└── ...
```
## 💻 Quick Start: Inference
To run PoseAnything, you need to extract the masked image containing only the target subject based on the input first frame and skeleton sequence. You can either store the masked image directly to DATA_DIR/video, or use the following example script for automatic extraction:

```bash
cd Extractor
bash mask.sh
```
We provide a demo sript to run PoseAnything:

```bash
bash jobs/test.sh
```
If you wish to test the version of our method that does not include the Part-aware Temporal Coherence (PTC) module, run the following command. This version does not require the masked first frame as input.
```bash
bash ./job/test_without_ptc.sh
```
✔ Tip: PoseAnything supports arbitrary skeleton inputs. For **strong** skeletal conditions (large motion/high density input), we suggest using a **smaller CFG scale or no CFG** for natural output. For **weak** skeletal conditions (small motion/low density input), **increase the CFG scale** to enhance fitting to the pose.

## 🗃️ Data Process
We also provide the code for automated skeleton extraction, which is built based on [BlumNet](https://github.com/cong-yang/BlumNet) and [Grounded-Sam-2](https://github.com/IDEA-Research/Grounded-SAM-2).

### Installation Guide
To avoid conflicts, we highly recommend creating a new Conda environment.
```bash
cd Extractor
conda create -n extractor python=3.10
conda activate extractor
pip install -r requirement.txt

# Compile CUDA operators (as required by BlumNet) 
cd BlumNet/models/ops
sh ./make.sh
cd ../../
```
### Download Weights
Please download the weights for [BlumNet](https://github.com/cong-yang/BlumNet) and [Grounded-Sam-2](https://github.com/IDEA-Research/Grounded-SAM-2) following the instructions provided in corresponding repository, then run
```bash
cp -r Grounded/sam2 ./
```

### Usage
For the extraction of skeletons from your own video data, paths to the videos to be processed and their corresponding captions are needed, as shown in `../data/example/raw_metadata.csv`.
To run the example：
```bash
bash ./job/test.sh

