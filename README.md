# 🦄 PoseAnything: Universal Pose-guided Video Generation with Part-aware Temporal Coherence

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.13465-b31b1b.svg)](http://arxiv.org/abs/2512.13465)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://ryan-w2024.github.io/project/PoseAnything/)

## 📝 Introduction

PoseAnything is a universal pose-guided video generation framework. It enables high-quality video generation for both human and non-human characters from arbitrary skeletal inputs

---

## 📅 Time Schedule

We are committed to open-sourcing our work to the community as quickly as possible. Below is the short-term release plan:

| No. | Content                             | Estimated Completion Time |
| :--- |:------------------------------------|:--------------------------|
| **1** | **Model Enhanced Using Human Data** | **✅**      |
| **2** | **XPose Dataset Release**           | **Within Two Weeks**      |

Please monitor our repository for the latest code and dataset release announcements.

---

## 🛠️ Installation Guide

### 1. 📂 Clone Repository

```bash
git clone https://github.com/Ryan-w2024/PoseAnything.git
cd PoseAnything
```
### 2. 🐍 Environment Setup
Install with conda
```bash
conda create -n poseanything python=3.10
conda activate poseanything
pip install -e .
pip install flash_attn --no-build-isolation
```

### 3. 💾 Model Weights Download
Use the following command to download the model weights:
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./models/Wan2.2-TI2V-5B
huggingface-cli download Ryan241005/PoseAnything --local-dir ./models/Pony
```

After downloading, the weights files should be organized as:
```bash
PoseAnything/
├── models/
│     ├─ Wan2.2-TI2V-5B/
│     │       ├── models_t5_umt5-xxl-enc-bf16.pth
│     │       ├── Wan2.2_VAE.pth
│     │       └── ...
│     └── Pony/
│           ├── diffusion_pytorch_model-00001-of-00002.safetensors
│           ├── diffusion_pytorch_model-00002-of-00002.safetensors
│           └── ...
└── ...
```
## 💻 Quick Start: Inference
To run PoseAnything, you need to extract the masked image of the target subject based on the first frame and skeleton. You can either store the masked image directly to DATA_DIR/video, or use the following example script for automatic extraction:

```bash
cd Extractor
bash mask.sh # May need to downgrade transformers to 4.40.2
```

The data will then be formatted as follows:
```bash
DATA_DIR/
├── first_frame/
│      └── {file_name}.png
├── skeleton_image/
│      └── {file_name}/
│              └── 000.png
│              └── 001.png
│              └── 002.png
├── video/
│      └── {file_name}_id.png
└── ...

```
Then, You can then use the provided example script to run the demo

```bash
bash test.sh
```
If you wish to test the version that does not include the PTC module, run the following command (masked image is not required).
```bash
bash test_without_ptc.sh
```
✔ Tip: PoseAnything supports arbitrary skeleton inputs. For **strong** skeletal conditions (large motion/high density input), we suggest using a **smaller CFG scale or no CFG** for natural output. For **weak** skeletal conditions (small motion/low density input), **increase the CFG scale** to enhance fitting to the pose.
### Demo Showcase

To test the TikTok dataset, refer to the script below:
```bash
bash test_tiktok.sh
```

| | | | |
| :---: | :---: |:---: | :---: |
| <video src="https://github.com/user-attachments/assets/a49d5190-4878-46e7-85ac-c799f4538749" width="100%" muted autoplay loop playsinline></video> | <video src="https://github.com/user-attachments/assets/637bd5ff-73ee-45b3-b28d-f07b92a50531" width="100%" muted autoplay loop playsinline></video> | <video src="https://github.com/user-attachments/assets/4f2580f7-4fa2-4c36-b50c-9fc53ec60c20" width="100%" muted autoplay loop playsinline></video> | <video src="https://github.com/user-attachments/assets/dfc415bc-d27d-4e4e-82b9-0473720965c1" width="100%" muted autoplay loop playsinline></video>  | 
| <video src="https://github.com/user-attachments/assets/400e7a68-f983-4102-9188-7660b602d910" width="100%" muted autoplay loop playsinline></video> | <video src="https://github.com/user-attachments/assets/5a532ed1-9226-40d3-b2f4-520652975b37" width="100%" muted autoplay loop playsinline></video> | <video src="https://github.com/user-attachments/assets/11d7c88f-86b5-461c-80c7-a02f51b4ac83" width="100%" muted autoplay loop playsinline></video> | <video src="https://github.com/user-attachments/assets/eccd2155-22e8-4ca2-9490-6755022434ac" width="100%" muted autoplay loop playsinline></video>  | 

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
cd ../../../
```
### Download Weights
Please download the weights for [BlumNet](https://github.com/cong-yang/BlumNet) and [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) following the instructions provided in corresponding repository, and run
```bash
cp -r Grounded/sam2 ./
```

### Usage
To automate the extraction of skeletons from your own video data, you must first provide the paths to the videos to be processed and their corresponding captions, as shown in  `../data/example/raw_metadata.csv`.
To run the example data：
```bash
bash run.sh
```
## 📧 Acknowledgement
Our implementation is based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio),  [BlumNet](https://github.com/cong-yang/BlumNet) and [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2). Thanks for their remarkable contribution and released code! If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.
