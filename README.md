# Anonymize-Anyone
Anonymize Anyone: Bridging Ethnic Fairness in Text-to-Face Synthesis using a Diffusion Model

This repository contains the implementation of the following paper:


<be>
 
## :open_book: Overview
<!-- ![overall_structure](.figure8800.png) -->
<img src="./ForGit800.png" width="100%">

We propose **Anonymize-Anyone**, 

<br>


## :heavy_check_mark: Updates
- [03/2024] ...


## :hammer: Setup

### 1. Environment

```bash
conda create -n anonymize python=3.10.13
conda activate anonymize

git clone https://github.com/dusrb37/Anonymize-Anyone.git
cd Anonymize-Anyone
pip install -r requirements.txt
```

### 2. Run example
```bash
python test.py
```
Find the output in `./test/anonymized`

<be>

## :hammer: Get the Source Mask

### 1. Get segmentation mask

```bash
python test_RGB.py
```
Check the pre-trained model in `./segmentation/RGB/model/Retina_model.pth`
Find the output in `./segmentation/RGB/RGB_mask`

Please set the code environment by referring to the GitHub link below.

Detectron2 -> https://github.com/facebookresearch/detectron2

Install guide -> https://detectron2.readthedocs.io/en/latest/tutorials/install.html


### 2. Convert to binary mask
```bash
python convert_binary.py
```
Find the output in `./segmentation/RGB/binary_mask`

<be>


## :hammer: Train

### 1. Fine-tuning for inpainting model

```bash
python test_RGB.py
```
Check the pre-trained model in `./segmentation/RGB/model/Retina_model.pth`
Find the output in `./segmentation/RGB/RGB_mask`

Please set the code environment by referring to the GitHub link below.

Detectron2 -> https://github.com/facebookresearch/detectron2

Install guide -> https://detectron2.readthedocs.io/en/latest/tutorials/install.html


### 2. Convert to binary mask
```bash
python convert_binary.py
```
Find the output in `./segmentation/RGB/binary_mask`

<be>
1. Clone repo

   ```
   
   ```

2. Create conda environment.<br>

   ```
   
   ```

3. Install dependencies

   ```
   
   ```


## 💙: Acknowledgement

...
