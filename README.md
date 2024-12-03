<div align="center">

# MaskRIS: Semantic Distortion-aware Data Augmentation for Referring Image Segmentation

**[Minhyun Lee](https://scholar.google.com/citations?user=2hUlCnQAAAAJ&hl=ko)<sup>1,2</sup>, 
[Seungho Lee](https://scholar.google.com/citations?user=vUM0nAgAAAAJ)<sup>1</sup>, 
[Song Park](https://scholar.google.co.kr/citations?user=VR1c0H8AAAAJ&hl=ko)<sup>2</sup>, 
[Dongyoon Han](https://sites.google.com/site/dyhan0920/)<sup>2</sup>, 
[Byeongho Heo](https://sites.google.com/view/byeongho-heo/home)<sup>2</sup>, 
[Hyunjung Shim](https://scholar.google.co.kr/citations?user=KB5XZGIAAAAJ&hl=ko)<sup>3</sup>**

**<sup>1</sup>Yonsei University, 
<sup>2</sup> [NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab), 
<sup>3</sup>KAIST**

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://github.com/naver-ai/maskris/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arxiv-green)](https://arxiv.org/abs/2411.19067)

</div>

Official PyTorch implementation of "MaskRIS: Semantic Distortion-aware Data Augmentation for Referring Image Segmentation" | [arxiv](https://www.arxiv.org/abs/2411.19067).

### Abstract
Referring Image Segmentation (RIS) is a vision-language task that identifies and segments objects in images based on free-form text descriptions. This study investigates effective data augmentation strategies and proposes a novel framework called Masked Referring Image Segmentation (MaskRIS). MaskRIS employs image and text masking to improve model robustness against occlusions and incomplete information. Experimental results show that MaskRIS integrates with existing models and achieves state-of-the-art performance on RefCOCO, RefCOCO+, and RefCOCOg datasets in both fully supervised and weakly supervised settings.

### Updates
- Dec 1, 2024: Arxiv paper is released.

### Requirements
This code is tested with:
- Python 3.8
- PyTorch 1.11.0

Other dependencies are listed in `requirements.txt`.

### Datasets
#### 1. Text Annotations
- **RefCOCO Series Annotations**
  - Download locations:
    - RefCOCO, RefCOCO+, RefCOCOg: Follow instructions in `.refer/README.md`
    - Combined annotations (refcocom): [Google Drive Link](https://drive.google.com/file/d/1_WnCziCIVHXpWYDsIsHbxzH_KCiYhflo/view?usp=sharing)

#### 2. Image Data
- **COCO Dataset**
  - Source: [COCO Official Website](https://cocodataset.org/#download)
  - Required file: `train_2014.zip` (83K images, 13GB)
  - Instructions:
    1. Download from the first link: "2014 Train images [83K/13GB]"
    2. Extract the downloaded `train_2014.zip` file

#### 3. Data structure
- Data paths should be as follows:
  ```
  DATA_PATH
      ├── refcocom
      ├── train2014
      └── refer
          ├── refcoco
          ├── refcoco+
          └── refcocog
  ```

### Pretrained Models
- Swin-B: [Swin Transformer-Base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)
- BERT-B: [BERT-Base](https://huggingface.co/bert-base-uncased/tree/main)


### Usage
By default, we use fp16 training for efficiency. To train a model on refcoco with 2 GPUs, 
modify `DATA_PATH`, `REFER_PATH`, `SWIN_PATH`, 
and `OUTPUT_PATH` in `scripts/script.sh` then run:
```
bash scripts/script.sh
```
You can change `DATASET` to `refcoco+`/`refcocog`/`refcocom` for training on different datasets. 
Note that for RefCOCOg, there are two splits (umd and google). You should add `--splitBy umd` or `--splitBy google` to specify the split.

## References
This repo is mainly built based on [CARIS](https://github.com/lsa1997/CARIS) and [mmdetection](https://github.com/open-mmlab/mmdetection). Thanks for their great work!

### License

```
MaskRIS
Copyright (c) 2024-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
