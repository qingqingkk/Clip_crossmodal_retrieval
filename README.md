# CLIP Crossmodal Retrieval Task

## Overview
This repository contains code and instructions for conducting a crossmodal retrieval task using OpenAI's CLIP model (https://github.com/openai/CLIP.git). The task involves retrieving relevant images given a textual query and vice versa ,using the MSCOCO 2017 and Flickr30k datasets. The retrieval is evaluated for both zero-shot and fine-tune scenarios.

# Dataset Preparation
1. **Download MSCOCO 2017:**
   - Utilize the PyTorch library 'cococaption' to process the dataset, ensuring that each image is associated with five textual descriptions.
   - Use MSCOCO val2017 as the zero-shot test dataset and fine-tune test dataset.
   - Split MSCOCO train2017 into training and validation subsets for training purposes.

2. **Download Flickr30k dataset:**
   - Preprocess the datasets by defining a class to create a unified format for images and textual descriptions.
   - Split the data into training, validation, and test sets（for zero-shot & fine-tune).

# Evaluation metrices
Recall@k
MAP@k

# Usage
1. **zero-shot:**
!python main.py --data_path ./result/encoded_data/flickr --task zero-shot --scheduler ReduceLROnPlateau --data_type encoded --train_mode with_adapter --loss_type cos_embedd --max_epochs 50
1. **Fine-Tune:**
!python main.py --data_path ./result/encoded_data/flickr --task fine_tune --scheduler ReduceLROnPlateau --data_type encoded --train_mode with_adapter --loss_type cos_embedd --max_epochs 50


Copy the github address:
!git clone https://github.com/qingqingkk/Clip_crossmodal_retrieval.git
