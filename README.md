# CLIP Crossmodal Retrieval Task

## Overview
This repository contains code and instructions for conducting a crossmodal retrieval task using OpenAI's CLIP model (https://github.com/openai/CLIP.git). The task involves retrieving relevant images given a textual query and vice versa ,using the MSCOCO 2017 and Flickr30k datasets. The retrieval is evaluated for both zero-shot and fine-tune scenarios.

## Dependencies
- PyTorch
- torchvision
- transformers
- CLIP (ensure you have the latest version)

Install the dependencies using:
```bash
pip install torch torchvision transformers git+https://github.com/openai/CLIP.git
