This is the official repository diagonal attention video captioning network for paper *Towards Intelligent Sports Analysis: Multimodal Models for Video Captioning* 

## Setup
Some of the following instructions come from NSVA paper that can be found here:

# Code Overview

This document outlines the required scripts and PyTorch code for various tasks related to video analysis. It includes sections for downloading datasets, training and evaluation scripts, and accessing pre-trained weights.

## A. Download Pre-Processed NSVA Dataset

Information on how to download the dataset is provided from [NSVA](https://github.com/jackwu502/NSVA). Follow their instructions on how to download it. We also provide additional player tokens at: [change me](lik)

## B. Training/Evaluation Script

This section includes scripts for:

1. **Video Captioning**: Code and instructions for video captioning tasks.
2. **Action Recognition**: Scripts for training and evaluating action recognition models.
3. **Player Identification**: Guidance on implementing player identification in videos.

## C. Pre-trained Weights

Details on how to access and utilize pre-trained weights for the above tasks will be included here.

## Install Dependencies

The environment for running the provided scripts requires specific versions of Python and various libraries. The following commands set up the environment:

```bash
conda create -n sportsformer python=3.9.0 tqdm boto3 requests pandas
conda activate sportsformer
pip install torch==torch==1.13.1
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```
