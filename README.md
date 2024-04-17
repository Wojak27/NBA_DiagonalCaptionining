This is the official repository diagonal attention video captioning network for paper *Towards Intelligent Sports Analysis: Multimodal Models for Video Captioning* 



## Setup
Some of the following instructions come from NSVA paper that can be found here:

# Code Overview

This document outlines the required scripts and PyTorch code for various tasks related to video analysis. It includes sections for downloading datasets, training and evaluation scripts, and accessing pre-trained weights.

## A. Download Pre-Processed NSVA Dataset

Information on how to download the dataset is provided from [NSVA](https://github.com/jackwu502/NSVA/tree/main/SportsFormer/data). Follow their instructions on how to download it. We also provide additional player tokens at: [change me](lik)

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
conda create -n damformer python=3.9.0 tqdm boto3 requests pandas
conda activate damformer
pip install torch==torch==1.13.1
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```

## Video captioning
Run the following code for training/evaluating from scratch video description captioning
```
cd SportsFormer
python -m torch.distributed.launch --nproc_per_node 4 main_task_caption.py --do_train --num_thread_reader 0 --epochs 20 --batch_size 48 --n_display 300 --train_csv data/ourds_train.44k.csv --val_csv data/ourds_JSFUSION_test.csv --data_path data/ourds_description_only.json --features_path data/ourds_videos_features.pickle --bbx_features_path data/cls2_ball_basket_sum_concat_original_courtline_fea.pickle --output_dir ckpt_ourds_caption --bert_model bert-base-uncased --do_lower_case --lr 3e-5 --max_words 30 --max_frames 30 --batch_size_val 1 --visual_num_hidden_layers 6 --decoder_num_hidden_layer 3 --cross_num_hidden_layers 3 --datatype ourds --stage_two --video_dim 768 --init_model weight/univl.pretrained.bin --train_tasks 0,0,1,0 --test_tasks 0,0,1,0

```

Or evalute with our pre-trained model at **weights** folder:
```
python -m torch.distributed.launch --nproc_per_node 4 main_task_caption.py --do_eval --num_thread_reader 0 --epochs 20 --batch_size 48 --n_display 300 --train_csv data/ourds_train.44k.csv --val_csv data/ourds_JSFUSION_test.csv --data_path data/ourds_description_only.json --features_path data/ourds_videos_features.pickle --bbx_features_path data/cls2_ball_basket_sum_concat_original_courtline_fea.pickle --output_dir ckpt_ourds_caption --bert_model bert-base-uncased --do_lower_case --lr 3e-5 --max_words 30 --max_frames 30 --batch_size_val 1 --visual_num_hidden_layers 6 --decoder_num_hidden_layer 3 --cross_num_hidden_layers 3 --datatype ourds --stage_two --video_dim 768 --init_model weight/best_model_vcap.bin --train_tasks 0,0,1,0 --test_tasks 0,0,1,0

```


| **Description Captioning**  | **C**  | **M** | **B@1** | **B@2** | **B@3** | **B@4** | **R_L** |
| -----------------------------| ------- | -------- |----------| ----------| ----------| ----------| ----------|
| **NSVA** | **1.1329**   | **0.2420**    | **0.5219**    | **0.4080**    |**0.3120**    |**0.2425**    |**0.5101** |
| **NSVA** | **1.1329**   | **0.2420**    | **0.5219**    | **0.4080**    |**0.3120**    |**0.2425**    |**0.5101** |
| **NSVA** | **1.1329**   | **0.2420**    | **0.5219**    | **0.4080**    |**0.3120**    |**0.2425**    |**0.5101** |
| **NSVA** | **1.1329**   | **0.2420**    | **0.5219**    | **0.4080**    |**0.3120**    |**0.2425**    |**0.5101** |


## Action recognition
Run the following code for training/evaluating from scratch video description captioning
```
cd SportsFormer
env CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./main_task_action_multifeat_multilevel.py
```

**Results** reproduced from pre-trained model 

| **Action Recognition**  | **SuccessRate**  | **mAcc.** | **mIoU** |
| -----------------------------| ------- | -------- |----------| 
| **Our full model Coarse** | **60.14**   | **61.20**    | **76.61**    |
| **Our full model Fine** | **46.88**   | **51.25**    | **57.08**    |
| **Our full model Event** | **37.67**   | **42.34**    | **46.45**    |

## Player identification
Run the following code for training/evaluating from scratch video description captioning
```
cd SportsFormer
env CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 ./main_task_player_multifeat.py
```

**Results** reproduced from pre-trained model 

| **Play Identification**  | **SuccessRate**  | **mAcc.** | **mIoU** |
| -----------------------------| ------- | -------- |----------|
| **Our full model** | **4.63**   | **6.97**    | **6.86**    | 

