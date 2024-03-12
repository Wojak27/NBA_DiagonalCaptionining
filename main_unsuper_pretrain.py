from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
import pickle5 as pickle
import time
import json
import argparse

import wandb
from dataloaders.dataloader_ourds_caption import OURDS_Caption_DataLoader
from dataloaders.dataloader_ourds_unsuper_pretrain import OURDS_PreTrain_DataLoader
from dataloaders.dataloader_ourds_unsuper_pretrain_audio import OURDS_PreTrainAudio_DataLoader
from dataloaders.dataloader_ourds_unsuper_pretrain_audio_44k import OURDS_PreTrainAudio_44k_DataLoader
from dataloaders.dataloader_ourds_unsuper_pretrain_audio_44k_scaled_1_5 import OURDS_PreTrainAudio44k_scaled_DataLoader
from main_task_caption import Args_Caption, init_device, init_model, load_model, prep_optimizer, save_model, set_seed_logger
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import UniVL
from modules.optimization import BertAdam
from dataloaders.dataloader_howto100m import Youtube_DataLoader
from dataloaders.dataloader_ourds_pretrain import OURDS_PT_DataLoader
from torch.utils.data import DataLoader
from util import get_logger
# torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='UniVL on Pretrain'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/HowTo100M_v1.csv', help='train csv')
    parser.add_argument('--features_path', type=str, default='feature', help='feature path for 2D features')
    parser.add_argument('--data_path', type=str, default='data/data.pickle', help='data pickle file path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--min_words', type=int, default=0, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
    parser.add_argument('--pretrain_enhance_vmodal', action='store_true', help="Enhance visual and other modalities when pretraining.")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_model", default="pytorch_model.bin.checkpoint", type=str, required=False,
                        help="Save the last model as a checkpoint.")

    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_pretrain:
        raise ValueError("`do_pretrain` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    args.checkpoint_model = '{}_{}_{}_{}.checkpoint'.format(args.checkpoint_model, args.bert_model, args.max_words, args.max_frames)

    return args



def dataloader_ourds_train(args, tokenizer):
    ourds_dataset = OURDS_PreTrain_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        mask_prob=args.mask_prob,
    )

    # train_sampler = torch.utils.data.Sampler(ourds_dataset)
    dataloader = DataLoader(
        ourds_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(ourds_dataset), None

def dataloader_ourds_audio_train(args, tokenizer):
    ourds_dataset = OURDS_PreTrainAudio_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.audio_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        mask_prob=args.mask_prob,
    )

    # train_sampler = torch.utils.data.Sampler(ourds_dataset)
    dataloader = DataLoader(
        ourds_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(ourds_dataset), None

def dataloader_ourds_audio_44k_train(args, tokenizer):
    ourds_dataset = OURDS_PreTrainAudio_44k_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.audio_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        mask_prob=args.mask_prob,
        extra_feat_annotation=args.extra_feat_annotation,
    )

    # train_sampler = torch.utils.data.Sampler(ourds_dataset)
    dataloader = DataLoader(
        ourds_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(ourds_dataset), None

def dataloader_ourds_audio_44k_scaled_train(args, tokenizer):
    ourds_dataset = OURDS_PreTrainAudio44k_scaled_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.audio_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        mask_prob=args.mask_prob,
        extra_feat_annotation=args.extra_feat_annotation,
    )

    # train_sampler = torch.utils.data.Sampler(ourds_dataset)
    dataloader = DataLoader(
        ourds_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(ourds_dataset), None

def dataloader_ourds_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_PreTrain_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks,
        mask_prob=args.mask_prob,
    )

    test_sampler = SequentialSampler(ourds_testset)
    dataloader_ourds = DataLoader(
        ourds_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False
    )
    return dataloader_ourds, len(ourds_testset)

def dataloader_pretrain(args, tokenizer, only_sim=False):
    if args.local_rank == 0:
        logger.info('Loading captions: {}'.format(args.data_path))
    #print(args.data_path)
    data_dict = json.load(open(args.data_path, 'r'))
    if args.local_rank == 0:
        logger.info('Done, data_dict length: {}'.format(len(data_dict)))

    dataset = OURDS_PT_DataLoader(
        csv=args.train_csv,
        features_path=args.features_path,
        data_dict=data_dict,
        min_time=args.min_time,
        max_words=args.max_words,
        min_words=args.min_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        n_pair=args.n_pair,
        max_frames=args.max_frames,
        use_mil=args.use_mil,
        only_sim=only_sim,
        sampled_use_mil=args.sampled_use_mil,
        pretrain_enhance_vmodal=args.pretrain_enhance_vmodal,
        video_dim=args.video_dim,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )

    return dataloader, len(dataset), sampler


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    print(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, masked_bbx, trimmed_video, trimmed_audio = batch
        
        decoder_mask = ((masked_video == 0)*video_mask).float()   
        decoder_mask_audio = ((masked_bbx == 0)*bbx_mask).float()   
        # decoder_mask = video_mask.float()
        
        

        loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=decoder_mask,
                     output_caption_ids=pairs_output_caption_ids,task_type=task_type, bbx=bbx.float(), bbx_mask=bbx_mask.float(), masked_bbx=masked_bbx.float(), trimmed_video=trimmed_video.float(), trimmed_audio=trimmed_audio.float(), decoder_mask_audio=decoder_mask_audio)
        
        if wandb is not None:
            wandb.log({"PreTRLoss/train": loss})

        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch(args, model, train_dataloader, device, n_gpu):
    global logger
    torch.cuda.empty_cache()
    model.eval()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    print(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, masked_bbx = batch
        
        decoder_mask = ((masked_video == 0)*video_mask).float()   
        

        loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=decoder_mask,
                     output_caption_ids=pairs_output_caption_ids,task_type=task_type, bbx=bbx.float(), bbx_mask=bbx_mask.float())
        
        if wandb is not None:
            wandb.log({"PreTRLoss/train": loss})


        total_loss += float(loss)

    total_loss = total_loss / len(train_dataloader)
    return total_loss

DATALOADER_DICT = {}
DATALOADER_DICT["ourds"] = {"train":dataloader_ourds_audio_44k_scaled_train, "val":dataloader_ourds_test}

def main(args):
    global logger
    if args == None:
        args = get_args()
    args, logger = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank)
    only_sim = model.module._stage_one if hasattr(model, 'module') else model._stage_one
    
    assert args.datatype in DATALOADER_DICT
    args.video_feature = pickle.load(open(args.features_path, 'rb'))
    args.video_bbx_feature = pickle.load(open(args.bbx_features_path, 'rb'))
    args.audio_feature = pickle.load(open(args.audio_features_path, 'rb'))
    #audio_features_path
    
    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)

    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    global_step = 0
    epoch = -1
    last_optim_state = None
    if args.load_checkpoint:
        epoch, global_step, last_optim_state, model = load_model(epoch, args, n_gpu, device, model, global_step=global_step)
        epoch += 1
        if args.local_rank == 0:
            logger.warning("Will continue to epoch: {}".format(epoch))
    epoch = 0 if epoch < 0 else epoch

    coef_lr = args.coef_lr
    if args.init_model:
        coef_lr = 1.0

    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
    if last_optim_state is not None:
        optimizer.load_state_dict(last_optim_state)

    if args.local_rank == 0:
        logger.info("***** Running pretraining *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        
    conf ={}
    for key,value in args.__dict__.items():
        if(key in ["video_feature", "video_bbx_feature", "cos", "device", "audio_feature"]):
            continue
        conf[key] = value
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Multimodal-Fusion-Bottleneck",
        name="{}_{}_{}_{}_{}_bottley_{}_bottdim_{}_enc_{}_cross_{}_declay_{}_conv_{}".format(args.task_type, args.datatype, args.bert_model, args.lr, args.batch_size, args.bottleneck_fusion_layers, args.bottleneck_dim, args.visual_num_hidden_layers, args.cross_num_hidden_layers, args.decoder_num_hidden_layers ,args.bottleneck_use_conv),
        # track hyperparameters and run metadata
        config=conf,
    ) 
    
    # artifact_data = wandb.Artifact('data', type='dataset')
    

    iter_ls_ = [itm for itm in range(args.epochs) if itm >= epoch]
    for epoch in iter_ls_:
        print(args.local_rank)
        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                           scheduler, global_step, local_rank=args.local_rank)
        
        

        if args.local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            if epoch % 20 == 0:
                save_model(epoch, args, model, type_name="pretrain")
            
        # if epoch > -1:
        #     logger.info("***** Running validation *****")
        #     val_loss = eval_epoch(args, model, val_dataloader, device, n_gpu)
            

if __name__ == "__main__":
    args = None
    
    vis_layers = 6
    cross_layers = 3
    mask_prob = 0.95
    diag_masking = False
    bs = 1024
    args = Args_Caption(features_dir="data", do_eval=False, task="unsup-pretrain-audio", output_dir="MAEpretrained_models/out_pretrain_{}e_{}c_1d_audio_kinetics_scaled_1_5_Prob{}_{}BS{}".format(vis_layers,cross_layers,mask_prob,bs, "_nodiag" if not diag_masking else ""))
    
    args.mask_prob = mask_prob
    args.load_checkpoint = False
    args.batch_size_val = 32
    args.bottleneck_dim = None
    args.datatype = "ourds"
    args.video_dim = 768
    args.max_frames = 30
    args.max_words = 30
    args.decoder_num_hidden_layers = 1
    args.visual_num_hidden_layers = vis_layers
    args.bottleneck_fusion_layers = None
    # args.features_path = "data/pretrain_ourds_videos_timesformer_features.pickle"
    args.features_path = "data/kinetics_ourds_videos_timesformer_features.pickle"
    # args.audio_features_path = "data/pretrain_ourds_audio_VGGish_scaled_1_5_features.pkl"
    args.audio_features_path = "data/kinetics_ourds_videos__VGGish_features.pickle"
    args.cross_num_hidden_layers = cross_layers
    # args.init_model = "/home/karolwojtulewicz/code/NSVA/out_pretrain_6e_0b_3c_1d_audio/pytorch_model.bin.pretrain.250"
    args.init_model = None
    args.unsup_pretrain = True
    args.epochs = 405
    args.lr = 0.00003
    args.lr_decay = 0.9
    args.batch_size = bs
    args.multibbxs = False
    args.cross_masking = None
    args.multi_mae = False
    args.full_cross = False
    args.visual_use_diagonal_masking = diag_masking
    args.extra_feat_annotation = "data/extra_videos.json"
    main(args)