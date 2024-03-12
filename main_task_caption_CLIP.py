from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
from dataloaders.dataloader_ourds_CLIP import OURDS_CLIP_DataLoader

from dataloaders.dataloader_ourds_caption import OURDS_Caption_DataLoader
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_anet_caption_flow_pickle import ActivityNet_Caption_DataLoader_Flow_Pickle
from tqdm import tqdm
import wandb
from datetime import datetime
from distutils.log import debug
from posixpath import split

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
from nlgeval import NLGEval
import time
import argparse
import json
from dataloaders.dataloader_anet_caption_audio import ActivityNet_Caption_DataLoader_Audio
from dataloaders.dataloader_anet_caption_flow import ActivityNet_Caption_DataLoader_Flow
from dataloaders.dataloader_msrvtt_caption_audio import MSRVTT_Caption_DataLoader_Audio
from dataloaders.dataloader_ourds_caption_audio_bbx import OURDS_Caption_Audio_BBX_DataLoader
from dataloaders.dataloader_ourds_caption_lang import OURDS_Caption_Lang_DataLoader
# from dataloaders.dataloader_msrvtt_caption_audio import MSRVTT_Caption_DataLoader_Audio
from dataloaders.dataloader_ourds_q_and_a import OURDS_QA_DataLoader
from dataloaders.dataloader_ourds_q_and_a_raw import OURDS_QA_RAW_DataLoader
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import UniVL
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
# from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
# from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
# from dataloaders.dataloader_ourds_caption import OURDS_Caption_DataLoader
from util import get_logger
from torch import nn
from torchsummary import summary
import pickle5 as pickle
import re
# torch.distributed.init_process_group(backend="nccl")

global logger

wandb.login()

# from torch.utils.tensorboard import SummaryWriter

def get_args(description='UniVL on Caption Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--use_prefix_tuning", action='store_true', help="Whether to use prefix tuning.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='./ourds_train.44k.csv', help='')
    parser.add_argument('--val_csv', type=str, default='./ourds_JSFUSION_test.csv', help='')
    parser.add_argument('--data_path', type=str, default='./ourds_data_timesformer/ourds_description.json',
                        help='caption and transcription pickle file path')
    parser.add_argument('--features_path', type=str, default='./ourds_videos_features.pickle',
                        help='feature path for 2D features')


    parser.add_argument('--bbx_features_path', type=str, default='./ourds_bbx_data/ourds_videos_features.pickle',
                        help='feature path for 2D features')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=4, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=300, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=768, help='video feature dimension') # switch between 768 and 1024?

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=30, help='')
    parser.add_argument('--max_frames', type=int, default=30, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')


    parser.add_argument("--output_dir", default='/media/chris/hdd1/UniVL_processing_code/ourds_data_timesformer_bbx/ckpt_ourds_caption', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False, help="Bert pre-trained model")
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

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `caption` to finetune.")
    parser.add_argument("--datatype", default="ourds", type=str, help="Point the dataset `youcook` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--context_only', action='store_true', help="Whether use contextual video feature, e.g., S3D or TimeSformer feature only")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=2, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=6, help="Layer NO. of decoder.")

    parser.add_argument('--train_tasks', default=[0,0,1,0],type=lambda s: [int(item) for item in s.split(',')], help="train with specific tasks: 1 for yes, 0 for no")
    parser.add_argument('--test_tasks',default=[0,0,1,0], type=lambda s: [int(item) for item in s.split(',')], help="test with specific tasks: 1 for yes, 0 for no")
    parser.add_argument('--t1_postprocessing', action='store_true', help="Whether postprocess output with action type")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
    args = parser.parse_args()


    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    # args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args, logger

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    # model = model.float()
    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_youcook_train(args, tokenizer):
    youcook_dataset = Youcook_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_test(args, tokenizer):
    youcook_testset = Youcook_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset)


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_anetc3d_caption_audio_train(args, tokenizer):
    anet_dataset = ActivityNet_Caption_DataLoader_Audio(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/anet_c3d_train.pkl",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/data/activity-net/structured-symlinks/aggregated_audio/vggish-audio-raw.pickle",
        json_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train.json",
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )
    dataloader = DataLoader(
        anet_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(anet_dataset), None

def dataloader_anetc3d_caption_audio_test(args, tokenizer, split_type="test"):
    if split_type == "test":
        json_path = "/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_2.json"
    else:
        json_path = "/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_1.json"
    anet_testset = ActivityNet_Caption_DataLoader_Audio(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/anet_c3d_val.pkl",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/data/activity-net/structured-symlinks/aggregated_audio/vggish-audio-raw.pickle",
        json_path=json_path,
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(anet_testset)
    dataloader_anet = DataLoader(
        anet_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_anet, len(anet_testset)

def dataloader_anetc3d_caption_flow_train(args, tokenizer):
    anet_dataset = ActivityNet_Caption_DataLoader_Flow(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/env_c3d",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/training",
        json_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train.json",
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )
    dataloader = DataLoader(
        anet_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(anet_dataset), None

def dataloader_anetc3d_caption_flow_test(args, tokenizer, split_type="test"):
    if split_type == "test":
        json_path = "/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_2.json"
    else:
        json_path = "/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_1.json"
    anet_testset = ActivityNet_Caption_DataLoader_Flow(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/env_c3d",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/validation",
        json_path=json_path,
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(anet_testset)
    dataloader_anet = DataLoader(
        anet_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_anet, len(anet_testset)

def dataloader_anetc3d_caption_flow_pickle_train(args, tokenizer):
    anet_dataset = ActivityNet_Caption_DataLoader_Flow_Pickle(
        annotations_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/train_ids.json",
        v_features_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/anet_c3d_train.pkl",
        a_features_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/anet_flow_train.pkl",
        json_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/train.json",
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )
    dataloader = DataLoader(
        anet_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(anet_dataset), None

def dataloader_anetc3d_caption_flow_pickle_test(args, tokenizer, split_type="test"):
    if split_type == "test":
        json_path = "/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/val_2.json"
    else:
        json_path = "/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/val_1.json"
    anet_testset = ActivityNet_Caption_DataLoader_Flow_Pickle(
        annotations_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/val_ids.json",
        v_features_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/anet_c3d_val.pkl",
        a_features_path="/proj/nba_multimodal_video/users/x_karwo/NBA_paper/NSVA/data/activitynet_captions/anet_caption/anet_flow_val.pkl",
        json_path=json_path,
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(anet_testset)
    dataloader_anet = DataLoader(
        anet_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_anet, len(anet_testset)

def dataloader_anet_caption_audio_train(args, tokenizer):
    anet_dataset = ActivityNet_Caption_DataLoader_Audio(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/anet_caption_tsp.pkl",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/data/activity-net/structured-symlinks/aggregated_audio/vggish-audio-raw.pickle",
        json_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train.json",
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )
    dataloader = DataLoader(
        anet_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(anet_dataset), None

def dataloader_anet_caption_audio_test(args, tokenizer, split_type="test"):
    if split_type == "test":
        json_path = "/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_2.json"
    else:
        json_path = "/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_1.json"
    anet_testset = ActivityNet_Caption_DataLoader_Audio(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/val_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/anet_caption_tsp.pkl",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/data/activity-net/structured-symlinks/aggregated_audio/vggish-audio-raw.pickle",
        json_path=json_path,
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(anet_testset)
    dataloader_anet = DataLoader(
        anet_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_anet, len(anet_testset)

def dataloader_msrvtt_audio_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader_Audio(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/data/MSRVTT/high-quality/structured-symlinks/train_list_full.txt",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/msrvtt_videos_features.pickle",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/data/MSRVTT/high-quality/structured-symlinks/aggregated_audio_feats/Audio_MSRVTT_new.pickle",
        json_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/MSRVTT_data.json",
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), None

def dataloader_msrvtt_audio_test(args, tokenizer, split_type="test",):
    if split_type == "test":
        annotations_path = "/4TBSSD_permanent/collaborative-experts/data/msrvtt/data/MSRVTT/high-quality/structured-symlinks/test_list_full.txt"
    else:
        annotations_path = "/4TBSSD_permanent/collaborative-experts/data/msrvtt/data/MSRVTT/high-quality/structured-symlinks/val_list_full.txt"
    msrvtt_testset = MSRVTT_Caption_DataLoader_Audio(
        annotations_path=annotations_path,
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/msrvtt_videos_features.pickle",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/data/MSRVTT/high-quality/structured-symlinks/aggregated_audio_feats/Audio_MSRVTT_new.pickle",
        json_path="/4TBSSD_permanent/collaborative-experts/data/msrvtt/MSRVTT_data.json",
        max_words=args.max_words,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_ourds_train(args, tokenizer):
    ourds_dataset = OURDS_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks
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
    ourds_testset = OURDS_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks
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

def dataloader_ourds_lang_train(args, tokenizer):
    ourds_dataset = OURDS_Caption_Lang_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks
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

def dataloader_ourds_lang_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_Caption_Lang_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks
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

def dataloader_ourds_CLIP_train(args, tokenizer):
    ourds_dataset = OURDS_CLIP_DataLoader(
        csv_path=args.train_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/new_ourds_description_only.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        use_answer=args.use_answer,
        is_pretraining=args.do_pretrain,
        use_random_embeddings=args.use_random_embeddings,
        num_samples=100000,
        mask_prob=0.25,
        only_players=True,
        use_real_name=False,
        player_embedding_order=args.player_embedding_order,
        use_BBX_features=args.use_BBX_features,
        player_embedding=args.player_embedding,
        max_rand_players=args.max_rand_players
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

def dataloader_ourds_CLIP_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_CLIP_DataLoader(
        csv_path=args.val_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/new_ourds_description_only.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        use_random_embeddings=args.use_random_embeddings,
        split_type=split_type,
        split_task = args.test_tasks,
        use_answer=args.use_answer,
        is_pretraining=args.do_pretrain,
        num_samples=0,
        only_players=True,
        use_real_name=False,
        player_embedding_order=args.player_embedding_order,
        use_BBX_features=args.use_BBX_features,
        player_embedding=args.player_embedding,
        max_rand_players=args.max_rand_players
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

def dataloader_ourds_QA_train(args, tokenizer):
    ourds_dataset = OURDS_QA_DataLoader(
        csv_path=args.train_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/ourds_description_only.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        use_answer=args.use_answer,
        is_pretraining=args.do_pretrain,
        num_samples=100000,
        mask_prob=0.25,
        only_players=True,
        use_real_name=False
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

def dataloader_ourds_QA_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_QA_DataLoader(
        csv_path=args.val_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/video_qa_unique_with_answers.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks,
        use_answer=args.use_answer,
        is_pretraining=args.do_pretrain,
        num_samples=0,
        only_players=True,
        use_real_name=False
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

def dataloader_ourds_QA_raw_train(args, tokenizer):
    ourds_dataset = OURDS_QA_RAW_DataLoader(
        csv_path=args.train_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/video_qa_unique_with_answers.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        use_answer=args.use_answer,
        is_pretraining=args.do_pretrain,
        num_samples=0,
        mask_prob=0.25,
        only_players=False,
        use_real_name=False,
        fine_tune_extractor=True,
        videos_filepath="/4TBSSD_permanent/NSVA/downscaled_videos"
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

def dataloader_ourds_QA_raw_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_QA_RAW_DataLoader(
        csv_path=args.val_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/video_qa_unique_with_answers.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks,
        use_answer=args.use_answer,
        is_pretraining=args.do_pretrain,
        num_samples=0,
        only_players=False,
        use_real_name=False,
        fine_tune_extractor=True,
        videos_filepath="/4TBSSD_permanent/NSVA/downscaled_videos"
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

def dataloader_ourds_audio_bbx_train(args, tokenizer):
    ourds_dataset = OURDS_Caption_Audio_BBX_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        audio_feature=args.audio_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks
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

def dataloader_ourds_audio_bbx_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_Caption_Audio_BBX_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        audio_feature=args.audio_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks
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

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))

    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0, writer=None):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
    
    for step, batch in progress_bar:

        

        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        if args.datatype == "ourds-audio-bbx":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, audio, audio_mask, bbx, bbx_mask = batch

            loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                        pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                        masked_video=masked_video, video_labels_index=video_labels_index,
                        input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                        output_caption_ids=pairs_output_caption_ids,task_type=task_type, bbx=bbx.float(), bbx_mask=bbx_mask.float(), audio=audio.float(), audio_mask=audio_mask.float())
        elif args.datatype == "ourds-QA":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, masked_bbx, bbx_labels_index, player_IDs = batch
            
            loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                        pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                        masked_video=masked_video, video_labels_index=video_labels_index,
                        input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                        output_caption_ids=pairs_output_caption_ids,task_type=task_type, bbx=bbx.float(), bbx_mask=bbx_mask.float(), masked_bbx=masked_bbx.float(), bbx_labels_index=bbx_labels_index, player_IDs=player_IDs)
        else:
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, masked_bbx, bbx_labels_index = batch
            
            loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                        pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                        masked_video=masked_video, video_labels_index=video_labels_index,
                        input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                        output_caption_ids=pairs_output_caption_ids,task_type=task_type, bbx=bbx.float(), bbx_mask=bbx_mask.float(), masked_bbx=masked_bbx.float(), bbx_labels_index=bbx_labels_index)
        
        progress_bar.set_description(f"Epoch {epoch+1}, Step {step+1}, Loss: {float(loss):.4f}")
        if writer is not None:
            writer.add_scalar("Loss/train", loss, epoch)
        if wandb is not None:
            wandb.log({"Loss/train": loss})

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()
            
            del video, bbx
            global_step += 1
            # if global_step % log_step == 0 and local_rank == 0:
            #     logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
            #                 args.epochs, step + 1,
            #                 len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
            #                 float(loss),
            #                 (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
            #     start_time = time.time()
    progress_bar.close()
    
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt,task_type_rpt, bbx_output_rpt, bbx_mask_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_bbx_output_rpt = collect_active_part(bbx_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    
    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_bbx_mask_rpt = collect_active_part(bbx_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
    active_task_type_rpt = collect_active_part(task_type_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    return (active_sequence_output_rpt, active_visual_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt,active_task_type_rpt, active_bbx_output_rpt, active_bbx_mask_rpt), \
           active_inst_idx_to_position_map

def collate_active_infoVL(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt,task_type_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    
    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
    active_task_type_rpt = collect_active_part(task_type_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    return (active_sequence_output_rpt, active_visual_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt,active_task_type_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None,task_type = None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples,task_type = task_type):
        
        if args.fine_tune_extractor == False:
            sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, bbz_output_rpt, bbx_mask = input_tuples
        else:
            sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        if args.fine_tune_extractor == False:
            dec_output = decoder(sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt,
                                video_mask_rpt, next_decoder_ids, next_decoder_mask, bbz_output_rpt, bbx_mask, shaped=True, get_logits=True,task_type = task_type)
        else:
            dec_output = decoder(sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt,
                                video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True,task_type = task_type)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._stage_one:
        return 0.

    test_tasks = [id for id, task in enumerate(args.test_tasks) if task ==1] 
    result_list_byTask = {t:[] for t in test_tasks}
    caption_list_byTask = {t:[] for t in test_tasks}
    all_result_lists = []
    all_caption_lists = []
    model.eval()
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Training")
    
    for b_id, batch in progress_bar:
        


        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        if args.datatype == "ourds-audio-bbx":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type,audio,audio_mask, bbx, bbx_mask = batch
            bbx, bbx_mask, audio, audio_mask = audio.float(), audio_mask.float(), bbx.float(), bbx_mask.float()
        elif args.datatype == "ourds-QA":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, _, _,_ = batch
        elif args.datatype == "ourds-QA-raw":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, _, _ = batch
        elif args.datatype == "ourds-CLIP":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, _, _ = batch
        else:
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask = batch


        with torch.no_grad():
            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask,task_type = task_type)
            if args.fine_tune_extractor == False:
                if model.multibbxs:
                    batch_sz,_,bbx_num,max_frame_num,fea_sz = bbx.shape
                    bbx = bbx.permute((0, 1, 3, 2, 4)).reshape(batch_sz,_,max_frame_num,fea_sz*bbx_num)
                    #bbx = model.bbx_fea_fusion_two(bbx)
                if "audio" in args.task_type:
                    bbx = model.audio_embed(bbx)
                    bbx = bbx.squeeze(1)
                    bbx_output = model.get_bbx_output(bbx.squeeze(1), bbx_mask.squeeze(1), shaped=True)
                else:
                    bbx_output = model.get_bbx_output(bbx.squeeze(1), bbx_mask.squeeze(1))

            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = sequence_output.device
            n_inst, len_s, d_h = sequence_output.size()
            _, len_v, v_h = visual_output.size()
            # visual_output = visual_output.unsqueeze(1)

            if args.fine_tune_extractor == False:
                decoder = model.decoder_caption
            else:
                decoder = model.decoder_captionVL

            # Note: shaped first, then decoder need the parameter shaped=True
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            input_mask = input_mask.view(-1, input_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            if args.fine_tune_extractor == False:
                bbx_mask = bbx_mask.view(-1, bbx_mask.shape[-1])

            # The following line need to be changed soon
            if args.use_prefix_tuning !=False:
                video_mask = torch.cat([torch.zeros(video_mask.size(0),model.visual_config.preseqlens[0],dtype=video_mask.dtype,device=video_mask.device),video_mask],dim=1)

            sequence_output_rpt = sequence_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            if args.fine_tune_extractor == False:
                bbx_output_rpt = bbx_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)

            input_ids_rpt = input_ids.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            input_mask_rpt = input_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            if args.fine_tune_extractor == False:
                bbx_mask_rpt = bbx_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            task_type_rpt = task_type.repeat(n_bm)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1):
                if args.fine_tune_extractor == False:
                    active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                            len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                            (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, bbx_output_rpt, bbx_mask_rpt), task_type = task_type_rpt)
                else:
                    active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                            len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                            (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt), task_type = task_type_rpt)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>
                
                if args.fine_tune_extractor == False:
                    (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt,task_type_rpt, bbx_output_rpt, bbx_mask_rpt), \
                    inst_idx_to_position_map = collate_active_info((sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt,task_type_rpt,bbx_output_rpt, bbx_mask_rpt),
                                                                inst_idx_to_position_map, active_inst_idx_list, n_bm, device
                                                                )
                else:
                    (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt,task_type_rpt), \
                    inst_idx_to_position_map = collate_active_infoVL((sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt,task_type_rpt),
                                                                inst_idx_to_position_map, active_inst_idx_list, n_bm, device
                                                                )

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                
                if args.t1_postprocessing:
                    match = re.search(r'action[0-9]+[ ]*', decode_text)
                    if match !=None:

                        match_action_type = match.group(0).replace(' ','')
                        replace_type = action_token2full_description[match_action_type].replace(' unknown','').replace('made shot ','').replace('missed shot','miss')
                        decode_text  = decode_text.replace(match_action_type, replace_type)
                result_list_byTask[task_type.tolist()[re_idx]].append(decode_text)
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                caption_list_byTask[task_type.tolist()[re_idx]].append(decode_text)
                all_caption_lists.append(decode_text)
    progress_bar.close()

    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.data_dict[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    all_caption_lists = None
    if args.datatype == "msrvtt":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]

    # Evaluate
    for task in test_tasks:
        if all_caption_lists is None:
            r  = [caption_list_byTask[task]]
            h  = result_list_byTask[task]
        else:
            r = all_caption_lists
            h = all_result_lists
        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=r, hyp_list=h)
        logger.info(">>> TASK {:d}: BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                    format(task, metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
        logger.info(">>> TASK {:d}: METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(task, metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))

        Scores = metrics_nlg
    return Scores

DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_train, "val":dataloader_youcook_test}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_audio_train, "val":dataloader_msrvtt_audio_test}
DATALOADER_DICT["anet"] = {"train":dataloader_anet_caption_audio_train, "val":dataloader_anet_caption_audio_test}
DATALOADER_DICT["anet_c3d"] = {"train":dataloader_anetc3d_caption_audio_train, "val":dataloader_anetc3d_caption_audio_test}
DATALOADER_DICT["anet_c3d_flow"] = {"train":dataloader_anetc3d_caption_flow_train, "val":dataloader_anetc3d_caption_flow_test}
DATALOADER_DICT["ourds"] = {"train":dataloader_ourds_train, "val":dataloader_ourds_test}
DATALOADER_DICT["ourds-lang"] = {"train":dataloader_ourds_lang_train, "val":dataloader_ourds_lang_test}
DATALOADER_DICT["ourds-QA"] = {"train":dataloader_ourds_QA_train, "val":dataloader_ourds_QA_test}
DATALOADER_DICT["ourds-QA-raw"] = {"train":dataloader_ourds_QA_raw_train, "val":dataloader_ourds_QA_raw_test}
DATALOADER_DICT["ourds-audio-bbx"] = {"train":dataloader_ourds_audio_bbx_train, "val":dataloader_ourds_audio_bbx_test}

DATALOADER_DICT["ourds-CLIP"] = {"train":dataloader_ourds_CLIP_train, "val":dataloader_ourds_CLIP_test}

# DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_train, "val":dataloader_youcook_test}
# DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_audio_train, "val":dataloader_msrvtt_audio_test}
# DATALOADER_DICT["anet"] = {"train":dataloader_anet_caption_audio_train, "val":dataloader_anet_caption_audio_test}
# DATALOADER_DICT["anet_c3d"] = {"train":dataloader_anetc3d_caption_audio_train, "val":dataloader_anetc3d_caption_audio_test}
DATALOADER_DICT["anet_c3d_flow"] = {"train":dataloader_anetc3d_caption_flow_pickle_train, "val":dataloader_anetc3d_caption_flow_pickle_test}


action_list = json.load(open('{}/data/action_list.json'.format(os.environ["DIR_PATH"]), 'r'))
action_token2full_description = {'action%s'%a_idx:a_l.lower().replace('_',' ').replace('-',' ') for a_idx, a_l in enumerate(action_list)}


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def init_training_caption(args):
    global logger
    if args == None:
        args = get_args()
        
    if args.do_eval:
        output_dir="attention_scores"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Delete the existing attention plots
            for filename in os.listdir(output_dir):
                if filename.endswith('.npy'):
                    os.remove(os.path.join(output_dir, filename))
    
    if isinstance(args, dict):
        args = DictToObject(args)
    args, _ = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer_original = BertTokenizer.from_pretrained(args.bert_model+'-original', do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank)
    
    
    for action_token, action_description in action_token2full_description.items():
        ids = tokenizer_original.convert_tokens_to_ids(tokenizer_original.tokenize(action_description))
        random_action_embed = model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])]
        new_action_embed = torch.mean(model.bert.embeddings.cpu()(torch.tensor([ids])),dim=1)
        #new_action_embed = new_action_embed.to(random_action_embed.device)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(random_action_embed.cpu(), new_action_embed)
        with torch.no_grad():
            model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])] = new_action_embed
            model.decoder.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])] = new_action_embed
        random_action_embed = model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])]
        output = cos(random_action_embed, new_action_embed)
        output = cos(model.decoder.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])], new_action_embed)

    model.to(device)
    model.bert.to(device)
    model.bert.embeddings.to(device)
    model.bert.embeddings.word_embeddings.to(device)
    
    if args.freeze_encoder != None and args.freeze_encoder == True:
        for param in model.bbx.parameters():
            param.requires_grad = False
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.cross.parameters():
            param.requires_grad = False
    elif args.freeze_encoder != None:
        # remove freeze_encoder flag
        args.freeze_encoder = None
        

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if 'bbx' in name:
    #             print(name)
    #         # print(name)
    #     else:
    #         # if 'bbx' in name:
    #         #     print('no grad ' +name)
    #         # print('no-grad ' +name)
    #         continue
    assert "caption" in args.task_type
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

    assert args.datatype in DATALOADER_DICT
    args.video_feature = pickle.load(open(args.features_path, 'rb'))
    args.video_bbx_feature = pickle.load(open(args.bbx_features_path, 'rb'))
    if args.datatype == "ourds-audio-bbx":
        args.audio_feature = pickle.load(open(args.audio_features_path, 'rb'))

    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='val')
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test')

    if args.local_rank == 0:
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = None
        best_output_model_file = None
        global_step = 0
        debug_eval = False
        conf ={}
        for key,value in args.__dict__.items():
            if(key in ["video_feature", "video_bbx_feature", "cos", "device", "audio_feature"]):
                continue
            conf[key] = value
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="Multimodal-Fusion-Bottleneck",
            name="{}_{}_{}_{}_{}_enc_{}_cross_{}_declay_{}_conv_{}".format(args.task_type, args.datatype, args.bert_model, args.lr, args.batch_size,  args.visual_num_hidden_layers, args.cross_num_hidden_layers, args.decoder_num_hidden_layers ,args.bottleneck_use_conv),
            # track hyperparameters and run metadata
            config=conf,
        ) 
        # writer = SummaryWriter(
        #     "runs/{}_{}_{}_{}_{}_{}_bottley_{}_bottdim_{}_enc_{}_cross_{}_declay_{}_conv_{}".format(datetime.now(),args.task_type, args.datatype, args.bert_model, args.lr, args.batch_size, args.bottleneck_fusion_layers, args.bottleneck_dim, args.visual_num_hidden_layers, args.cross_num_hidden_layers,args.decoder_num_hidden_layers, args.bottleneck_use_conv)
        # )
        writer = None

        if debug_eval is True:
            Scores = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
            _ = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
            raise

        for epoch in range(args.epochs):
            # train_sampler.set_epoch(epoch)

            if debug_eval is False:
                logger.info("Epoch: %d/%s", epoch + 1, args.epochs)
                tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank, writer=writer)
            else:
                tr_loss = 0
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, type_name="")
                if epoch > 1:
                    logger.info("***** Running validation *****")
                    Scores = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    if writer is not None:
                        writer.add_scalar("Bleu_4/test", Scores["Bleu_4"], epoch)
                        for key, value in Scores.items():
                            writer.add_scalar("%s/test"%key, value, epoch)
                    if wandb is not None:
                        scores = {key : value for key, value in Scores.items()}
                        wandb.log(scores)
                    # Scores = Scores["Bleu_4"]
                    average_improvement = 0
                    if best_score is not None:
                        for key in Scores.keys():
                            average_improvement += 1 if  (Scores[key] - best_score[key]) > 0 else 0
                        
                    if (best_score == None or average_improvement > 0) and epoch > 1:
                        best_score = Scores
                        best_output_model_file = output_model_file
                        logger.info('This is the best model in val set so far, testing test set....')
                        
                        Scores = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                        if writer is not None:
                            writer.add_scalar("Bleu_4/test", Scores["Bleu_4"], epoch)
                            for key, value in Scores.items():
                                writer.add_scalar("%s/test"%key, value, epoch)
                        if wandb is not None:
                            scores = {key : value for key, value in Scores.items()}
                            wandb.log(scores)
                    logger.info("The best model is: {}, the Bleu_4 is: {:.4f}".format(best_output_model_file, best_score["Bleu_4"] if best_score is not None else 0.0))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test')
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
        wandb.finish()
    elif args.do_eval:
        if args.local_rank == 0:

            test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test')
            model = load_model(-1, args, n_gpu, device, model_file=args.init_model)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

class Args_Caption:
    def __init__(self, data_dir="data", features_dir="features", do_eval=True, task="caption", output_dir="output", export_attention_scores=False):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.do_pretrain = False
        self.use_prefix_tuning = False
        self.do_train = not do_eval
        self.do_eval = do_eval
        self.train_csv = "{}/ourds_train.44k.csv".format(self.data_dir)
        self.val_csv = "{}/ourds_JSFUSION_test.csv".format(self.data_dir)
        self.data_path = "{}/new_ourds_description_only.json".format(self.data_dir)
        # self.features_path = "{}/ourds_features_w_size_32.pkl".format(self.features_dir)
        # self.features_path = "{}/ourds_features_w_size_8.pkl".format(self.features_dir)
        self.bbx_features_path = "{}/cls2_ball_basket_sum_concat_original_courtline_fea_1.pickle".format(self.data_dir)
        self.features_path = "{}/ourds_videos_timesformer_features.pickle".format(self.features_dir)
        self.audio_features_path = "{}/ourds_audio_VGGish_features.pkl".format(self.data_dir)
        self.num_thread_reader = 0
        self.lr = 3e-5
        self.epochs = 10
        self.batch_size = 32
        self.batch_size_val = 16
        self.lr_decay = 0.9
        self.n_display = 100
        self.video_dim = 768
        self.audio_dim = 128
        self.seed = 42
        self.max_words = 30
        self.max_frames = 48
        self.feature_framerate = 1
        self.min_time = 5.0
        self.margin = 0.1
        self.hard_negative_rate = 0.5
        self.negative_weighting = 1
        self.n_pair = 1
        self.output_dir = '{}/{}'.format(os.environ["DIR_PATH"], output_dir)
        self.bert_model = "bert-base-uncased"
        self.visual_model = "visual-base"
        self.cross_model = "cross-base"
        self.decoder_model = "decoder-base"
        self.init_model = "{}/weight/univl.pretrained.bin".format(".")
        self.do_lower_case = True
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1
        self.n_gpu = 1
        self.cache_dir = ""
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.task_type = task 
        self.datatype = "ourds-CLIP" 
        self.world_size = 0
        self.local_rank = 0
        self.coef_lr = 0.1
        self.use_mil = False
        self.context_only = False
        self.multibbxs = True
        self.sampled_use_mil = False
        self.text_num_hidden_layers = 12
        self.visual_num_hidden_layers = 6
        self.cross_num_hidden_layers = 3
        self.decoder_num_hidden_layers = 3
        self.loss = "MSE"
        self.bottleneck_dim = 0,
        self.bottleneck_fusion_layers = 0,
        self.bottleneck_use_conv = False
        self.export_attn_scores = export_attention_scores
        self.visual_use_diagonal_masking = False
        self.train_tasks = [0,0,1,0]
        self.cross_masking = None # "upper", "lower", "upper-no-inp", "lower-no-inp", None, "random", "random-global"
        self.test_tasks = [0,0,1,0]
        self.t1_postprocessing = True
        self.stage_two = True
        self.unsup_pretrain = False
        self.bert_weights_only = False
        self.use_answer = None
        self.join_vision_audio = True
        self.fine_tune_extractor = False
        self.extractor = "videomae"
        self.player_embedding = "CLIP" # BERT, CLIP, none, BERT-Stat
        self.use_random_embeddings = False
        self.player_embedding_order = "lineup" # lineup, lineup-ordered, posession, none, BC
        self.use_BBX_features = True
        self.max_rand_players = 5
if __name__ == "__main__":
    args = None
    
    args = Args_Caption(features_dir="data", do_eval=False, output_dir="Finetuned_models/tmp3", export_attention_scores=False, task="caption-CLIP")
    args.freeze_encoder = False
    init_training_caption(args)

