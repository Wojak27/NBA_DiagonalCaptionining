from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from distutils.log import debug
from posixpath import split
from sys import breakpointhook
import wandb

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
from dataloaders.dataloader_ourds_CLIP import OURDS_CLIP_DataLoader
from eval_utils import acc_iou, mean_category_acc, success_rate
from main_task_caption_DAM import Args_Caption
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import UniVL
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
from dataloaders.dataloader_ourds_caption_multifeat import OURDS_Caption_DataLoader
from util import *
from torch import nn
from torchsummary import summary
import pickle5 as pickle
import re

global logger

wandb.login()

def get_args(description='UniVL on Caption Task'):
    parser = argparse.ArgumentParser(description="Arguments for Video Captioning Task")

    parser.add_argument("--data_dir", default="data", help="Directory for the data")
    parser.add_argument("--features_dir", default="features", help="Directory for the features")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation")
    parser.add_argument("--task", default="caption", choices=["caption", "other_tasks"], help="Task to perform")
    parser.add_argument("--output_dir", default="output", help="Directory for the output")
    parser.add_argument("--export_attention_scores", action='store_true', help="Whether to export attention scores")
    
    parser.add_argument("--num_thread_reader", type=int, default=0, help="Number of threads for data reading")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate decay")
    parser.add_argument("--n_display", type=int, default=100, help="Number of iterations to display training progress")
    
    parser.add_argument("--video_dim", type=int, default=768, help="Dimensionality for video features")
    parser.add_argument("--audio_dim", type=int, default=128, help="Dimensionality for audio features")
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomness")
    parser.add_argument("--max_words", type=int, default=30, help="Maximum number of words in captions")
    parser.add_argument("--max_frames", type=int, default=48, help="Maximum number of frames in videos")
    parser.add_argument("--feature_framerate", type=int, default=1, help="Frame rate for feature extraction")
    
    parser.add_argument("--bert_model", default="bert-base-uncased", help="Pre-trained BERT model")
    parser.add_argument("--visual_model", default="visual-base", help="Model for visual features")
    parser.add_argument("--cross_model", default="cross-base", help="Model for cross-modal features")
    parser.add_argument("--decoder_model", default="decoder-base", help="Model for decoding")
    parser.add_argument("--init_model", default="./weight/univl.pretrained.bin", help="Initial model weights")

    parser.add_argument("--do_lower_case", action='store_true', help="Whether to lower case the input text")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Proportion of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--fp16", action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    
    parser.add_argument("--use_mil", action='store_true', help="Whether to use Multiple Instance Learning")
    parser.add_argument("--context_only", action='store_true', help="Use context only for training")
    parser.add_argument("--multibbxs", action='store_true', help="Whether to use multiple bounding boxes")
    parser.add_argument("--visual_use_diagonal_masking", action='store_true', help="Use diagonal masking for visual features")
    parser.add_argument("--player_embedding", default="CLIP", choices=["BERT", "CLIP", "none", "BERT-Stat"], help="Type of player embedding to use")
    parser.add_argument("--player_embedding_order", default="lineup", choices=["lineup", "lineup-ordered", "posession", "none", "BC"], help="Order of player embedding")
    parser.add_argument("--use_BBX_features", action='store_true', help="Use bounding box features")
    parser.add_argument("--max_rand_players", type=int, default=5, help="Maximum number of random players")

    
    parser.add_argument('--action_level', default=1, help="Whether decide which action do we want to perform recognition, range from 0-2")

    args = parser.parse_args()
    args.do_train = not args.do_eval
    args.output_dir = os.path.join(os.environ["DIR_PATH"], args.output_dir)

    args = parser.parse_args()

    args.do_train = True
    args.stage_two = True
    args.do_lower_case = True
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

    torch.cuda.set_device(args.local_rank)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

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

def init_model(args, device, n_gpu, local_rank, type_vocab_size=2):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args, type_vocab_size=type_vocab_size)
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

def dataloader_ourds_CLIP_train(args, tokenizer, action_converter=None):
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
        use_random_embeddings=args.player_embedding == "Rand",
        num_samples=100000,
        mask_prob=0.25,
        only_players=True,
        use_real_name=False,
        player_embedding_order=args.player_embedding_order,
        use_BBX_features=args.use_BBX_features,
        player_embedding=args.player_embedding,
        max_rand_players=args.max_rand_players,
        action_convert_dict=action_converter
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

def dataloader_ourds_CLIP_test(args, tokenizer, split_type="test", action_converter=None):
    ourds_testset = OURDS_CLIP_DataLoader(
        csv_path=args.val_csv,
        json_path="/home/karolwojtulewicz/code/NSVA/data/new_ourds_description_only.json",
        video_feature=args.video_feature,
        bbx_feature=args.video_bbx_feature,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        use_random_embeddings=args.player_embedding == "Rand",
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
        max_rand_players=args.max_rand_players,
        action_convert_dict=action_converter
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

    #model_file = '/home/ubuntu/vcap/content/ckpts/ckpt_ourds_caption/pytorch_model.bin.6'
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
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        # print("One batch data takes {} to prepare".format(time2-time1))

        input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, masked_bbx, bbx_labels_index = batch
            
        ime1 = time.time()
        loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                        pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                        masked_video=masked_video, video_labels_index=video_labels_index,
                        input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                        output_caption_ids=pairs_output_caption_ids,task_type=task_type, bbx=bbx.float(), bbx_mask=bbx_mask.float(), masked_bbx=masked_bbx.float(), bbx_labels_index=bbx_labels_index)
        if wandb is not None:
            wandb.log({"LossAction/train": loss})
        time2 = time.time()
        # print("One batch captioning result takes {} to run".format(time2-time1))

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
    sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, feature_tuple, feature_mask_tuple, task_type_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    feature_tuple = collect_active_part(feature_tuple, active_inst_idx, n_prev_active_inst, n_bm)
    feature_mask_tuple = collect_active_part(feature_mask_tuple, active_inst_idx, n_prev_active_inst, n_bm)


    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)


    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
    active_task_type_rpt = collect_active_part(task_type_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    return (active_sequence_output_rpt, active_visual_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt, feature_tuple, feature_mask_tuple, active_task_type_rpt), \
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
        sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, feature_tuple, feature_mask_tuple = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        # assert court_output_rpt.shape == bbz_output_rpt.shape, "Wrong shape court {} vs. bbx {}".format(court_output_rpt.shape, bbz_output_rpt.shape)
        # assert court_output_rpt.shape == visual_output_rpt.shape, "Wrong shape court {} vs. feat {}".format(court_output_rpt.shape, visual_output_rpt.shape)
        # assert bbz_output_rpt.shape == visual_output_rpt.shape, "Wrong shape bbx {} vs. feat {}".format(bbz_output_rpt.shape, visual_output_rpt.shape)

        dec_output = decoder(sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt,
                             video_mask_rpt, next_decoder_ids, next_decoder_mask, feature_tuple, feature_mask_tuple, shaped=True, get_logits=True,task_type = task_type)
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
    all_rst_lists = []
    all_gt_lists = []
    all_caption_lists = []
    model.eval()

    for b_id, batch in enumerate(test_dataloader):
        # if b_id > 1:
        #     continue

        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, _, _ = batch

        # "Map feature tuple to cuda()"
        # for key, value in feature_tuple.items():
        #     feature_tuple[key] = value.to(device=device, non_blocking=True)
        # for key, value in feature_mask_tuple.items():
        #     feature_mask_tuple[key] = value.to(device=device, non_blocking=True)

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

            decoder = model.decoder_caption

            # Note: shaped first, then decoder need the parameter shaped=True
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            input_mask = input_mask.view(-1, input_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            
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

            # new_feat_output_tuple = {}
            # for key, value in fea_output_tuple.items():
            #     value = value.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            #     new_feat_output_tuple[key] = value

            input_ids_rpt = input_ids.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            input_mask_rpt = input_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            # bbx_mask_rpt = bbx_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            # court_mask_rpt = court_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            
            if args.fine_tune_extractor == False:
                bbx_output_rpt = bbx_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)


            task_type_rpt = task_type.repeat(n_bm)
            if args.fine_tune_extractor == False:
                bbx_mask_rpt = bbx_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)


            # {0:0, 1:1, 2:2, 3:3, 4:4}
            # print(inst_idx_to_position_map)
            # print(list(range(n_inst)))
            # breakpoint()

            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1):
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, bbx_output_rpt, bbx_mask_rpt), task_type = task_type_rpt)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, bbx_output_rpt, bbx_mask_rpt, task_type_rpt), \
                inst_idx_to_position_map = collate_active_info((sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, bbx_output_rpt, bbx_mask_rpt, task_type_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device
                                                               )

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            # pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, 30) # hard-code with 30 as max-length
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            "Save Intermediate Results, Do Squeeze on irregular shapes"
            new_batch_hpy = []
            for item in batch_hyp:
                new_item = item[0]
                new_item.extend([0] * (30-len(new_item)))
                new_batch_hpy.append(new_item)
            for l in caption_list:
                all_gt_lists.append([l])
            
            for l in new_batch_hpy:
                all_rst_lists.append([l])

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
                orig_decode_text = decode_text
                try:
                    if args.t1_postprocessing:
                        match = re.search(r'action[0-9]+[ ]*', decode_text)
                        if match !=None:

                            match_action_type = match.group(0).replace(' ','')
                            replace_type = action_token2full_description[match_action_type].replace(' unknown','').replace('made shot ','').replace('missed shot','miss')
                            decode_text  = decode_text.replace(match_action_type, replace_type)
                    result_list_byTask[task_type.tolist()[re_idx]].append(decode_text)
                    all_result_lists.append("Decoded:"+decode_text + "Groundtruth:" + orig_decode_text)
                except:
                    print("Error in decoding action token to text")

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
    
    """ Process the whole results """
    final_pred = np.stack(all_rst_lists, 0).reshape(-1, 30)
    final_gt = np.concatenate(all_gt_lists, 0)

    sr_list = []
    acc_list = []
    mIoU = []
    mInter = []

    """ Calculate the metrics all together """
    for pred, gt in zip(final_pred, final_gt):
        if len(pred.shape) > 1: # make sure the shape is good;
            pred = np.squeeze(pred)

        nonzero_mask = np.where(gt > 0, 1, 0)
        assert np.sum(nonzero_mask) != 0, "All groundtruth is zeors!!!!!!!!!!!"
        nonzero_gt = gt[np.nonzero(gt)][:-1] # get rid of padding and 102 (End-of_Seq) token
        # nonzero_pred  = pred[np.nonzero(pred * nonzero_mask)][:-1] # get rid of padding and 102 (End-of_Seq) token
        nonzero_pred = pred[:len(nonzero_gt)]

        sr_list.append(success_rate(np.expand_dims(nonzero_pred, 0), np.expand_dims(nonzero_gt, 0)))
        acc_list.append(mean_category_acc(nonzero_pred.tolist(), nonzero_gt.tolist()))
        mIoU.append(acc_iou(np.expand_dims(nonzero_pred, 0), np.expand_dims(nonzero_gt, 0)))
        score = []
        for item in nonzero_pred:
            if item in nonzero_gt:
                score.append(1.0)
            else:
                score.append(0.0)
        mInter.append(sum(score) / len(score))

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

    # Save translated results
    hyp_path = os.path.join(args.output_dir, "hyp_trans.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    if wandb is not None:
        scores = {
            "Action_SR": sum(sr_list)/ len(sr_list),
            "Action_ACC": sum(acc_list)/ len(acc_list),
            "Action_mIoU": sum(mIoU)/ len(mIoU),
            "Action_mInter": sum(mInter) / len(mInter),
        }
        wandb.log(scores)

    return sum(sr_list)/ len(sr_list), sum(acc_list)/ len(acc_list), sum(mIoU)/ len(mIoU), sum(mInter) / len(mInter)

DATALOADER_DICT = {}
DATALOADER_DICT["ourds-DAM"] = {"train":dataloader_ourds_CLIP_train, "val":dataloader_ourds_CLIP_test}

"Set the action-level to have results for different level"
# action_list = json.load(open('/media/chris/hdd1/UniVL_processing_code/UniVL-main/action_list.json', 'r'))
action_list = json.load(open('./data/action_list.json', 'r'))
# action_list_event = set(json.load(open('/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/UniVL-main/action_list_Event.json', 'r')).values())
action_list_fine_dict = json.load(open('./data/action_list_Fine.json', 'r'))
action_list_coarse_dict = json.load(open('./data/action_list_Coarse.json', 'r'))

action_list_fine_set = set(json.load(open('./data/action_list_Fine.json', 'r')).values())
action_list_coarse_set = set(json.load(open('./data/action_list_Coarse.json', 'r')).values())

action_token2full_description = {'action%s'%a_idx:a_l.lower().replace('_',' ').replace('-',' ') for a_idx, a_l in enumerate(action_list)}
action_token2full_description_fine = {'action%s'%a_idx:a_l.lower().replace('_',' ').replace('-',' ') for a_idx, a_l in enumerate(action_list_fine_set)}
action_token2full_description_coarse = {'action%s'%a_idx:a_l.lower().replace('_',' ').replace('-',' ') for a_idx, a_l in enumerate(action_list_coarse_set)}

"""Action dictionaries"""
action_token2full_description_dict = {a_l : 'action%s'%a_idx for a_idx, a_l in enumerate(action_list)}
action_token2full_description_fine_dict = {a_l: 'action%s'%a_idx for a_idx, a_l in enumerate(action_list_fine_set)}
action_token2full_description_coarse_dict = {a_l:'action%s'%a_idx for a_idx, a_l in enumerate(action_list_coarse_set)}


"""Action converter"""
action_token2full_converter_fine= {}
for key, value in action_token2full_description_dict.items():
    tmp_value = action_list_fine_dict[key]
    final_value = action_token2full_description_fine_dict[tmp_value]
    action_token2full_converter_fine[value] = final_value

action_token2full_converter_coarse= {}
for key, value in action_token2full_description_dict.items():
    tmp_value = action_list_coarse_dict[key]
    final_value = action_token2full_description_coarse_dict[tmp_value]
    action_token2full_converter_coarse[value] = final_value

action_converter_level ={
    0: None,
    1: action_token2full_converter_fine,
    2: action_token2full_converter_coarse
}

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def main(args):
    global logger
    if args == None:
        args = get_args()
    if isinstance(args, dict):
        args = DictToObject(args)
    args = set_seed_logger(args)
    assert args.action_level in [0, 1, 2]
    print("Running action recognition on level {}".format(args.action_level))

    device, n_gpu = init_device(args, args.local_rank)
    n_gpu = 1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## Collect declared feature, to initialize model + dataloader
    feature_tuple = {}
    # for arg in vars(args):
    #     if '_features_path' in arg:
    #         feature_tuple[arg.split('_')[0]] = getattr(args, arg)
    # breakpoint()
    conf ={}
    for key,value in args.__dict__.items():
        if(key in ["video_feature", "video_bbx_feature", "cos", "device"]):
            continue
        conf[key] = value
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Multimodal-Fusion-Bottleneck",
        name="{}_{}_{}_{}_enc_{}_cross_{}_declay_{}".format("action{}".format(args.action_level), args.datatype, args.lr, args.batch_size, args.visual_num_hidden_layers, args.cross_num_hidden_layers, args.decoder_num_hidden_layers),
        # track hyperparameters and run metadata
        config=conf,
    ) 
    """
    Mannually use the default fine-grained features, which are
    1. ball_basket_cl2_sum
    2. courtline segmentation
    """
    feature_tuple['courtseg'] = './data/cls2_ball_basket_sum_concat_original_courtline_fea_1.pickle'
    # feature_tuple['bbxcls2'] = '/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/ourds_courtlineseg_data/cls2_ball_basket_sum_concat_original_courtline_fea.pickle',

    logger.info("***** Using the following Features %s *****", feature_tuple)
    num_token = len(feature_tuple) + 1 # Timesformer feature + others

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer_original = BertTokenizer.from_pretrained(args.bert_model+'-original', do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank, type_vocab_size=num_token)
    
    for action_token, action_description in action_token2full_description.items():
        ids = tokenizer_original.convert_tokens_to_ids(tokenizer_original.tokenize(action_description))
        action_token = tokenizer_original.tokenize(action_token)
        new_action_embed = torch.mean(model.bert.embeddings.cpu()(torch.tensor([ids])),dim=1)
        with torch.no_grad():
            model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids(action_token)] = new_action_embed
            model.decoder.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids(action_token)] = new_action_embed

    model.to(device)
    model.bert.to(device)
    model.bert.embeddings.to(device)
    model.bert.embeddings.word_embeddings.to(device)

    assert args.task_type == "caption"
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

    assert args.datatype in DATALOADER_DICT
    args.video_feature = pickle.load(open(args.features_path, 'rb'))
    args.video_bbx_feature = pickle.load(open(args.bbx_features_path, 'rb'))
    args.feature_tuple = {}

    common_keys = []
    for name, path in feature_tuple.items():
        args.feature_tuple[name] = []
        logger.info(" Loading feature pickle from %s", path)

        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        for key, value in data.items():
            data.update({key: data[key].transpose(1, 2, 0).reshape(-1, 768*2)})
        
        args.feature_tuple[name].append(data)

        assert name in ['bbx', 'courtseg', 'keypoint', 'allplayer', 'bbxcls2', 'bbxball', 'bbxbasket']

        if name == 'bbx' or name == 'bbxcls2' or name == 'bbxball' or name == 'bbxbasket':
            args.feature_tuple[name][0]['video10084'] = np.zeros((30, 768)) # Add this to avoid key-error
            args.feature_tuple[name].append((30, 768)) # [numFrame, dimFeature]
        elif name == 'courtseg':
            args.feature_tuple[name].append((30, 768 * 2))
        elif name == 'keypoint':
            args.feature_tuple[name][0]['video22693'] = np.zeros((30, 768)) # Add this to avoid key-error
            args.feature_tuple[name][0]['video5273'] = np.zeros((30, 768)) # Add this to avoid key-error
            args.feature_tuple[name].append((30, 768))
        else:
            args.feature_tuple[name].append((30, 10, 768)) # For allplayer, [numFrame, numPlayer, dimFeature]

    "Check the common keys in all feature_tuple"

    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, split_type='val', action_converter=action_converter_level[args.action_level])
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, split_type='test', action_converter=action_converter_level[args.action_level])

    if args.local_rank == 0:
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer,action_converter=action_converter_level[args.action_level])
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

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        debug_eval = False

        if debug_eval is True:
            sr, acc, mIoU, mInter = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
            # sr, acc, mIoU = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

        for epoch in range(args.epochs):
            # train_sampler.set_epoch(epoch)

            if debug_eval is False:
                tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)
            else:
                tr_loss = 0
            
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, type_name="")
                if epoch > 2:
                    sr, acc, mIoU, mInter = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    if best_score <= acc and epoch > 2:
                        best_score = acc
                        best_output_model_file = output_model_file
                        logger.info('This is the best model in val set so far, testing test set....')
                        eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    logger.info("The best model is: {}, the Best-Acc {}, SR {}, mIoU {}, mInter {}".format(best_output_model_file, best_score, sr, mIoU, mInter))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            # test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test',action_converter=action_converter_level[args.action_level])
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
    elif args.do_eval:
        if args.local_rank == 0:

            test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test')
            model = load_model(-1, args, n_gpu, device, model_file=args.init_model)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

if __name__ == "__main__":
    args = get_args()
    
    main(args)




