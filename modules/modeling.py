# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
import json
import modules.VideoMAE.utils as utils

import logging
import random
import numpy as np
from pyrsistent import b
from timm.models import create_model
import modules.VideoMAE.modeling_finetune

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from modules.Timesformer.timesformer.models.vit import TimeSformer
from modules.module_bottleneck import MTB_Bottleneck_Gaze, MTB_Bottleneck_Git, MTB_Bottleneck_Lin, MTB_Bottleneck_Orig, Scaled_Fusion
from modules.module_bottleneck_plug import BottleNeckMTB

from modules.until_module import MaskedL1Loss, MaskedMSELoss, PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import BBXModel, VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross import CrossAttention, CrossAttention_KVQ, CrossModel, CrossConfig
from modules.module_decoder import DecoderModel, DecoderConfig
from torch.nn.functional import normalize

logger = logging.getLogger(__name__)


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.decoder = None

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, cross_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
    
        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)

        assert model.bert is not None
        assert model.visual is not None

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video

class PlayerClassifier(nn.Module):
    def __init__(self, num_classes, embedding_size=768):
        super(PlayerClassifier, self).__init__()
        self.classifier = nn.Linear(embedding_size, num_classes)
    
    def forward(self, embeddings):
        logits = self.classifier(embeddings)
        return logits

class NormalizeBBX(nn.Module):
    def __init__(self, task_config):
        super(NormalizeBBX, self).__init__()
        self.visual_norm2d = LayerNorm(1536)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video
    
def reshape_bbx(bbx):
    bbx = torch.as_tensor(bbx).float()
    bbx = bbx.view(bbx.shape[0], bbx.shape[-2], -1)
    return bbx
        

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class GenericEmbedding(nn.Module):
    def __init__(self, dim_from, dim_to,config = None):
        super(GenericEmbedding, self).__init__()
        self.config = config
        self.embedding = nn.Linear(dim_from, dim_to)
        self.norm = LayerNorm(dim_to)

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        return x
    
class AudioEmbedding(nn.Module):
    def __init__(self, config, audio_dim=128):
        super(AudioEmbedding, self).__init__()
        self.config = config
        self.audio_embedding = nn.Linear(audio_dim, config.hidden_size)
        self.audio_norm = LayerNorm(config.hidden_size)

    def forward(self, audio):
        audio = self.audio_embedding(audio)
        audio = self.audio_norm(audio)
        return audio
class AudioEmbeddingReverse(nn.Module):
    def __init__(self, config, audio_dim=128):
        super(AudioEmbeddingReverse, self).__init__()
        self.config = config
        self.audio_embedding = nn.Linear(config.hidden_size, audio_dim)
        self.audio_norm = LayerNorm(audio_dim)

    def forward(self, audio):
        audio = self.audio_embedding(audio)
        audio = self.audio_norm(audio)
        return audio

class LossBalancerKendall(nn.Module):
    def __init__(self):
        super(LossBalancerKendall, self).__init__()
        self.alpha = nn.Parameter(data=torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(data=torch.randn(1), requires_grad=True)
        
    def forward(self, loss1, loss2):
        loss = 1/(2*self.alpha**2)*loss1 + 1/(2*self.beta**2)*loss2 + torch.log(self.alpha*self.beta)
        return loss
    
def custom_contrastive_loss(zq, zk):
    """
    Custom contrastive loss for full vector sequences.
    
    :param zq: Tensor of query sequences, shape (batch_size, seq_len, embed_dim).
    :param zk: Tensor of knowledge sequences, shape (batch_size, seq_len, embed_dim).
    :return: Scalar tensor representing the loss.
    """
    # Average the embeddings across the sequence length
    zq_avg = torch.mean(zq, dim=1)  # Shape: (batch_size, embed_dim)
    zk_avg = torch.mean(zk, dim=1)  # Shape: (batch_size, embed_dim)
    
    # Calculate the dot product between averaged zq and zk (positive pair)
    pos_scores = torch.sum(zq_avg.unsqueeze(1) * zk_avg, dim=-1)  # Shape: (batch_size, batch_size)
    
    # Diagonal elements are zq . zk (positive samples)
    pos_scores_diag = torch.diag(pos_scores)
    
    # Calculate the exponentiated scores for normalization
    exp_scores = torch.exp(pos_scores)  # Shape: (batch_size, batch_size)
    
    # Sum of exponentiated scores for all pairs, excluding the positive pair for each query
    sum_exp_scores = torch.sum(exp_scores, dim=1)  # Sum across all pairs
    
    # Calculate the log probabilities
    log_probs = pos_scores_diag - torch.log(sum_exp_scores)
    
    # Average loss across the batch
    loss = -torch.mean(log_probs)
    
    return loss

class BBXEmbeddingReverse(nn.Module):
    def __init__(self, config, bbx_dim=128):
        super(BBXEmbeddingReverse, self).__init__()
        self.config = config
        self.bbx_embedding = nn.Linear(config.hidden_size, bbx_dim)
        # self.bbx_norm = LayerNorm(bbx_dim)

    def forward(self, bbx):
        bbx = self.bbx_embedding(bbx)
        # bbx = self.bbx_norm(bbx)
        return bbx
class UniVL(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config):
        super(UniVL, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.unsup_pretrain = "unsup-pretrain" in task_config.task_type
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        
        if self.task_config.fine_tune_extractor == True:
            if self.task_config.extractor == "timesformer":
                self.visual = TimeSformer(
                    img_size=224,
                    num_classes=400,
                    num_frames=8,
                    attention_type="divided_space_time",
                    pretrained_model="/4TBSSD_permanent/NSVA/tools/Timesformer/models/TimeSformer_divST_32x32_224_HowTo100M.pyth",
                ).to("cuda")
            else:
                finetune = "/home/karolwojtulewicz/code/NSVA/modules/VideoMAE/models/VideoMAE_Base_16_2_3.pth"
                # VideoMAE
                self.visual = create_model(
                    'vit_base_patch16_224',
                    pretrained=False,
                    num_classes=400,
                    all_frames=30,
                    tubelet_size=2,
                    fc_drop_rate=0,
                    drop_rate=0,
                    drop_path_rate=0.1,
                    attn_drop_rate=0,
                    drop_block_rate=None,
                    use_checkpoint=True,
                    use_mean_pooling=False,
                    init_scale=0.001
                )
                checkpoint = torch.load(finetune, map_location='cpu')

                print("Load ckpt from %s" % finetune)
                checkpoint_model = None
                if checkpoint_model is None:
                    checkpoint_model = checkpoint
                state_dict = self.visual.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                checkpoint_model = checkpoint_model['model']
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('backbone.'):
                        new_dict[key[9:]] = checkpoint_model[key]
                    elif key.startswith('encoder.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        new_dict[key] = checkpoint_model[key]
                checkpoint_model = new_dict

                # interpolate position embedding
                if 'pos_embed' in checkpoint_model:
                    pos_embed_checkpoint = checkpoint_model['pos_embed']
                    embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
                    num_patches = self.visual.patch_embed.num_patches # 
                    num_extra_tokens = self.visual.pos_embed.shape[-2] - num_patches # 0/1

                    # height (== width) for the checkpoint position embedding 
                    orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(self.task_config.max_frames // self.visual.patch_embed.tubelet_size)) ** 0.5)
                    # height (== width) for the new position embedding
                    new_size = int((num_patches // (self.task_config.max_frames // self.visual.patch_embed.tubelet_size) )** 0.5)
                    # class_token and dist_token are kept unchanged
                    if orig_size != new_size:
                        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                        # B, L, C -> BT, H, W, C -> BT, C, H, W
                        pos_tokens = pos_tokens.reshape(-1, self.task_config.max_frames // self.visual.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                        pos_tokens = torch.nn.functional.interpolate(
                            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, self.task_config.max_frames // self.visual.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                        pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                        checkpoint_model['pos_embed'] = new_pos_embed

                utils.load_state_dict(self.visual, checkpoint_model, prefix="")

        self._stage_one = True
        self._stage_two = False

        self.multibbxs = task_config.multibbxs
        self.context_only = self.task_config.context_only
        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.train_sim_after_cross = False
        # self.train_sim_after_cross = True
        
        if "-knowledge" in self.task_config.task_type:
            players_dict = json.load(open("/home/karolwojtulewicz/code/NSVA/data/unique_players.json","r"))
            num_classes = len(players_dict)
            
            self.player_loss = nn.BCEWithLogitsLoss()
            self.player_classifier = PlayerClassifier(num_classes=num_classes)
            self.cross_attn_layers = nn.ModuleList()
            for i in range(3):
                self.cross_attn_layers.append(CrossAttention(cross_config))

        if self._stage_one and check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        ########### Initialize Models ###########
        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        if self.task_config.fine_tune_extractor == False:
            try:
                if task_config.video_dim:
                    self.visual = VisualModel(visual_config, dim=task_config.video_dim)
                else:
                    self.visual = VisualModel(visual_config)
            except:
                self.visual = VisualModel(visual_config)
            visual_word_embeddings_weight = self.visual.embeddings.to_visual_embeddings.weight
            # <=== End of Video Encoder

            # BBX Encoder ===>
            bbx_config = update_attr("bbx_config", visual_config, "num_hidden_layers",
                                        self.task_config, "visual_num_hidden_layers")
            if "audio" in self.task_config.task_type:
                self.audio_embed = AudioEmbedding(bbx_config, audio_dim=task_config.audio_dim)
            
            if "audio" in self.task_config.task_type:
                if "caption" in self.task_config.task_type:
                    self.bbx = BBXModel(bbx_config, dim=768)
                else:
                    self.audio_reverse = AudioEmbeddingReverse(bbx_config)
                    self.bbx = BBXModel(bbx_config, dim=768)
            else:
                self.bbx = BBXModel(bbx_config)

            self.bbxs_fusion_one = torch.nn.Parameter(torch.ones(3, 1,dtype = torch.float32))
            # nn.init.ones_(self.bbxs_fusion_one)
            self.bbxs_fusion_one.requires_grad = True


            self.bbxs_fusion_two = nn.Linear(768*3, 768)
            # nn.init.ones_(self.bbxs_fusion_one)
            self.bbxs_fusion_two.requires_grad = True
            self.bbxs_dropout = nn.Dropout(0.5)
            self.bottleneck = None
            ############## End of Video Encoder ##############

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = None
            if(cross_config.num_hidden_layers != 0):
                self.cross = CrossModel(cross_config, export_attn_scores=(not self.task_config.do_train and self.task_config.export_attn_scores), args=self.task_config)
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                           self.task_config, "decoder_num_hidden_layers")
                if self.unsup_pretrain:
                    if "audio" in self.task_config.task_type:
                        self.decoder_audio = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight, output_concat=True, args=self.task_config)
                        self.decoder_video = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight, output_concat=True, args=self.task_config)
                    else:
                        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight, output_concat=True, args=self.task_config)
                else: 
                    if self.task_config.task_type == "caption-audio-bbx":
                        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
                        self.decoder_bbx = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight, output_concat=True, args=self.task_config)
                    else:
                        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)
                # <=== End of Decoder

            if self.task_config.do_pretrain:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.normalize_video = NormalizeVideo(task_config)
        self.normalize_bbx = NormalizeBBX(task_config)

        mILNCELoss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        maxMarginRankingLoss = MaxMarginRankingLoss(margin=task_config.margin,
                                                    negative_weighting=task_config.negative_weighting,
                                                    batch_size=task_config.batch_size // task_config.n_gpu,
                                                    n_pair=task_config.n_pair,
                                                    hard_negative_rate=task_config.hard_negative_rate, )
        if self.unsup_pretrain:
            if task_config.loss == "MSE":
                self.decoder_loss_fct = MaskedMSELoss()
            else: #"L1"
                self.decoder_loss_fct = MaskedL1Loss()
        if "audio-bbx" in task_config.task_type:
            self.bbx_embed_reverse = BBXEmbeddingReverse(bbx_config, bbx_dim=1536)
            self.loss_balancer = LossBalancerKendall()
            if task_config.loss == "MSE":
                self.bbx_loss = MaskedMSELoss()
            else: #"L1"
                self.bbx_loss = MaskedL1Loss()
                
        if self.task_config.do_pretrain:
            self.bbx_norm = GenericEmbedding(1536,768)
            
        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mILNCELoss
            self._pretrain_sim_loss_fct = mILNCELoss
        else:
            self.loss_fct = CrossEn() if self._stage_two else maxMarginRankingLoss
            self._pretrain_sim_loss_fct = maxMarginRankingLoss

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None,task_type=None, bbx=None, bbx_mask=None, masked_bbx=None,bbx_labels_index=None, trimmed_video=None, trimmed_audio=None, decoder_mask_audio=None, audio=None, audio_mask=None, player_IDs=None, player_labels=None, player_mask=None, player_labels_index=None, *args, **kwargs):

        if self.task_config.datatype == "ourds-audio-bbx":
            bbx, bbx_mask, audio, audio_mask = audio, audio_mask, bbx, bbx_mask
            audio = audio.squeeze(1)
            audio = reshape_bbx(audio)
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            
        if self.unsup_pretrain:
            assert trimmed_video is not None
            video_orig = video.clone().detach()
            video_orig = video_orig.squeeze(1) 
            video_orig = normalize(video_orig, dim=-1) 
            video = trimmed_video.clone().detach()
            # video_mask = (masked_video*video_mask).float()
            # bbx_mask = (masked_bbx[:,:,0,:]*bbx_mask).float()   
            if "audio" in self.task_config.task_type:
                assert trimmed_audio is not None
                audio_orig = bbx.clone().detach()
                audio_orig = audio_orig.squeeze(1) 
                audio_orig = normalize(audio_orig, dim=-1)
                bbx = trimmed_audio.clone().detach()
                bbx = self.audio_embed(bbx)
                bbx = bbx.squeeze(1)
                decoder_mask_audio = decoder_mask_audio.squeeze(1)
        
        elif "audio" in self.task_config.task_type:
            bbx = self.audio_embed(bbx)
            bbx = bbx.squeeze(1)
        
            
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        
        if self.task_config.fine_tune_extractor == False:
            video = self.normalize_video(video)
            bbx_mask = bbx_mask.view(-1, bbx_mask.shape[-1])
        

        if self.multibbxs:
            batch_sz,_,bbx_num,max_frame_num,fea_sz = bbx.shape
            bbx = bbx.permute((0, 1, 3, 2, 4)).reshape(batch_sz,_,max_frame_num,fea_sz*bbx_num)
            masked_bbx = masked_bbx.permute((0, 1, 3, 2, 4)).reshape(batch_sz,_,max_frame_num,fea_sz*bbx_num)
            #bbx = self.bbx_fea_fusion_two(bbx)


        if "audio" not in self.task_config.task_type:
            bbx = self.normalize_bbx(bbx)
        
        if self.unsup_pretrain:
            bbx_mask = torch.ones((bbx.shape[0],bbx.shape[1]), device=bbx.device)
            video_mask = torch.ones((video.shape[0],video.shape[1]), device=video.device)
                

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        # Get sequence and visual output and bbx output
        if self.unsup_pretrain and self.task_config.multi_mae == True:
            t_sum = 4
            t_rand = int(random.random() * t_sum)
            if t_rand == 0:
                t_rand = 1
            elif t_rand == t_sum:
                t_rand = t_sum - 1
            video = video[:,:t_rand,:]
            video_mask = torch.ones((video.shape[0],video.shape[1]), device=video.device)
            bbx = bbx[:,:t_sum - t_rand,:]
            bbx_mask = torch.ones((bbx.shape[0],bbx.shape[1]), device=bbx.device)
            
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                        video, video_mask, shaped=True, task_type=task_type)
        
        
        if self.task_config.fine_tune_extractor == False:
            bbx_output = self.get_bbx_output(bbx,bbx_mask, shaped=True)
        
        if self.task_config.join_vision_audio and not self.task_config.fine_tune_extractor:
            visual_output = torch.add(visual_output, bbx_output)
        
        if self.task_config.fine_tune_extractor == False:
            video_mask = torch.cat([torch.ones(video_mask.size(0),visual_output.size(1)-video_mask.size(1),device=video_mask.device,dtype=video_mask.dtype),video_mask],dim=1)
        else:
            video_mask = torch.tensor(video_mask,device=video_mask.device,dtype=video_mask.dtype)
        
        if self.training:
            loss = 0.
            if self._stage_one:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                        video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss += sim_loss

            if self._stage_two:
                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])
                    bbx_labels_index = bbx_labels_index.view(-1, bbx_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_output(pairs_masked_text, token_type_ids,
                                                                                             attention_mask, masked_video, video_mask, shaped=True)
                    
                    if self.task_config.fine_tune_extractor == False:
                        masked_bbx = self.normalize_bbx(masked_bbx)
                        bbx_output_alm = self.get_bbx_output(masked_bbx,bbx_mask, shaped=True)
                    
                        if self.task_config.join_vision_audio:
                            visual_output_alm = torch.add(visual_output_alm, bbx_output_alm)
                            video_mask = torch.add(video_mask, bbx_mask)
                            video_mask[video_mask > 0] = 1
                            cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output_alm, visual_output_alm, attention_mask, video_mask)
                            sequence_cross_output, visual_cross_output = torch.split(cross_output, [attention_mask.size(-1), video_mask.size(-1)], dim=1)
                        else:
                            cross_output, pooled_output, concat_mask = self._get_cross_output_lang(sequence_output_alm, bbx_output_alm, visual_output_alm, attention_mask,bbx_mask, video_mask)
                            sequence_cross_output, bbx_cross_output, visual_cross_output = torch.split(cross_output, [attention_mask.size(-1), video_mask.size(-1), bbx_mask.size(-1)], dim=1)

                        alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
                        if not torch.isnan(alm_loss):
                            loss += alm_loss

                        nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)
                    
                        if not self.task_config.join_vision_audio:
                            nce_loss += self._calculate_mfm_loss(bbx_cross_output, bbx, bbx_mask, bbx_labels_index)
                    
                        if not torch.isnan(nce_loss):
                            loss += nce_loss

                    sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                    sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                    # sim_matrix = self.get_similarity_logits(sequence_output, bbx_output, attention_mask, video_mask,
                    #                                         shaped=True, _pretrain_joint=True)
                    # sim_loss_joint += self._pretrain_sim_loss_fct(sim_matrix)
                    # sim_matrix = self.get_similarity_logits(bbx_output, visual_output, bbx_mask, video_mask,
                    #                                         shaped=True, _pretrain_joint=True)
                    # sim_loss_joint += self._pretrain_sim_loss_fct(sim_matrix)
                    if not torch.isnan(sim_loss_joint):
                        loss += sim_loss_joint
                    if torch.isnan(loss):
                        print("nan")

                if (input_caption_ids is not None) and \
                        (self.task_config.do_pretrain
                         or (self.task_config.do_pretrain is False and (("caption" in self.task_config.task_type) or "unsup-pretrain" in self.task_config.task_type))):
                    if self.task_config.do_pretrain:
                        decoder_scores, res_tuples, _ = self._get_decoder_score(sequence_output_alm, visual_output_alm,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=True)
                    elif "caption" in self.task_config.task_type:
                        if self.task_config.fine_tune_extractor == True:
                            sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                            sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                            if not torch.isnan(sim_loss_joint):
                                loss += sim_loss_joint
                            decoder_scores, res_tuples = self._get_decoder_scoreVL(sequence_output, visual_output,
                                                                                input_ids, attention_mask, video_mask,
                                                                                input_caption_ids, decoder_mask, shaped=True,task_type = task_type)
                        else:  
                            
                            
                            if "-knowledge" in self.task_config.task_type:
                                decoder_scores, res_tuples, contrast_loss = self._get_decoder_score(sequence_output, visual_output,
                                                                                input_ids, attention_mask, video_mask,
                                                                                input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=True,task_type = task_type, player_IDs=player_IDs)
                                loss += contrast_loss
                            else:
                                decoder_scores, res_tuples, contrast_loss = self._get_decoder_score(sequence_output, visual_output,
                                                                                input_ids, attention_mask, video_mask,
                                                                                input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=True,task_type = task_type)
                        if "audio-bbx" in self.task_config.task_type:
                            decoder_video = self._get_decoder_bbx(visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, audio_mask, bbx_output, bbx_mask, shaped=True,task_type = task_type)
                        
                    elif self.task_config.task_type == "unsup-pretrain":
                        decoder_scores = self._get_decoder_unsup_pretrain(sequence_output, visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=True,task_type = task_type)
                    elif self.task_config.task_type == "unsup-pretrain-audio":
                        decoder_scores, decoder_video = self._get_decoder_unsup_pretrain_audio(visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask_audio, decoder_mask, bbx_output, bbx_mask, shaped=True,task_type = task_type)
                    else:
                        raise NotImplementedError
                    
                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = None
                    if self.unsup_pretrain and "audio" not in self.task_config.task_type:
                        # pretrain_mask = torch.cat([bbx_mask == 0, video_mask == 0], dim=1)
                        pretrain_mask = decoder_mask
                        decoder_loss = self.decoder_loss_fct(decoder_scores, video_orig, mask=pretrain_mask)
                    elif self.unsup_pretrain and "audio" in self.task_config.task_type:
                        pretrain_mask = decoder_mask
                        pretrain_mask_audio = decoder_mask_audio
                        
                        decoder_scores = self.audio_reverse(decoder_scores)

                        decoder_loss_audio = self.decoder_loss_fct(decoder_scores, audio_orig, mask=pretrain_mask_audio)
                        decoder_loss_video = self.decoder_loss_fct(decoder_video, video_orig, mask=pretrain_mask)
                        decoder_loss = decoder_loss_audio + decoder_loss_video
                    else:
                        decoder_loss = self.decoder_loss_fct(decoder_scores.view(-1, self.bert_config.vocab_size), output_caption_ids.view(-1))
                        if "audio-bbx" in self.task_config.task_type:
                            decoder_video = self.bbx_embed_reverse(decoder_video)
                            bbx_loss = self.bbx_loss(decoder_video, audio, mask=audio_mask)
                            decoder_loss = self.loss_balancer(decoder_loss, bbx_loss)
                    if decoder_loss is not None:
                        loss += decoder_loss

                if self.task_config.do_pretrain or self.task_config.task_type == "retrieval":
                    if self.task_config.do_pretrain:
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output_alm, visual_output_alm,
                                                                            attention_mask, video_mask, shaped=True)
                    elif self.task_config.task_type == "retrieval":
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output, visual_output,
                                                                            attention_mask, video_mask, shaped=True)
                    else:
                        raise NotImplementedError

                    sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
                    loss += sim_loss_text_visual

            return loss
        else:
            return None


    # def bbx_fea_fusion_one(self,bbx_fea):
    #     batch_sz,_,bbx_num,max_frame_num,fea_sz = bbx_fea.shape
    #     fused_bbx = torch.matmul(bbx_fea.permute((0, 1, 3, 4, 2)).view(batch_sz,-1,bbx_num),self.bbxs_fusion_one.repeat(batch_sz,1,1).cuda()).permute(0,2,1).view(batch_sz,-1,max_frame_num,fea_sz)
        
    #     return fused_bbx
    def bbx_fea_fusion_two(self,bbx_fea):
        batch_sz,_,bbx_num,max_frame_num,fea_sz = bbx_fea.shape
        #fused_bbx = torch.matmul(bbx_fea.permute((0, 1, 3, 4, 2)).view(batch_sz,-1,bbx_num),self.bbxs_fusion_one.repeat(batch_sz,1,1).cuda()).permute(0,2,1).view(batch_sz,-1,max_frame_num,fea_sz)
        bbx_fea = bbx_fea.permute((0, 1, 3, 4, 2)).reshape(batch_sz,_,max_frame_num,-1)
        fused_bbx = self.bbxs_fusion_two(bbx_fea)
        fused_bbx = self.bbxs_dropout(fused_bbx)
        return fused_bbx
        
    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    
    def get_fuzed_bbx_visual_output(self,video, bbx,bbx_mask=None,video_mask=None, shaped=False, task_type=None):
        if shaped is False:
            bbx = self.normalize_bbx(bbx)
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)
        
        visual_output, bbx_output = self.bottleneck(bbx, video,  bbx_mask,video_mask, output_all_encoded_layers=True, task_type=task_type)
        
        return  bbx_output, visual_output

    def get_bbx_output(self, bbx, bbx_mask, shaped=False, task_type=None):
        if shaped is False:
            bbx = self.normalize_bbx(bbx)
        bbx_layers, _ = self.bbx(bbx, bbx_mask, output_all_encoded_layers=True,task_type=task_type, triang_attn=self.task_config.visual_use_diagonal_masking)
        bbx_output = bbx_layers[-1]
        return  bbx_output
    
    def get_visual_output(self, video, video_mask, shaped=False, task_type = None):
        
        if shaped is False and self.task_config.fine_tune_extractor == False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)
        
        if self.task_config.fine_tune_extractor == True:
            if self.task_config.extractor == "timesformer":
                video_feat = self.visual.model.forward_time_features(video.squeeze(1))
            else:
                video_feat = self.visual.forward_time_features(video.squeeze(1))
            return video_feat
        else:
            visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True,task_type=task_type, triang_attn=self.task_config.visual_use_diagonal_masking)
            visual_output = visual_layers[-1]
        return visual_output
    
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        text_output = encoded_layers[-1]
        
        return text_output

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, task_type = None):
        
        text_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=shaped)
        visual_output = self.get_visual_output(video, video_mask, shaped=shaped, task_type=task_type)

        return text_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)

        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        if self.context_only:
            concat_features = visual_output
            concat_mask = video_mask
            concat_type = video_type_

        if self.cross is not None:
            cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
            cross_output = cross_layers[-1]
            return cross_output, pooled_output, concat_mask # pooled not used
        else:
            cross_output = concat_features
            return cross_output, None, concat_mask # pooled not used
        
    def _get_cross_output_lang(self, sequence_output, bbx_output, visual_output, sequence_mask, bbx_mask, video_mask):
        concat_features = torch.cat((sequence_output, bbx_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((sequence_mask,bbx_mask, video_mask), dim=1)

        text_type_ = torch.zeros_like(bbx_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        if self.context_only:
            concat_features = visual_output
            concat_mask = video_mask
            concat_type = video_type_

        if self.cross is not None:
            cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
            cross_output = cross_layers[-1]
            return cross_output, pooled_output, concat_mask # pooled not used
        else:
            cross_output = concat_features
            return cross_output, None, concat_mask # pooled not used
        
    def _get_cross_output_VL(self, sequence_output, visual_output, sequence_mask, video_mask):
        if len(visual_output.shape) == 2:
            visual_output = visual_output.unsqueeze(1)
        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((sequence_mask,video_mask), dim=1)

        text_type_ = torch.zeros_like(sequence_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        if self.context_only:
            concat_features = visual_output
            concat_mask = video_mask
            concat_type = video_type_

        if self.cross is not None:
            cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
            cross_output = cross_layers[-1]
            return cross_output, pooled_output, concat_mask # pooled not used
        else:
            cross_output = concat_features
            return cross_output, None, concat_mask # pooled not used
        


    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if (self._stage_two and _pretrain_joint is False) or self.train_sim_after_cross:
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
            if self.task_config.use_mil is False:
                text_out = F.normalize(text_out, dim=-1)
                video_out = F.normalize(video_out, dim=-1)
            retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits
    
    def _get_decoder_unsup_pretrain(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=False,task_type = None):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        # cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        # remove the cross-attention with 'start', 'end' language token;
        attention_mask = torch.zeros_like(attention_mask)

        cross_output, pooled_output, concat_mask = self._get_cross_output(bbx_output, visual_output, bbx_mask, video_mask)
        decoder_cat = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask,task_type=task_type)
        return decoder_cat
    
    def _get_decoder_unsup_pretrain_audio(self, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask_audio, decoder_mask_video, bbx_output, bbx_mask, shaped=False,task_type = None):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask_audio = decoder_mask_audio.view(-1, decoder_mask_audio.shape[-1])

        # cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        # remove the cross-attention with 'start', 'end' language token;
        attention_mask = torch.zeros_like(attention_mask)

        cross_output, pooled_output, concat_mask = self._get_cross_output(bbx_output, visual_output, bbx_mask, video_mask) # cat(audio, video)
        if self.task_config.full_cross == True:
            audio = cross_output
            video = cross_output
            audio_mask = concat_mask
            video_mask = concat_mask
        else:
            audio = cross_output[:,:bbx_output.shape[1],:]
            video = cross_output[:,bbx_output.shape[1]:,:]
            audio_mask = concat_mask[:,:bbx_output.shape[1]]
            video_mask = concat_mask[:,bbx_output.shape[1]:]
        decoder_audio = self.decoder_audio(input_caption_ids, encoder_outs=audio, answer_mask=decoder_mask_audio, encoder_mask=audio_mask,task_type=task_type)
        decoder_video = self.decoder_video(input_caption_ids, encoder_outs=video, answer_mask=decoder_mask_video, encoder_mask=video_mask,task_type=task_type)
        return decoder_audio, decoder_video
    
    def _get_decoder_bbx(self, visual_output, input_ids, attention_mask, video_encoder_mask, input_caption_ids, decoder_mask_bbx, support_output, support_encoder_mask, shaped=False,task_type = None):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_encoder_mask = video_encoder_mask.view(-1, video_encoder_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])

        # cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        # remove the cross-attention with 'start', 'end' language token;
        attention_mask = torch.zeros_like(attention_mask)

        cross_output, pooled_output, concat_mask = self._get_cross_output(support_output, visual_output, support_encoder_mask, video_encoder_mask) # cat(audio, video)
        
        video = cross_output[:,support_output.shape[1]:,:]
        video_encoder_mask = concat_mask[:,support_output.shape[1]:]
        
        decoder_video = self.decoder_bbx(input_caption_ids, encoder_outs=video, answer_mask=decoder_mask_bbx, encoder_mask=video_encoder_mask,task_type=task_type)
        return decoder_video
    
    def _get_decoder_scoreVL(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, shaped=False,task_type = None):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        attention_mask = torch.zeros_like(attention_mask)
        cross_output, pooled_output, concat_mask = self._get_cross_output_VL(sequence_output, visual_output, attention_mask, video_mask)

        # cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        # remove the cross-attention with 'start', 'end' language token;

        decoder_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask,task_type=task_type)
        return decoder_scores, res_tuples

    def _get_decoder_score(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=False,task_type = None, player_IDs = None):

        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        attention_mask = torch.zeros_like(attention_mask)
        cross_output, pooled_output, concat_mask = self._get_cross_output(bbx_output, visual_output, bbx_mask, video_mask)
        # cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output, visual_output, attention_mask, video_mask)
        # remove the cross-attention with 'start', 'end' language token;
        contrastive_loss = 0
        if "-knowledge" in self.task_config.task_type and self.training:
            # use contrastive learning
            # contrastive_loss = custom_contrastive_loss(cross_output, sequence_output)
            # cross_output = torch.cat((cross_output,sequence_output), dim = 1)
            # concat_mask = torch.cat((concat_mask, attention_mask),dim=-1)
            # player_logits = self.player_classifier(cross_output.mean(dim=1))
            # player_loss = self.player_loss(player_logits.float(), player_IDs.float())
            # contrastive_loss += player_loss
            cross_output = torch.cat((cross_output,sequence_output), dim = 1)
            concat_mask = torch.cat((concat_mask, attention_mask),dim=-1)
            for layer in self.cross_attn_layers:
                cross_output = layer(cross_output, concat_mask.unsqueeze(1).unsqueeze(1))

        decoder_scores = self.decoder(input_caption_ids, encoder_outs=cross_output, answer_mask=decoder_mask, encoder_mask=concat_mask,task_type=task_type)
        return decoder_scores, res_tuples, contrastive_loss

    def decoder_caption(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=False, get_logits=False,task_type = None):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        # if bbx_output.shape != visual_output.shape:
        #     batch_size = visual_output.shape[0]
        #     bbx_output = bbx_output[:batch_size]
        #     bbx_mask = bbx_mask[:batch_size]

        decoder_scores, _, _ = self._get_decoder_score(sequence_output, visual_output,
                                                    input_ids, attention_mask, video_mask,
                                                    input_caption_ids, decoder_mask, bbx_output, bbx_mask, shaped=True,task_type = task_type)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)
        return decoder_scores_result
    
    def decoder_captionVL(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask, shaped=False, get_logits=False,task_type = None):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        # if bbx_output.shape != visual_output.shape:
        #     batch_size = visual_output.shape[0]
        #     bbx_output = bbx_output[:batch_size]
        #     bbx_mask = bbx_mask[:batch_size]

        decoder_scores, _ = self._get_decoder_scoreVL(sequence_output, visual_output,
                                                    input_ids, attention_mask, video_mask,
                                                    input_caption_ids, decoder_mask, shaped=True,task_type = task_type)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)
        return decoder_scores_result
    
    