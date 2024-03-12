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

import random
import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'visual_config.json'
WEIGHTS_NAME = 'visual_pytorch_model.bin'

class FEAModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config):
        super(FEAModel, self).__init__(config)
        self.embeddings = VisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)

    def forward(self, video, attention_mask=None, output_all_encoded_layers=True,task_type = None, triang_attn=False):

        if attention_mask is None:
            attention_mask = torch.ones(video.size(0), video.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if (len(video.shape) == 2):
            video = video.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        embedding_output = self.embeddings(video,task_type)
        if not triang_attn:

            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
            extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)

        
        if triang_attn:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
            sz_b, len_s, _ = embedding_output.size()
            subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=embedding_output.device, dtype=embedding_output.dtype), diagonal=0)
            self_attn_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1).unsqueeze(1)  # b x 1 x ls x ls
            slf_attn_mask = ((1.0 - extended_attention_mask) + self_attn_mask).gt(0).to(dtype=self.dtype)
            extended_attention_mask = slf_attn_mask * -10000.0
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class COURTModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config):
        super(COURTModel, self).__init__(config)
        config.vocab_size=1536

        self.embeddings = VisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)

    def forward(self, video, attention_mask=None, output_all_encoded_layers=True,task_type = None):

        if attention_mask is None:
            attention_mask = torch.ones(video.size(0), video.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(video,task_type)

        prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
        extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class VisualConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `VisualModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file=4096,
                 hidden_size=768,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02):
        """Constructs VisualConfig.

        Args:
            vocab_size_or_config_json_file: Size of the encoder layers and the pooler layer.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class VisualEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(VisualEmbeddings, self).__init__()

        self.to_visual_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #my code for prefix tuning
        # self.wtes = []
        # self.control_trans_list = []
        # for idx,preseqlen in enumerate(config.preseqlens):
        #     self.wtes.append(torch.rand(preseqlen, config.mid_dims[idx]))
        #     self.control_trans_list.append(nn.Sequential(
        #         nn.Linear(config.mid_dims[idx], config.hidden_size),
        #         # nn.Tanh(),
        #         # nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd)
        #         )

        #     )
        self.t0_emb = torch.nn.Parameter(torch.rand(max(config.preseqlens), max(config.mid_dims)))
        self.t1_emb = torch.nn.Parameter(torch.rand(max(config.preseqlens), max(config.mid_dims)))
        self.t2_emb = torch.nn.Parameter(torch.rand(max(config.preseqlens), max(config.mid_dims)))
        self.t3_emb = torch.nn.Parameter(torch.rand(max(config.preseqlens), max(config.mid_dims)))
        
        self.t0_project_matrix = nn.Sequential(
                 nn.Linear(max(config.mid_dims), config.hidden_size)
        )
        self.t1_project_matrix = nn.Sequential(
                 nn.Linear(max(config.mid_dims), config.hidden_size)
        )

        self.t2_project_matrix = nn.Sequential(
                 nn.Linear(max(config.mid_dims), config.hidden_size)
        )
        self.t3_project_matrix = nn.Sequential(
                 nn.Linear(max(config.mid_dims), config.hidden_size)
        )
        self.wtes = [self.t0_emb,self.t1_emb,self.t2_emb,self.t3_emb]
        self.control_trans_list = [self.t0_project_matrix,self.t1_project_matrix,self.t2_project_matrix,self.t3_project_matrix]
        
        self.use_prefix_tuning = config.use_prefix_tuning

    def forward(self, input_embeddings,task_type):
        
        if self.use_prefix_tuning !=0:
            prefix_embeddingsByTypes = None
            prefix_after_projection  = None
            for tid,task in enumerate(task_type.tolist()):
                if  tid == 0:
                    prefix_embeddingsByTypes = self.wtes[task].view(1,self.wtes[task].shape[0],self.wtes[task].shape[1])
                    prefix_after_projection = self.control_trans_list[task](prefix_embeddingsByTypes)

                else:
                    prefix_after_projection_for_one_sample = self.control_trans_list[task](self.wtes[task].view(1,self.wtes[task].shape[0],self.wtes[task].shape[1]))
                    prefix_after_projection = torch.cat([prefix_after_projection,prefix_after_projection_for_one_sample],dim=0)


            prefix_after_projection = prefix_after_projection.to(input_embeddings.device)
            seq_length = input_embeddings.size(1) + prefix_after_projection.size(1)

        else:
            seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)

        visual_embeddings = self.to_visual_embeddings(input_embeddings)
        # words_embeddings = self.transform_act_fn(words_embeddings)

        if self.use_prefix_tuning !=0:
            visual_embeddings = torch.cat([prefix_after_projection,visual_embeddings],dim=1)

        position_embeddings = self.position_embeddings(position_ids)
        # embeddings = words_embeddings
        embeddings = visual_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VisualSelfAttention(nn.Module):
    def __init__(self, config):
        super(VisualSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # print("!!!!!!!!!!!!!!!!{}".format(x.shape))
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in VisualModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class VisualSelfOutput(nn.Module):
    def __init__(self, config):
        super(VisualSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualAttention(nn.Module):
    def __init__(self, config):
        super(VisualAttention, self).__init__()
        self.self = VisualSelfAttention(config)
        self.output = VisualSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class VisualIntermediate(nn.Module):
    def __init__(self, config):
        super(VisualIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class VisualOutput(nn.Module):
    def __init__(self, config):
        super(VisualOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualLayer(nn.Module):
    def __init__(self, config):
        super(VisualLayer, self).__init__()
        self.attention = VisualAttention(config)
        self.intermediate = VisualIntermediate(config)
        self.output = VisualOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super(VisualEncoder, self).__init__()
        layer = VisualLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class VisualPooler(nn.Module):
    def __init__(self, config):
        super(VisualPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(VisualPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualLMPredictionHead, self).__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.weight = visual_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(visual_model_embedding_weights.size(1)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(self.weight) + self.bias
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualOnlyMLMHead, self).__init__()
        self.predictions = VisualLMPredictionHead(config, visual_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class VisualOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(VisualOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class VisualPreTrainingHeads(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualPreTrainingHeads, self).__init__()
        self.predictions = VisualLMPredictionHead(config, visual_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config, dim=768):
        super(VisualModel, self).__init__(config)
        config.vocab_size=dim
        self.embeddings = VisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)

    def forward(self, video, attention_mask=None, output_all_encoded_layers=True,task_type = None, triang_attn=False):

        if attention_mask is None:
            attention_mask = torch.ones(video.size(0), video.size(1))
        
        
        embedding_output = self.embeddings(video,task_type)
        
        if not triang_attn:

            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
            extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)

        
        if triang_attn:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
            sz_b, len_s, _ = embedding_output.size()
            subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=embedding_output.device, dtype=embedding_output.dtype), diagonal=0)
            self_attn_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1).unsqueeze(1)  # b x 1 x ls x ls
            slf_attn_mask = ((1.0 - extended_attention_mask) + self_attn_mask).gt(0).to(dtype=self.dtype)
            extended_attention_mask = slf_attn_mask * -10000.0

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BBXModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config, dim=1536):
        super(BBXModel, self).__init__(config)
        config.vocab_size=dim
        self.embeddings = VisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)

    def forward(self, video, attention_mask=None, output_all_encoded_layers=True,task_type = None, triang_attn=False):

        if attention_mask is None:
            attention_mask = torch.ones(video.size(0), video.size(1))

        
        embedding_output = self.embeddings(video,task_type)
        
        if not triang_attn:

            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
            extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)

        
        if triang_attn:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
            sz_b, len_s, _ = embedding_output.size()
            subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=embedding_output.device, dtype=embedding_output.dtype), diagonal=0)
            self_attn_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1).unsqueeze(1)  # b x 1 x ls x ls
            slf_attn_mask = ((1.0 - extended_attention_mask) + self_attn_mask).gt(0).to(dtype=self.dtype)
            extended_attention_mask = slf_attn_mask * -10000.0
            
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output