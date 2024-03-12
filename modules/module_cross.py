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

import os
import numpy as np
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import random

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
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
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
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
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        if concat_type is None:
            concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossSelfAttention(nn.Module):
    def __init__(self, config):
        super(CrossSelfAttention, self).__init__()
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
        # Apply the attention mask is (precomputed for all layers in CrossModel forward() function)
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
        return context_layer, attention_scores
    
'''
Just like the self-attention, we can also use the cross-attention to get the query, key, value
'''
class CrossSelfAttention_KVQ(nn.Module):
    def __init__(self, config):
        super(CrossSelfAttention_KVQ, self).__init__()
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
        return x.permute(0, 2, 1, 3)

    def forward(self, q, kv, attention_mask):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(kv)
        mixed_value_layer = self.value(kv)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in CrossModel forward() function)
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
        return context_layer, attention_scores
    
'''
Just like the self-attention, we can also use the cross-attention to get the query, key, value
'''
class CrossSelfAttention_KVQ_NoSelf(nn.Module):
    def __init__(self, config):
        super(CrossSelfAttention_KVQ_NoSelf, self).__init__()
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
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k,v, attention_mask):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in CrossModel forward() function)
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
        return context_layer, attention_scores


class CrossSelfOutput(nn.Module):
    def __init__(self, config):
        super(CrossSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = CrossSelfAttention(config)
        self.output = CrossSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output,_ = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output
    

class CrossAttention_KVQ(nn.Module):
    def __init__(self, config):
        super(CrossAttention_KVQ, self).__init__()
        self.self = CrossSelfAttention_KVQ(config)
        self.output = CrossSelfOutput(config)

    def forward(self, q,kv, attention_mask):
        self_output,_ = self.self(q,kv, attention_mask)
        attention_output = self.output(self_output, q)
        return attention_output
    
class CrossAttention_KVQ_NoSelf(nn.Module):
    def __init__(self, config):
        super(CrossAttention_KVQ_NoSelf, self).__init__()
        self.self = CrossSelfAttention_KVQ_NoSelf(config)
        self.output = CrossSelfOutput(config)

    def forward(self, q,k,v, attention_mask):
        self_output,_ = self.self(q,k,v, attention_mask)
        attention_output = self.output(self_output, q)
        return attention_output


class CrossIntermediate(nn.Module):
    def __init__(self, config):
        super(CrossIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CrossOutput(nn.Module):
    def __init__(self, config):
        super(CrossOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Hook function to capture attention scores and save them
def hook(module, input, output, layer_num, prefix="attention", out_dir="attention_scores"):
    head_scores = output[1][1].detach().cpu().numpy()
    for head in range(head_scores.shape[0]):
        scores = np.expand_dims(head_scores[head], axis=0)
        filename = os.path.join(out_dir, f"{prefix}_layer_{layer_num}_head_{head}.npy")
        # If file exists, load and append new scores, otherwise just save the scores
        if os.path.exists(filename):
            existing_scores = np.load(filename, allow_pickle=True)
            updated_scores = np.vstack([existing_scores, scores])
            np.save(filename, updated_scores)
        else:
            np.save(filename, scores)
    

class CrossLayer(nn.Module):
    def __init__(self, config):
        super(CrossLayer, self).__init__()
        self.attention = CrossAttention(config)
        self.intermediate = CrossIntermediate(config)
        self.output = CrossOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class CrossLayer_NoSelf(nn.Module):
    def __init__(self, config):
        super(CrossLayer_NoSelf, self).__init__()
        self.attention = CrossAttention_KVQ_NoSelf(config)
        self.intermediate = CrossIntermediate(config)
        self.output = CrossOutput(config)

    def forward(self, hidden_states, attention_mask):
        # Separate the hidden_states into two parts, its modalities and create 3 vectors
        # q - video
        # k - features
        # v - the original hidden_states (video , features)
        m1 = hidden_states[:, hidden_states.size(1)//2:, :] # modality 1, video
        m2 = hidden_states[:, :hidden_states.size(1)//2, :] # modality 2, features
        q = torch.cat((m1, m1), dim=1)
        k = torch.cat((m2, m2), dim=1)
        v = hidden_states
        attention_output = self.attention(q,k,v, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CrossEncoder(nn.Module):
    def __init__(self, config, no_self_attn=False):
        super(CrossEncoder, self).__init__()
        if no_self_attn:
            layer = CrossLayer_NoSelf(config)
        else:
            layer = CrossLayer(config)
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


class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(CrossPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class CrossLMPredictionHead(nn.Module):
    def __init__(self, config, cross_model_embedding_weights):
        super(CrossLMPredictionHead, self).__init__()
        self.transform = CrossPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(cross_model_embedding_weights.size(1),
                                 cross_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = cross_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(cross_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class CrossOnlyMLMHead(nn.Module):
    def __init__(self, config, cross_model_embedding_weights):
        super(CrossOnlyMLMHead, self).__init__()
        self.predictions = CrossLMPredictionHead(config, cross_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class CrossOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(CrossOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class CrossPreTrainingHeads(nn.Module):
    def __init__(self, config, cross_model_embedding_weights):
        super(CrossPreTrainingHeads, self).__init__()
        self.predictions = CrossLMPredictionHead(config, cross_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class CrossModel(PreTrainedModel):
    """
    CrossModel: A model designed to handle various cross-attention masking strategies.

    Features:
        - upper: Masks the upper triangle of the attention matrix, allowing each token to attend only to preceding tokens.
        - lower: Masks the lower triangle of the attention matrix, allowing each token to attend only to subsequent tokens.
        - random: Introduces random masking in the attention matrix.
        - dice: Divides the attention matrix into quadrants and masks the diagonal quadrants.
        - global: Ensures that the first two tokens always have global attention.
        - pipe: Indicates that the model should use a piped cross-attention mechanism.
        - no-inp: A modifier that can be combined with other features to modify their behavior.

    Combinations:
        Features can be combined using a "-" separator. For example, "upper-random" combines the "upper" and "random" features.

        - Single Feature: Use any single feature by itself, e.g., "upper", "lower", "random", etc.
        - Double Combinations: Combine any two features, e.g., "upper-random", "lower-dice", "random-global", etc.
        - Triple Combinations: Combine any three features, e.g., "upper-lower-random", "upper-random-dice", etc.

        Note: Not all combinations might be meaningful or valid. Ensure that each combination makes sense in the context of the application.

    Usage:
        To use a particular feature or combination, set the `cross_masking` argument to the desired string value when initializing the model. For example:
        model = CrossModel(config, args={"cross_masking": "upper-random"})

        Ensure that the model is in evaluation mode if you don't want any masking during evaluation.

    """
    def __init__(self, config, export_attn_scores=False, model_name="attention_scores", args=None):
        super(CrossModel, self).__init__(config)
        self.args = args
        if (args.cross_masking is not None ) and ("pipe" in args.cross_masking):
            self.cross_attn_pipes = True
        else:
            self.cross_attn_pipes = False
        self.embeddings = CrossEmbeddings(config)
        self.encoder = CrossEncoder(config, no_self_attn=self.cross_attn_pipes)
        self.pooler = CrossPooler(config)
        if self.args is not None and self.args.cross_masking is not None and "learnable" in self.args.cross_masking:
            # Random init boolean mask
            self.learnable_mask = nn.Parameter(torch.ones((args.max_frames*2, args.max_frames*2)), requires_grad=True)
        self.apply(self.init_weights)
        # Using a lambda function to pass additional arguments to the hook
        # Register the hook for all the attention layers in BERT
        if export_attn_scores:
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            for idx, layer in enumerate(self.encoder.layer):
                # Using a lambda function to pass additional arguments to the hook
                layer.attention.self.attention_score_hook = layer.attention.self.register_forward_hook(
                    lambda module, input, output, layer_num=idx: hook(module, input, output, layer_num, out_dir=model_name, prefix="cross_attention")
                )

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):
        

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        
        
        if self.args.cross_masking is not None:
            tmp = attention_mask
            attention_mask = torch.ones((attention_mask.shape[0],attention_mask.shape[1], attention_mask.shape[1])).to(attention_mask.device)
            if  "no-inp" not in self.args.cross_masking:
                attention_mask *= tmp.unsqueeze(1) 
            
            if "pipe" in self.args.cross_masking:
                # dice-no-self
                if  "no-inp" not in self.args.cross_masking:
                    attention_mask = torch.ones((attention_mask.shape[0],attention_mask.shape[1], attention_mask.shape[1])).to(attention_mask.device)
                    attn_mod2 = attention_mask[:, attention_mask.size(1)//2:]
                    mod2_cat = torch.cat((attn_mod2, attn_mod2), dim=1) # features
                    # apply the mask to the modality 1, first dimension
                    # attention_mask *= mod1_cat.unsqueeze(2)
                    attention_mask *= mod2_cat.unsqueeze(1)
                
            if "upper" in self.args.cross_masking:
                attention_mask *= torch.tril(torch.ones((attention_mask.shape[1],attention_mask.shape[1])).unsqueeze(0).to(attention_mask.device))
            elif "lower" in self.args.cross_masking:
                attention_mask *= torch.triu(torch.ones((attention_mask.shape[1],attention_mask.shape[1])).unsqueeze(0).to(attention_mask.device))
            if self.training and ("dicerand1" in self.args.cross_masking):
                thres_a = random.random()
                if(thres_a > 0.75):
                    attention_mask[:, :attention_mask.size(1)//2, :attention_mask.size(1)//2] = 0 
                    attention_mask[:, attention_mask.size(1)//2:, attention_mask.size(1)//2:] = 0    
            elif self.training and ("dicerand2" in self.args.cross_masking):
                thres_a = random.random()
                thres_b = random.random()
                if(thres_a > 0.5):
                    attention_mask[:, :attention_mask.size(1)//2, :attention_mask.size(1)//2] = 0 
                if(thres_b > 0.5):
                    attention_mask[:, attention_mask.size(1)//2:, attention_mask.size(1)//2:] = 0
                
            elif "dice" in self.args.cross_masking:
                attention_mask[:, :attention_mask.size(1)//2, :attention_mask.size(1)//2] = 0 
                attention_mask[:, attention_mask.size(1)//2:, attention_mask.size(1)//2:] = 0 
                
            if self.training and ("random" in self.args.cross_masking):
                random_thres = random.random()
                random_thres = random_thres if random_thres <= 0.75 else 0.75 
                random_m = torch.rand_like(attention_mask)
                random_m[random_m > random_thres] = 1
                random_m[random_m <= random_thres] = 0
                # Have some probability to mask the input only during training
                attention_mask *= random_m
            if "learnable" in self.args.cross_masking:
                mask = self.learnable_mask.expand_as(attention_mask)
                attention_mask *= mask
                    
            if ("global" in self.args.cross_masking):
                # global attention
                attention_mask[:, :2, :2] = 1
                    
                    
            extended_attention_mask = attention_mask.unsqueeze(1)
            
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # breakpoint()
        embedding_output = self.embeddings(concat_input, concat_type)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
