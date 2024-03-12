import torch
import torch.nn as nn

from modules.module_PET import AdaptFormer, VanillaEncoder, VanillaEncoderLin, VanillaEncoderOrig, VanillaEncoderGaze, VanillaEncoderScaledFusion

class MTB_Bottleneck_Git(nn.Module):
    
    # Cred to :github.com/NMS05
    # orig code: https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks/blob/main/MBT/models/pet_modules.py
    #
    # This is a havily modified version of the original code
    #
    
    def __init__(self, num_latents, dim, num_layers=0, enc_1=None, enc_2=None, use_conv=False):
        '''
        num_latents: number of latents to use for fusion (B, num_latents, emb_dim)
        dim: dimension of the bottleneck
        num_layers: number of layers to use for the encoder
        enc_1: layers from the encoder 1
        enc_2: layers from the encoder 2
        '''
        super(MTB_Bottleneck_Git, self).__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        self.num_latents = num_latents
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        
        self.enc_1_layer_skip = len(self.enc_1.encoder.layer) - self.num_layers
        self.enc_2_layer_skip = len(self.enc_2.encoder.layer) - self.num_layers

        encoder_layers = nn.ModuleList()
        for i in range(num_layers):

            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoder(num_latents=num_latents, enc_1=self.enc_1.encoder.layer[i+self.enc_1_layer_skip], enc_2=self.enc_2.encoder.layer[i+self.enc_2_layer_skip]))

            # Frozen Transformer Encoder with AdaptFormer 
            # encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim, enc_1=self.enc_1.encoder.layer[i+self.enc_1_layer_skip], enc_2=self.enc_2.encoder.layer[i+self.enc_2_layer_skip]))
             
        if len(encoder_layers) > 0:
            self.audio_visual_blocks = nn.Sequential(*encoder_layers)
        
        
    def forward_enc2(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_2)
       
        return x, extended_attention_mask
    
    def forward_enc1(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_1)
       
        return x, extended_attention_mask
    
    def forward_mod_enc(self,x, x_attn_mask,output_all_encoded_layers, task_type, mod_encoder):
        if x_attn_mask is None:
            x_attn_mask = torch.ones(x.size(0), x.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = x_attn_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=mod_encoder.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = mod_encoder.embeddings(x, task_type)
        
        prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
        extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)
        
        all_encoder_layers = []
        hidden_states = embedding_output
        for i in range(self.enc_1_layer_skip):
            curr_encoder_block = mod_encoder.encoder.layer[i]
            hidden_states = curr_encoder_block(hidden_states,extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if self.enc_1_layer_skip == 0:
            all_encoder_layers.append(hidden_states)
             
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers = all_encoder_layers[-1]
            
            
        return all_encoder_layers, extended_attention_mask
        

    def fusion_encoder(self,x,y, x_attn_mask, y_attn_mask):     
        x = x[-1]
        y = y[-1]
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            x,y = blk(x,y,x_attn_mask,y_attn_mask)
            

        # x = self.spec_post_norm(x)
        # y = self.rgb_post_norm(y)

        return x,y
        
    def forward(self, x,y, x_attn_mask=None, y_attn_mask=None,output_all_encoded_layers=True, task_type=None):

        
        x, x_mask = self.forward_enc1(x, x_attn_mask,output_all_encoded_layers, task_type)
        y, y_mask = self.forward_enc2(y, y_attn_mask,output_all_encoded_layers, task_type)
        
        if self.num_layers > 0:
            x,y = self.fusion_encoder(x,y, x_mask, y_mask)
        else:
            x, y = x[-1], y[-1]

        # logits = (x+y)*0.5
        
        return x , y
    
    

class MTB_Bottleneck_Orig(nn.Module):
    
    # Cred to :github.com/NMS05
    # orig code: https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks/blob/main/MBT/models/pet_modules.py
    #
    # This is a havily modified version of the original code
    #
    
    def __init__(self, num_latents, dim, num_layers=0, enc_1=None, enc_2=None, args=None, use_conv=False):
        '''
        num_latents: number of latents to use for fusion (B, num_latents, emb_dim)
        dim: dimension of the bottleneck
        num_layers: number of layers to use for the encoder
        enc_1: layers from the encoder 1
        enc_2: layers from the encoder 2
        '''
        super(MTB_Bottleneck_Orig, self).__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        self.num_latents = num_latents
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        
        self.enc_1_layer_skip = len(self.enc_1.encoder.layer) - self.num_layers
        self.enc_2_layer_skip = len(self.enc_2.encoder.layer) - self.num_layers

        encoder_layers = nn.ModuleList()
        for i in range(num_layers):

            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoderOrig(num_latents, [self.enc_1.encoder.layer[i+self.enc_1_layer_skip], self.enc_2.encoder.layer[i+self.enc_2_layer_skip]], use_conv=use_conv))

            # Frozen Transformer Encoder with AdaptFormer 
            # encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim, enc_1=self.enc_1.encoder.layer[i+self.enc_1_layer_skip], enc_2=self.enc_2.encoder.layer[i+self.enc_2_layer_skip]))
             
        if len(encoder_layers) > 0:
            self.audio_visual_blocks = nn.Sequential(*encoder_layers)
        
        
    def forward_enc2(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_2)
       
        return x, extended_attention_mask
    
    def forward_enc1(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_1)
       
        return x, extended_attention_mask
    
    def forward_mod_enc(self,x, x_attn_mask,output_all_encoded_layers, task_type, mod_encoder):
        if x_attn_mask is None:
            x_attn_mask = torch.ones(x.size(0), x.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = x_attn_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=mod_encoder.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = mod_encoder.embeddings(x, task_type)
        
        prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
        extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)
        
        all_encoder_layers = []
        hidden_states = embedding_output
        for i in range(self.enc_1_layer_skip):
            curr_encoder_block = mod_encoder.encoder.layer[i]
            hidden_states = curr_encoder_block(hidden_states,extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if self.enc_1_layer_skip == 0:
            all_encoder_layers.append(hidden_states)
             
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers = all_encoder_layers[-1]
            
            
        return all_encoder_layers, extended_attention_mask
        

    def fusion_encoder(self,x,y, x_mask, y_mask):     
        x = x[-1]
        y = y[-1]
        bottleneck_v = torch.empty(x.shape[0],self.num_latents,768).normal_(std=0.02).to(x.device)
        x_mask = torch.cat([x_mask, torch.ones(x_mask.shape[0],self.num_latents).to(x_mask.device).unsqueeze(1).unsqueeze(2)], dim=-1)
        y_mask = torch.cat([y_mask, torch.ones(y_mask.shape[0],self.num_latents).to(y_mask.device).unsqueeze(1).unsqueeze(2)], dim=-1)
        
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            [x,y], bottleneck_v = blk([x,y],[x_mask,y_mask], bottleneck_v)
            

        # x = self.spec_post_norm(x)
        # y = self.rgb_post_norm(y)

        return x,y
        
    def forward(self, x,y, x_attn_mask=None, y_attn_mask=None,output_all_encoded_layers=True, task_type=None):

        
        x, x_mask = self.forward_enc1(x, x_attn_mask,output_all_encoded_layers, task_type)
        y, y_mask = self.forward_enc2(y, y_attn_mask,output_all_encoded_layers, task_type)
        
        
        if self.num_layers > 0:
            x,y = self.fusion_encoder(x,y, x_mask, y_mask)
        else:
            x, y = x[-1], y[-1]

        # logits = (x+y)*0.5
        
        return x , y
    
    
class MTB_Bottleneck_Gaze(nn.Module):
    
    
    def __init__(self, num_latents, dim, num_layers=0, enc_1=None, enc_2=None, args=None):
        '''
        num_latents: number of latents to use for fusion (B, num_latents, emb_dim)
        dim: dimension of the bottleneck
        num_layers: number of layers to use for the encoder
        enc_1: layers from the encoder 1
        enc_2: layers from the encoder 2
        '''
        super(MTB_Bottleneck_Gaze, self).__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        self.num_latents = num_latents
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        
        self.enc_1_layer_skip = len(self.enc_1.encoder.layer) - self.num_layers
        self.enc_2_layer_skip = len(self.enc_2.encoder.layer) - self.num_layers

        encoder_layers = nn.ModuleList()
        for i in range(num_layers):

            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoderGaze(num_latents, [self.enc_1.encoder.layer[i+self.enc_1_layer_skip], self.enc_2.encoder.layer[i+self.enc_2_layer_skip]]))

            # Frozen Transformer Encoder with AdaptFormer 
            # encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim, enc_1=self.enc_1.encoder.layer[i+self.enc_1_layer_skip], enc_2=self.enc_2.encoder.layer[i+self.enc_2_layer_skip]))
            
        if len(encoder_layers) > 0:
            self.audio_visual_blocks = nn.Sequential(*encoder_layers)
        
        
    def forward_enc2(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_2)
    
        return x, extended_attention_mask
    
    def forward_enc1(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_1)
    
        return x, extended_attention_mask
    
    def forward_mod_enc(self,x, x_attn_mask,output_all_encoded_layers, task_type, mod_encoder):
        if x_attn_mask is None:
            x_attn_mask = torch.ones(x.size(0), x.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = x_attn_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=mod_encoder.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = mod_encoder.embeddings(x, task_type)
        
        prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
        extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)
        
        all_encoder_layers = []
        hidden_states = embedding_output
        for i in range(self.enc_1_layer_skip):
            curr_encoder_block = mod_encoder.encoder.layer[i]
            hidden_states = curr_encoder_block(hidden_states,extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if self.enc_1_layer_skip == 0:
            all_encoder_layers.append(hidden_states)
            
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers = all_encoder_layers[-1]
            
            
        return all_encoder_layers, extended_attention_mask
        

    def fusion_encoder(self,x,y, x_mask, y_mask):     
        x = x[-1]
        y = y[-1]
        x_mask = torch.ones_like(x_mask)
        y_mask = torch.ones_like(y_mask)
        
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            [x,y] = blk([x,y],[x_mask,y_mask])
            

        # x = self.spec_post_norm(x)
        # y = self.rgb_post_norm(y)

        return x,y
        
    def forward(self, x,y, x_attn_mask=None, y_attn_mask=None,output_all_encoded_layers=True, task_type=None):

        
        x, x_mask = self.forward_enc1(x, x_attn_mask,output_all_encoded_layers, task_type)
        y, y_mask = self.forward_enc2(y, y_attn_mask,output_all_encoded_layers, task_type)
        
        
        if self.num_layers > 0:
            x,y = self.fusion_encoder(x,y, x_mask, y_mask)
        else:
            x, y = x[-1], y[-1]

        # logits = (x+y)*0.5
        
        return x , y
    

class MTB_Bottleneck_Lin(nn.Module):
    
    
    def __init__(self, num_latents, dim, num_layers=0, enc_1=None, enc_2=None, args=None):
        '''
        num_latents: number of latents to use for fusion (B, num_latents, emb_dim)
        dim: dimension of the bottleneck
        num_layers: number of layers to use for the encoder
        enc_1: layers from the encoder 1
        enc_2: layers from the encoder 2
        '''
        super(MTB_Bottleneck_Lin, self).__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        self.num_latents = num_latents
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        
        self.enc_1_layer_skip = len(self.enc_1.encoder.layer) - self.num_layers
        self.enc_2_layer_skip = len(self.enc_2.encoder.layer) - self.num_layers

        encoder_layers = nn.ModuleList()
        for i in range(num_layers):

            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoderLin(num_latents, [self.enc_1.encoder.layer[i+self.enc_1_layer_skip], self.enc_2.encoder.layer[i+self.enc_2_layer_skip]]))

            # Frozen Transformer Encoder with AdaptFormer 
            # encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim, enc_1=self.enc_1.encoder.layer[i+self.enc_1_layer_skip], enc_2=self.enc_2.encoder.layer[i+self.enc_2_layer_skip]))
            
        if len(encoder_layers) > 0:
            self.audio_visual_blocks = nn.Sequential(*encoder_layers)
        
        
    def forward_enc2(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_2)
    
        return x, extended_attention_mask
    
    def forward_enc1(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_1)
    
        return x, extended_attention_mask
    
    def forward_mod_enc(self,x, x_attn_mask,output_all_encoded_layers, task_type, mod_encoder):
        if x_attn_mask is None:
            x_attn_mask = torch.ones(x.size(0), x.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = x_attn_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=mod_encoder.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = mod_encoder.embeddings(x, task_type)
        
        prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
        extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)
        
        all_encoder_layers = []
        hidden_states = embedding_output
        for i in range(self.enc_1_layer_skip):
            curr_encoder_block = mod_encoder.encoder.layer[i]
            hidden_states = curr_encoder_block(hidden_states,extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if self.enc_1_layer_skip == 0:
            all_encoder_layers.append(hidden_states)
            
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers = all_encoder_layers[-1]
            
            
        return all_encoder_layers, extended_attention_mask
        

    def fusion_encoder(self,x,y, x_mask, y_mask):     
        x = x[-1]
        y = y[-1]
        x_mask = torch.ones_like(x_mask)
        y_mask = torch.ones_like(y_mask)
        
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            [x,y] = blk([x,y],[x_mask,y_mask])
            

        # x = self.spec_post_norm(x)
        # y = self.rgb_post_norm(y)

        return x,y
        
    def forward(self, x,y, x_attn_mask=None, y_attn_mask=None,output_all_encoded_layers=True, task_type=None):

        
        x, x_mask = self.forward_enc1(x, x_attn_mask,output_all_encoded_layers, task_type)
        y, y_mask = self.forward_enc2(y, y_attn_mask,output_all_encoded_layers, task_type)
        
        
        if self.num_layers > 0:
            x,y = self.fusion_encoder(x,y, x_mask, y_mask)
        else:
            x, y = x[-1], y[-1]

        # logits = (x+y)*0.5
        
        return x , y
    
    

class Scaled_Fusion(nn.Module):
    
    
    def __init__(self,cross_config,num_layers=0, enc_1=None, enc_2=None):
        '''
        dim: dimension of the bottleneck
        num_layers: number of layers to use for the encoder
        enc_1: layers from the encoder 1
        enc_2: layers from the encoder 2
        '''
        super(Scaled_Fusion, self).__init__()
        
        self.num_layers = num_layers
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        
        self.enc_1_layer_skip = len(self.enc_1.encoder.layer) - self.num_layers
        self.enc_2_layer_skip = len(self.enc_2.encoder.layer) - self.num_layers

        encoder_layers = nn.ModuleList()
        for i in range(num_layers):

            # Vanilla Transformer Encoder (use for full fine tuning)
            encoder_layers.append(VanillaEncoderScaledFusion([self.enc_1.encoder.layer[i+self.enc_1_layer_skip], self.enc_2.encoder.layer[i+self.enc_2_layer_skip]], cross_attn_config=cross_config))

            
        if len(encoder_layers) > 0:
            self.audio_visual_blocks = nn.Sequential(*encoder_layers)
        
        
    def forward_enc2(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_2)
    
        return x, extended_attention_mask
    
    def forward_enc1(self,x, x_attn_mask,output_all_encoded_layers, task_type):
        
        x, extended_attention_mask = self.forward_mod_enc(x, x_attn_mask,output_all_encoded_layers, task_type, self.enc_1)
    
        return x, extended_attention_mask
    
    def forward_mod_enc(self,x, x_attn_mask,output_all_encoded_layers, task_type, mod_encoder):
        if x_attn_mask is None:
            x_attn_mask = torch.ones(x.size(0), x.size(1))

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = x_attn_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=mod_encoder.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = mod_encoder.embeddings(x, task_type)
        
        prefix_attention_mask = torch.zeros(embedding_output.size(0),1,1,embedding_output.size(1)-extended_attention_mask.size(-1),device=extended_attention_mask.device)
        extended_attention_mask = torch.cat([prefix_attention_mask,extended_attention_mask],dim=-1)
        
        all_encoder_layers = []
        hidden_states = embedding_output
        for i in range(self.enc_1_layer_skip):
            curr_encoder_block = mod_encoder.encoder.layer[i]
            hidden_states = curr_encoder_block(hidden_states,extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if self.enc_1_layer_skip == 0:
            all_encoder_layers.append(hidden_states)
            
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_layers = all_encoder_layers[-1]
            
            
        return all_encoder_layers, extended_attention_mask
        

    def fusion_encoder(self,x,y, x_mask, y_mask):     
        x = x[-1]
        y = y[-1]
        x_mask = torch.ones_like(x_mask)
        y_mask = torch.ones_like(y_mask)
        
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            # [x_prev,y_prev] = [x,y]
            [x,y] = blk([x,y],[x_mask,y_mask])
            # [x,y] = [x_prev+x,y_prev+y]
            

        # x = self.spec_post_norm(x)
        # y = self.rgb_post_norm(y)

        return x,y
        
    def forward(self, x,y, x_attn_mask=None, y_attn_mask=None,output_all_encoded_layers=True, task_type=None):

        
        x, x_mask = self.forward_enc1(x, x_attn_mask,output_all_encoded_layers, task_type)
        y, y_mask = self.forward_enc2(y, y_attn_mask,output_all_encoded_layers, task_type)
        
        
        if self.num_layers > 0:
            x,y = self.fusion_encoder(x,y, x_mask, y_mask)
        else:
            x, y = x[-1], y[-1]

        # logits = (x+y)*0.5
        
        return x , y