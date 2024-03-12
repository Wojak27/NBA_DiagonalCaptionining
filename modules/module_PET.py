import torch
import torch.nn as nn
from modules.module_cross import CrossAttention_KVQ, CrossIntermediate, CrossOutput

from modules.until_module import LayerNorm

class VanillaEncoder(nn.Module):
    def __init__(self, num_latents, enc_1, enc_2, use_conv=False):
        super(VanillaEncoder, self).__init__()
        self.use_conv = use_conv
        
        self.enc_1 = enc_1
        self.enc_2 = enc_2

        # Latents
        self.num_latents = num_latents
        self.latents_v = nn.Parameter(torch.empty(1,num_latents,768).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))
        if use_conv:
            print("Using Conv1D on fused latents")
            self.conv_a = nn.Conv1d(768, 768, 3, padding=1)
            self.conv_v = nn.Conv1d(768, 768, 3, padding=1)
            self.norm_a = LayerNorm(768)
            self.norm_v = LayerNorm(768)
            self.act_a = nn.GELU()
            self.act_v = nn.GELU()


    def attention(self,q,k,v): # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x
    
    # Latent Fusion
    def latent_fusion(self, audio_tokens, visual_tokens):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        # concat_ = torch.cat((audio_tokens,visual_tokens),dim=1)
        # cross attention (A -->> latent_a)
        
        fused_latents_a = self.attention(q=self.latents_v.expand(BS,-1,-1), k=audio_tokens, v=audio_tokens)
        if self.use_conv:
            fused_latents_a = self.conv_a(fused_latents_a.transpose(1,2)).transpose(1,2)
            fused_latents_a = self.norm_a(fused_latents_a)
            fused_latents_a = self.act_a(fused_latents_a)
        # cross attention (latent_a + V -->> V)
        visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fused_latents_a, v=fused_latents_a)
        # cross attention (V -->> latents_v)
        fused_latents_v = self.attention(q=self.latents_v.expand(BS,-1,-1), k=visual_tokens, v=visual_tokens)
        if self.use_conv:
            fused_latents_v = self.conv_v(fused_latents_a.transpose(1,2)).transpose(1,2)
            fused_latents_v = self.norm_v(fused_latents_v)
            fused_latents_v = self.act_v(fused_latents_v)
        audio_tokens = audio_tokens + self.scale_a * self.attention(q=audio_tokens, k=fused_latents_v, v=fused_latents_v)
        # cross attention (latent_v + A -->> A)
        return audio_tokens, visual_tokens
    
    def forward(self, x, y, x_attn_mask, y_attn_mask):
        
        orig_x = x
        orig_y = y
        # Bottleneck Fusion
        # TODO: Move this to after y_output
        x_fusion,y_fusion = self.latent_fusion(x,y)
        
        
        x = self.enc_1.attention(x_fusion, x_attn_mask)
        y = self.enc_2.attention(y_fusion, y_attn_mask)
        
        x_intermediate = self.enc_1.intermediate(x)
        y_intermediate = self.enc_2.intermediate(y)
        
        x = self.enc_1.output(x_intermediate, x)
        y = self.enc_2.output(y_intermediate, y)
        
        return x,y
    

class VanillaEncoderOrig(nn.Module):
    def __init__(self, num_latents, encoders, use_conv=False):
        super(VanillaEncoderOrig, self).__init__()
        self.use_conv = use_conv
        
        self.encoders = encoders

        # Latents
        self.num_latents = num_latents
        if use_conv:
            self.embedding_btl = nn.Sequential(*[
                nn.Linear(768, 768),
                LayerNorm(768),
                nn.GELU()    
            ])
            print("Using Linear on fused latents")
    
    def forward_enc(self, encoder, x, x_attn_mask):
        x = encoder.attention(x, x_attn_mask)
        x_intermediate = encoder.intermediate(x)
        x = encoder.output(x_intermediate, x)
        return x
    
    def forward(self, out_ms, attn_masks, bottleneck_v = None):
        
        
        # Below is just the regular vanilla encoder implemented just like in the original code
        assert bottleneck_v is not None, "Bottleneck needs to be forwarded"
        assert bottleneck_v.shape[1] == self.num_latents, "Bottleneck needs to be of shape (B, num_latents, emb_dim)"
        assert len(out_ms) == len(attn_masks), "Number of outputs and attn masks must be the same"
        
        if self.use_conv:
            bottleneck_v = self.embedding_btl(bottleneck_v)
        outs = []
        bls = []
        
        for i in range(len(self.encoders)):
            inp = torch.cat((out_ms[i], bottleneck_v), dim=1)
            x = self.forward_enc(self.encoders[i], inp , attn_masks[i])
            x_l, b_l = torch.split(x, [x.shape[1]-self.num_latents, self.num_latents], dim=1)
            outs.append(x_l)
            bls.append(b_l)
        
        bottleneck_v = torch.stack(bls, dim=-1).mean(dim=-1)
        
        
        return outs, bottleneck_v
    
    
class VanillaEncoderGaze(nn.Module):
    def __init__(self, n_latents, encoders):
        super(VanillaEncoderGaze, self).__init__()
        self.n_latents = n_latents
        self.encoders = encoders
    
    def forward_enc(self, encoder, x, x_attn_mask):
        x = encoder.attention(x, x_attn_mask)
        x_intermediate = encoder.intermediate(x)
        x = encoder.output(x_intermediate, x)
        return x
    
    def forward(self, out_ms, attn_masks):
        
        assert len(out_ms) == len(attn_masks), "Number of outputs and attn masks must be the same"
        
        outs = []
        ins = []
        ins.append(torch.cat((out_ms[0][:,:-self.n_latents,:], out_ms[1][:,-self.n_latents:,:]), dim=1))
        ins.append(torch.cat((out_ms[1][:,:-self.n_latents,:], out_ms[0][:,-self.n_latents:,:]), dim=1))
        
        for i in range(len(self.encoders)):
            outs.append(self.forward_enc(self.encoders[i], ins[i] , attn_masks[i]))
        
        return outs

class LinearBottleneck(nn.Module):
    def __init__(self,n_latents, embed_dim):
        super(LinearBottleneck, self).__init__()
        self.n_latents = n_latents
        self.embed_dim = embed_dim
        
        self.embedding_btl = nn.Sequential(*[
            nn.Linear(embed_dim, n_latents),
            LayerNorm(n_latents),
            nn.GELU(),
            nn.Linear(n_latents, embed_dim),
            LayerNorm(embed_dim),
            nn.GELU()
        ])
        
    def forward(self, x):
        return self.embedding_btl(x)

class VanillaEncoderLin(nn.Module):
    def __init__(self, n_latents, encoders):
        super(VanillaEncoderLin, self).__init__()
        self.n_latents = n_latents
        self.encoders = encoders
        self.bottleneck = LinearBottleneck(n_latents, 768)
    
    def forward_enc(self, encoder, x, x_attn_mask):
        x = encoder.attention(x, x_attn_mask)
        x_intermediate = encoder.intermediate(x)
        x = encoder.output(x_intermediate, x)
        return x
    
    def forward(self, out_ms, attn_masks):
        
        assert len(out_ms) == len(attn_masks), "Number of outputs and attn masks must be the same"
        
        outs = []
        ins = []
        
        for i in range(len(out_ms)):
            for j in range(len(out_ms)):
                if i != j:
                    ins.append(out_ms[i] + self.bottleneck(out_ms[j]))
        
        for i in range(len(self.encoders)):
            outs.append(self.forward_enc(self.encoders[i], ins[i] , attn_masks[i]))
        
        return outs


from torch.functional import F 
class DotProdAttention(nn.Module):
    def forward(self, q, kv, mask=None):

        query = q.clone()
        key = kv.clone()
        value = kv.clone()

        d_k = torch.Tensor([key.size(-1)]).to(key.device)

        score =  query @ key.transpose(-2, -1) / torch.sqrt(d_k)

        if mask is not None:
            score = score + mask

        score = F.softmax(score, dim=-1)
        out = score @ value

        return out


class VanillaEncoderScaledFusion(nn.Module):
    # Similar to: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9383573
    # But with cat instead of sum at the end
    
    def __init__(self, encoders, cross_attn_config):
        super(VanillaEncoderScaledFusion, self).__init__()
        self.cross_attn_layers = nn.ModuleList()
        self.encoders = encoders
        self.scalars = nn.ParameterList()
        for i in range(len(encoders)):
            # self.cross_attn_layers.append(DotProdAttention())
            self.cross_attn_layers.append(CrossAttention_KVQ(cross_attn_config))
            self.scalars.append(nn.Parameter(torch.empty(1).normal_(std=0.02)))
            
        self.cross_intermediate = CrossIntermediate(cross_attn_config)
        self.cross_out = CrossOutput(cross_attn_config)
        
    def forward_enc(self, q,kv, x_attn_mask, mod):
        # x = encoder.attention(x, x_attn_mask)
        out = self.cross_attn_layers[mod](q,kv, x_attn_mask)
        out = self.scalars[mod] * out + (1-self.scalars[mod]) * q
        return out
    
    def forward(self, out_ms, attn_masks):
        
        assert len(out_ms) == len(attn_masks), "Number of outputs and attn masks must be the same"
        
        outs = []
        
        # Allow free flow of information
        mask = torch.ones_like(attn_masks[0]) 
        
        for i in range(len(self.encoders)):
            for j in range(len(self.encoders)):
                if i != j:
                    outs.append(self.forward_enc(out_ms[i], out_ms[j] , mask, i))
        
        # cat outs in dim=1
        outs = torch.cat(outs, dim=1)
        
        intermidiete_out = self.cross_intermediate(outs)
        outs = self.cross_out(intermidiete_out, outs)
        
        # split the concat to the same size as the original
        
        tmp = 0
        out = []
        for i in range(len(out_ms)):
            out.append(outs[:,tmp:tmp+out_ms[i].shape[1],:])
            tmp += out_ms[i].shape[1]
            
        
        return out


        
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdaptFormer(nn.Module):
    
    def __init__(self, num_latents, dim, enc_1, enc_2):
        '''
        num_latents: number of latents to use for fusion (B, num_latents, emb_dim)
        dim: dimension of the bottleneck
        enc_1: layer from the encoder 1
        enc_2: layer from the encoder 2
        
        '''
        super(AdaptFormer, self).__init__()
        
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        
        """
        Freeze parameters
        """
        # Trainable by default = Spec+RGB pos embed and cls token, linear classifier
        
        # spec
        for p in self.enc_1.parameters():p.requires_grad=False

        # spec
        for p in self.enc_2.parameters():p.requires_grad=False  
        

        # Adapter params
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # Spectrogram
        self.spec_down = nn.Linear(768, dim)
        self.spec_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.spec_down.weight)
        nn.init.zeros_(self.spec_down.bias)
        nn.init.zeros_(self.spec_up.weight)
        nn.init.zeros_(self.spec_up.bias)
        self.spec_scale = nn.Parameter(torch.ones(1))

        # RGB images
        self.rgb_down = nn.Linear(768, dim)
        self.rgb_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.zeros_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)
        self.rgb_scale = nn.Parameter(torch.ones(1))

        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1,num_latents,768).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))
        self.scale_b = nn.Parameter(torch.zeros(1))
        


    def attention(self,q,k,v): # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x
    
    # Latent Fusion
    def latent_fusion(self, audio_tokens, visual_tokens):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((audio_tokens,visual_tokens),dim=1)
        # cross attention (AV -->> latents)
        fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_)
        # cross attention (latents -->> AV)
        audio_tokens = audio_tokens + self.scale_a * self.attention(q=audio_tokens, k=fused_latents, v=fused_latents)
        visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fused_latents, v=fused_latents)
        return audio_tokens, visual_tokens

    def forward_sequence_AF(self, x):
        x_down = self.spec_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.spec_up(x_down)
        return x_up

    def forward_visual_AF(self, x):
        x_down = self.rgb_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.rgb_up(x_down)
        return x_up

    def forward(self, x, y, x_attn_mask, y_attn_mask):
        
        orig_x = x
        orig_y = y
        # Bottleneck Fusion
        # TODO: Move this to after y_output
        x,y = self.latent_fusion(x,y)
        
        
        x = x + self.enc_1.attention(x, x_attn_mask)
        y = y + self.enc_2.attention(y, y_attn_mask)
        
        x_intermediate = self.enc_1.intermediate(x)
        y_intermediate = self.enc_2.intermediate(y)
        
        x_output = self.enc_1.output(x_intermediate, x)
        y_output = self.enc_2.output(y_intermediate, y)

        # from: https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks/blob/main/README.md
        # 
        # FFN + skip conections
        x = x + self.forward_sequence_AF(x) * self.spec_scale + x_output
        y = y + self.forward_visual_AF(x) * self.rgb_scale + y_output
        
        return x,y
        