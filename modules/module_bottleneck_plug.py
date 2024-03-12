from torch import nn
import torch
from modules.blocks import TransformerBlock

from modules.module_cross import CrossPooler

class BottleNeckMTB(nn.Module):
    # code: https://github.com/google-research/scenic/blob/main/scenic/projects/mbt/model.py#L466
    # paper: https://arxiv.org/pdf/2107.00135.pdf
    
    def __init__(self, num_layers, in_size_video=512, in_size_audio=128, out_size=512, kernel_size=19, bottleneck_dim=64, n_head=4) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        self.num_layers = num_layers
        
        self.in_size_audio = in_size_audio
        self.in_size_video = in_size_video
        self.bottleneck_dim = bottleneck_dim
        
        self.ln_v = nn.LayerNorm((in_size_video,))
        self.ln_a = nn.LayerNorm((in_size_audio,))
        self.bottlenecks = nn.ParameterList()
        
        if(self.in_size_audio != self.in_size_video):
            self.linear_down_a = nn.Linear(in_size_audio, in_size_video)
        
        self.v_enc = nn.ModuleList()
        self.a_enc = nn.ModuleList()
        for i in range(num_layers):
            self.v_enc.append(TransformerBlock(
                    in_size_video, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.1,
                    proj_pdrop=0.1,
                    # mha_win_size=-1,
                ))

            self.a_enc.append(TransformerBlock(
                    in_size_video, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=0.1,
                    proj_pdrop=0.1,
                    # mha_win_size=-1,
                ))
            self.bottlenecks.append(nn.Parameter(torch.empty(1,bottleneck_dim,in_size_video).normal_(std=0.02)))
                
        # self.out = nn.Conv1d(in_size_video+in_size_video, out_size, 1)
        
    def attention(self,q,k,v): # requires q,k,v to have same dim
        # q = q.transpose(1,2)
        
        B, N, C = q.shape
        attn = (q @ k.transpose(1,2)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N ,C)
        return x
    
    # Latent Fusion
    def fusion(self, audio_tokens, visual_tokens, latents):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((audio_tokens,visual_tokens),dim=1)
        # cross attention (AV -->> latents)
        fused_latents = self.attention(q=latents.expand(BS,-1,-1), k=concat_, v=concat_)
        # cross attention (latents -->> AV)
        audio_tokens = audio_tokens + self.attention(q=audio_tokens, k=fused_latents, v=fused_latents)
        visual_tokens = visual_tokens + self.attention(q=visual_tokens, k=fused_latents, v=fused_latents)
        return audio_tokens, visual_tokens
        
    def forward(self,out_video, out_audio, masks) -> torch.Tensor:
        
        # out_video = out_video[-1]
        # out_audio = out_audio[-1]
        BS = out_video.shape[0]
        mask = masks
        # tmp_m = torch.ones((out_video.shape[0], 1, out_video.shape[-1]+self.bottleneck_dim)).to(out_video.device)
        # tmp_m[:,:,:out_video.shape[-1]] = mask
        # mask = tmp_m
        if(self.in_size_audio != self.in_size_video):
            out_audio = self.linear_down_a(out_audio)
            
        for i in range(self.num_layers):
            f_a, f_v = self.fusion(out_audio, out_video, self.bottlenecks[i].expand(BS,-1,-1))

            out_v, _ = self.v_enc[i](f_v.transpose(1,2), mask.transpose(1,2))
            out_video = out_v.transpose(1,2) + out_video
            
            out_a, _ = self.a_enc[i](f_a.transpose(1,2), mask.transpose(1,2))
            out_audio = out_a.transpose(1,2) + out_audio
            
        out = torch.cat([out_video, out_audio], dim=1)
        return out