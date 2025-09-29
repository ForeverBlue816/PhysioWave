"""
Transformer Core Modules
Contains RoPE attention, Transformer blocks, Patch embedding, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        import warnings
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                     "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbed(nn.Module):
    """
    Patch Embedding Layer
    Converts 2D input to patch sequence
    """
    def __init__(self, input_channels=1, patch_size=(1,20), embed_dim=128, 
                 norm_layer=nn.LayerNorm, activation_fct=nn.GELU, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(input_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.act = activation_fct() if activation_fct else nn.Identity()
        self.flatten = flatten
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, C, H, W] - Input tensor
            
        Returns:
            [B, num_patches, embed_dim] if flatten=True
            [B, embed_dim, H', W'] if flatten=False
        """
        x = self.proj(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x); x = self.act(x)
        x = x.permute(0,3,1,2)
        if self.flatten:
            x = x.flatten(2).transpose(1,2)
        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Relative position encoding, friendly for long sequences
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim%2==0
        self.dim, self.base = dim, base
        
    def forward(self, x, seq_len):
        """
        Args:
            x: [B, H, N, D] - Input tensor
            seq_len: Sequence length
            
        Returns:
            [B, H, N, D] - Tensor with RoPE applied
        """
        B,H,N,D = x.shape
        pos = torch.arange(N,device=x.device)
        inv = 1.0/(self.base**(torch.arange(0,D,2,device=x.device)/D))
        freqs = pos[:,None]*inv[None,:]
        cosv = freqs.cos().unsqueeze(1).unsqueeze(1)
        sinv = freqs.sin().unsqueeze(1).unsqueeze(1)
        xr = x.permute(2,0,1,3).reshape(N,B,H,D//2,2)
        e,o = xr[...,0], xr[...,1]
        eo = e*cosv - o*sinv
        oo = e*sinv + o*cosv
        xo = torch.stack([eo,oo],dim=-1).reshape(N,B,H,D)
        return xo.permute(1,2,0,3)


class RoPEAttention(nn.Module):
    """
    Multi-head Attention with RoPE
    Better position modeling compared to standard attention
    """
    def __init__(self, dim, num_heads=8, rope_dim=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim%num_heads==0
        self.num_heads, self.dim = num_heads, dim
        self.hd = dim//num_heads; self.scale = self.hd**-0.5
        self.qkv = nn.Linear(dim,dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim); self.proj_drop = nn.Dropout(proj_drop)
        self.rope_dim = rope_dim or self.hd
        self.rope_q = RotaryEmbedding(self.rope_dim)
        self.rope_k = RotaryEmbedding(self.rope_dim)
        
    def forward(self,x,mask=None):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input sequence
            mask: Attention mask
            
        Returns:
            [B, N, C] - Output sequence
        """
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.hd).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        # Apply RoPE
        if self.rope_dim>0 and self.rope_dim<=self.hd:
            qr,rem = q[...,:self.rope_dim],q[...,self.rope_dim:]
            kr,_   = k[...,:self.rope_dim],k[...,self.rope_dim:]
            qr = self.rope_q(qr,seq_len=N); kr = self.rope_k(kr,seq_len=N)
            q = torch.cat([qr,rem],dim=-1); k = torch.cat([kr,_],dim=-1)
        
        # Attention computation
        attn = (q @ k.transpose(-2,-1))*self.scale
        if mask is not None: attn = attn+mask
        attn = attn.softmax(-1); attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(2,1).reshape(B,N,C)
        x_out = self.proj(x_out); x_out = self.proj_drop(x_out)
        return x_out


class TransformerBlock(nn.Module):
    """
    Transformer Block
    Pre-Norm structure, contains RoPE attention and MLP
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1, rope_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads, rope_dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,int(dim*mlp_ratio)), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio),dim),
            nn.Dropout(dropout)
        )
        
    def forward(self,x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input sequence
            
        Returns:
            [B, N, C] - Output sequence
        """
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class PositionEmbedding(nn.Module):
    """
    Position Encoding Module
    Supports 1D and 2D sinusoidal position encoding
    """
    def __init__(self, embed_dim, max_len=5000, pos_type='1d'):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_type = pos_type
        
        if pos_type == '1d':
            self.pos_embed = self._create_1d_pos_embed(max_len, embed_dim)
        else:
            self.pos_embed = None  # 2D position encoding generated dynamically
    
    def _create_1d_pos_embed(self, max_len, embed_dim):
        """Create 1D sinusoidal position encoding"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """Generate 1D sinusoidal position encoding"""
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega

        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return emb

    def get_2d_sincos_pos_embed(self, embed_dim, freq_size, time_size):
        """Generate 2D position encoding"""
        assert embed_dim % 2 == 0
        
        freq_embed_dim = embed_dim // 2
        time_embed_dim = embed_dim - freq_embed_dim
        
        freq_pos = torch.arange(freq_size, dtype=torch.float32)
        freq_embed = self.get_1d_sincos_pos_embed_from_grid(freq_embed_dim, freq_pos)
        
        time_pos = torch.arange(time_size, dtype=torch.float32)
        time_embed = self.get_1d_sincos_pos_embed_from_grid(time_embed_dim, time_pos)
        
        freq_embed_2d = freq_embed.repeat_interleave(time_size, dim=0)
        time_embed_2d = time_embed.repeat(freq_size, 1)
        
        pos_embed_2d = torch.cat([freq_embed_2d, time_embed_2d], dim=1)
        return pos_embed_2d
    
    def forward(self, tokens, freq_size=None, time_size=None):
        """
        Add position encoding
        
        Args:
            tokens: [B, L, D] - Input tokens
            freq_size: Frequency dimension size (for 2D position encoding)
            time_size: Time dimension size (for 2D position encoding)
            
        Returns:
            [B, L, D] - Tokens with position encoding added
        """
        B, L, D = tokens.shape
        
        if self.pos_type == '1d':
            # 1D position encoding
            if self.pos_embed is not None and L <= self.pos_embed.shape[1]:
                pos_embed = self.pos_embed[:, :L, :]
            else:
                # Generate dynamically
                pos_embed = self.get_1d_sincos_pos_embed_from_grid(D, torch.arange(L, dtype=torch.float32))
                pos_embed = pos_embed.unsqueeze(0)
            
            return tokens + pos_embed.to(tokens.device)
        
        elif self.pos_type == '2d':
            # 2D position encoding
            if freq_size is None or time_size is None:
                # Fall back to 1D
                pos_embed = self.get_1d_sincos_pos_embed_from_grid(D, torch.arange(L, dtype=torch.float32))
                pos_embed = pos_embed.unsqueeze(0)
            else:
                expected_patches = freq_size * time_size
                if L != expected_patches:
                    # Patch count mismatch, use 1D
                    pos_embed = self.get_1d_sincos_pos_embed_from_grid(D, torch.arange(L, dtype=torch.float32))
                    pos_embed = pos_embed.unsqueeze(0)
                else:
                    # 2D position encoding
                    pos_embed = self.get_2d_sincos_pos_embed(D, freq_size, time_size)
                    pos_embed = pos_embed.unsqueeze(0)
            
            return tokens + pos_embed.to(tokens.device)
        
        else:
            return tokens


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    Composed of multiple TransformerBlocks
    """
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1, rope_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, rope_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, N, C] - Input sequence
            
        Returns:
            [B, N, C] - Encoded sequence
        """
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)