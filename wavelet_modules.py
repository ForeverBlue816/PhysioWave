"""
Wavelet Processing Modules
Contains learnable wavelet filters, adaptive wavelet selectors, and other components
for multi-scale time-frequency analysis of physiological signals.

This module implements the core wavelet decomposition pipeline described in the paper,
including learnable filters that adapt to signal characteristics and cross-scale
feature fusion mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def load_wavelet_kernel(wave_name, kernel_size):
    """
    Load and initialize wavelet kernel coefficients from PyWavelets.
    
    This function extracts the low-pass (dec_lo) and high-pass (dec_hi) 
    decomposition filters from a specified wavelet family and resamples 
    them to the desired kernel size while preserving their mathematical properties.
    
    Args:
        wave_name: Name of the wavelet (e.g., 'db6', 'sym4', 'coif3')
        kernel_size: Target size for the convolution kernel
        
    Returns:
        dec_lo_t: Low-pass filter coefficients (approximation)
        dec_hi_t: High-pass filter coefficients (detail)
    """
    wave = pywt.Wavelet(wave_name)
    dec_lo = wave.dec_lo  # Low-pass decomposition filter
    dec_hi = wave.dec_hi  # High-pass decomposition filter

    dec_lo_t = torch.tensor(dec_lo, dtype=torch.float)
    dec_hi_t = torch.tensor(dec_hi, dtype=torch.float)

    # Resample filters to match desired kernel size using interpolation
    if len(dec_lo_t) != kernel_size:
        dec_lo_t = dec_lo_t.view(1,1,-1)
        dec_lo_t = F.interpolate(dec_lo_t, size=kernel_size, mode='linear', align_corners=True).squeeze()
        # Preserve L1 norm after interpolation to maintain filter properties
        dec_lo_t = dec_lo_t * (torch.sum(torch.tensor(dec_lo)) / torch.sum(dec_lo_t))
    if len(dec_hi_t) != kernel_size:
        dec_hi_t = dec_hi_t.view(1,1,-1)
        dec_hi_t = F.interpolate(dec_hi_t, size=kernel_size, mode='linear', align_corners=True).squeeze()
        # Preserve L1 norm for high-pass filter
        dec_hi_t = dec_hi_t * (torch.sum(torch.abs(torch.tensor(dec_hi))) / torch.sum(torch.abs(dec_hi_t)))
    return dec_lo_t, dec_hi_t


class LearnableWaveFilter(nn.Module):
    """
    Learnable Wavelet Filter
    
    Implements adaptive wavelet decomposition where filter coefficients are initialized
    from standard wavelets but can be fine-tuned during training to better match
    the signal characteristics. This allows the model to learn optimal frequency
    decomposition for specific physiological signals.
    """
    def __init__(self, in_ch=8, kernel_size=16, wave_init='db6', separate_per_channel=True):
        """
        Args:
            in_ch: Number of input channels (e.g., 8 for EMG, 12 for ECG)
            kernel_size: Size of the wavelet kernel
            wave_init: Initial wavelet type for filter initialization
            separate_per_channel: If True, learn separate filters for each channel
        """
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.wave_init = wave_init
        self.separate_per_channel = separate_per_channel

        # Initialize filters from standard wavelet
        low_init, high_init = load_wavelet_kernel(wave_init, kernel_size)
        
        # Groups parameter determines if filters are shared across channels
        groups = in_ch if separate_per_channel else 1

        # Create depthwise or regular convolution layers for wavelet decomposition
        self.low_filter = nn.Conv1d(in_ch, in_ch, kernel_size, padding=kernel_size//2, groups=groups, bias=False)
        self.high_filter = nn.Conv1d(in_ch, in_ch, kernel_size, padding=kernel_size//2, groups=groups, bias=False)
        
        # Initialize convolution weights with wavelet coefficients
        with torch.no_grad():
            if separate_per_channel:
                # Each channel gets its own learnable filter
                for c in range(in_ch):
                    self.low_filter.weight[c,0,:] = low_init
                    self.high_filter.weight[c,0,:] = high_init
            else:
                # All channels share the same filter
                self.low_filter.weight[0,0,:] = low_init
                self.high_filter.weight[0,0,:] = high_init

    def forward(self, x):
        """
        Perform one level of wavelet decomposition.
        
        Args:
            x: [B, C, T] - Input signal
            
        Returns:
            approx: [B, C, T//2] - Low-frequency approximation coefficients
            detail: [B, C, T//2] - High-frequency detail coefficients
        """
        approx = self.low_filter(x)
        detail = self.high_filter(x)
        # Downsample by factor of 2 (standard wavelet decimation)
        return approx[..., ::2], detail[..., ::2]


class AdaptiveWaveletSelector(nn.Module):
    """
    Adaptive Wavelet Selector
    
    Implements the wavelet selection mechanism described in the paper.
    This module maintains multiple wavelet bases and learns to select
    the optimal combination based on input signal characteristics.
    The selection is done via attention mechanism over signal statistics.
    """
    def __init__(self, in_ch=8, wavelet_names=None, kernel_size=16, separate_per_channel=True):
        """
        Args:
            in_ch: Number of input channels
            wavelet_names: List of wavelet families to choose from
            kernel_size: Size of wavelet kernels
            separate_per_channel: Whether to use channel-wise processing
        """
        super().__init__()
        if wavelet_names is None:
            # Default wavelet bases as mentioned in the paper
            wavelet_names = ['db4','db6','sym4','coif3']
        
        # Create a bank of learnable wavelet filters
        self.wavelet_filters = nn.ModuleList([
            LearnableWaveFilter(in_ch, kernel_size, wname, separate_per_channel)
            for wname in wavelet_names
        ])
        
        # Selection network: uses global average pooling followed by MLP
        # to compute selection weights for each wavelet basis
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_ch, 128),    # Hidden layer for feature extraction
            nn.ReLU(),
            nn.Linear(128, len(wavelet_names)),  # Output selection scores
            nn.Softmax(dim=1)         # Normalize to probability distribution
        )

    def forward(self, x):
        """
        Adaptively select and apply wavelet decomposition.
        
        Args:
            x: [B, C, T] - Input signal
            
        Returns:
            approx: [B, C, T//2] - Weighted combination of approximations
            detail: [B, C, T//2] - Weighted combination of details
        """
        B, C, T = x.shape
        
        # Compute selection weights based on input characteristics
        weights = self.selector(x)  # [B, num_wavelets]
        
        approx_list, detail_list = [], []
        # Apply each wavelet and weight by selection score
        for i, filt in enumerate(self.wavelet_filters):
            a, d = filt(x)
            w = weights[:, i].view(B,1,1)  # Reshape for broadcasting
            approx_list.append(a * w)
            detail_list.append(d * w)
        
        # Sum weighted decompositions
        return sum(approx_list), sum(detail_list)


class ElementScale(nn.Module):
    """
    Element-wise scaling layer with learnable parameters.
    Used for residual connections in cross-scale fusion.
    """
    def __init__(self, shape, init_value=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(*shape)*init_value)
    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """
    Channel Aggregation Feed-Forward Network
    
    Implements the channel aggregation mechanism for processing
    multi-channel physiological signals. Uses depthwise convolutions
    for efficient channel-wise processing.
    """
    def __init__(self, embed_dims, ffn_ratio=4., kernel_size=3, dropout=0.1):
        super().__init__()
        hid = int(embed_dims*ffn_ratio)
        
        # 1x1 convolution for channel mixing
        self.fc1 = nn.Conv2d(embed_dims, hid, 1)
        # Depthwise convolution for spatial processing
        self.dw = nn.Conv2d(hid, hid, kernel_size, padding=kernel_size//2, groups=hid)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid, embed_dims,1)
        self.drop = nn.Dropout(dropout)
        
        # Decomposition branch for removing DC component
        self.decomp = nn.Conv2d(hid,1,1)
        self.decomp_act = nn.GELU()
        # Learnable residual scaling
        self.sigma = ElementScale([1,hid,1,1])
        
    def forward(self, x):
        """
        Process features with channel aggregation and decomposition.
        
        Args:
            x: [B, C, H, W] - Input feature map
            
        Returns:
            [B, C, H, W] - Processed features
        """
        out = self.fc1(x)
        out = self.dw(out)
        out = self.act(out)
        out = self.drop(out)
        
        # Decompose and subtract mean component
        t = self.decomp_act(self.decomp(out))
        out = out - t
        out = out + self.sigma(out)  # Scaled residual
        
        out = self.fc2(out)
        out = self.drop(out)
        return out


class CrossScaleCAFFN(nn.Module):
    """
    Cross-Scale Channel Aggregation FFN
    
    Extends CAFFN with cross-scale attention mechanism.
    This allows features at different decomposition levels
    to exchange information, as described in Section 2.1 of the paper.
    """
    def __init__(self, embed_dims, ffn_ratio=4., kernel_size=3, dropout=0.1):
        super().__init__()
        self.base = ChannelAggregationFFN(embed_dims, ffn_ratio, kernel_size, dropout)
        # Multi-head attention for cross-scale feature fusion
        self.attn = nn.MultiheadAttention(embed_dims, num_heads=4, batch_first=True)
        # Learnable scaling factor for attention output
        self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, prev_feats=[]):
        """
        Process with cross-scale attention.
        
        Args:
            x: [B, C, H, W] - Current scale features
            prev_feats: List of features from previous scales
            
        Returns:
            [B, C, H, W] - Enhanced features with cross-scale information
        """
        out = self.base(x)
        
        if prev_feats:
            B,C,H,W = out.shape
            # Reshape for attention: [B, H*W, C]
            q = out.permute(0,2,3,1).reshape(B,-1,C)
            # Concatenate all previous scale features as keys/values
            ks = torch.cat([pf.permute(0,2,3,1).reshape(B,-1,C) for pf in prev_feats],dim=1)
            # Apply cross-scale attention
            attn_out,_ = self.attn(q,ks,ks)
            attn_out = attn_out.view(B,H,W,C).permute(0,3,1,2)
            # Add scaled attention output
            out = out + self.scale * attn_out
        return out


class MultiHeadGate(nn.Module):
    """
    Multi-Head Gating Mechanism
    
    Implements adaptive gating for soft wavelet decomposition.
    Uses multi-head attention over channel statistics to compute
    gate values that blend original and upsampled signals.
    """
    def __init__(self, in_ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(in_ch, in_ch)
        self.k = nn.Linear(in_ch, in_ch)
        self.v = nn.Linear(in_ch, in_ch)
        
    def forward(self, x):
        """
        Compute adaptive gate values.
        
        Args:
            x: [B, C, T] - Input signal
            
        Returns:
            [B, C, 1] - Gate values between 0 and 1
        """
        B,C,T = x.shape
        x_pool = x.mean(-1)  # Global average pooling
        
        # Multi-head attention computation
        Q = self.q(x_pool).view(B,self.num_heads,-1)
        K = self.k(x_pool).view(B,self.num_heads,-1)
        V = self.v(x_pool).view(B,self.num_heads,-1)
        
        hdim = C//self.num_heads
        attn = F.softmax((Q @ K.transpose(1,2))/math.sqrt(hdim), dim=-1)
        out = (attn @ V).view(B,-1)
        
        return torch.sigmoid(out).unsqueeze(-1)  # Sigmoid for gating


class SoftGateWaveletDecomp(nn.Module):
    """
    Soft-Gated Wavelet Decomposition Module
    
    Main wavelet decomposition module implementing the complete pipeline
    described in Section 2.1 of the paper. Combines:
    1. Multi-level wavelet decomposition with adaptive selection
    2. Soft gating for smooth information flow between scales
    3. Cross-scale feature fusion via attention
    
    This module produces the multi-scale frequency-band representation
    Spec(X) that serves as input to the Transformer encoder.
    """
    def __init__(self, in_channels=8, max_level=3, kernel_size=16, wavelet_names=None,
                 use_separate_channel=True, ffn_ratio=4., ffn_kernel_size=5, ffn_drop=0.1):
        """
        Args:
            in_channels: Number of input channels (e.g., 8 for EMG, 12 for ECG)
            max_level: Number of decomposition levels (L in the paper)
            kernel_size: Size of wavelet kernels
            wavelet_names: List of wavelet families to use
            use_separate_channel: Channel-wise processing flag
            ffn_ratio: FFN expansion ratio
            ffn_kernel_size: Kernel size for FFN convolutions
            ffn_drop: Dropout rate in FFN
        """
        super().__init__()
        self.max_level = max_level
        
        # Adaptive wavelet selector (Section 2.1 - Adaptive Wavelet Selector)
        self.selector = AdaptiveWaveletSelector(in_channels, wavelet_names, kernel_size, use_separate_channel)
        
        # Multi-head gating for soft decomposition
        self.gate = MultiHeadGate(in_channels)
        
        # Cross-scale CAFFN modules for each decomposition level
        self.sub_ffn = nn.ModuleList([
            CrossScaleCAFFN(2*in_channels, ffn_ratio, ffn_kernel_size, ffn_drop)
            for _ in range(max_level)
        ])
        
        # Learnable residual scales for each level
        self.res_scales = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(max_level)])
        
    def forward(self, x):
        """
        Perform multi-level soft-gated wavelet decomposition.
        
        This implements the complete wavelet decomposition pipeline:
        1. Iterative decomposition across L levels
        2. Soft gating between original and upsampled signals
        3. Cross-scale feature fusion at each level
        4. Concatenation to form Spec(X)
        
        Args:
            x: [B, C, T] - Input time series signal
            
        Returns:
            [B, (max_level+1)*C, T] - Multi-scale wavelet decomposition
                Contains L detail bands and 1 approximation band
        """
        B,C,T = x.shape
        approx, detail_accum = x, torch.zeros_like(x)
        prev_feats, bands = [], []
        
        for i in range(self.max_level):
            # Wavelet decomposition at current level
            a2, d2 = self.selector(approx)
            
            # Upsample to original size for soft gating
            up_a = F.interpolate(a2.unsqueeze(1), size=(C,T), mode='nearest').squeeze(1)
            up_d = F.interpolate(d2.unsqueeze(1), size=(C,T), mode='nearest').squeeze(1)
            
            # Soft gating: blend original and upsampled signals
            # This prevents aliasing and preserves important details
            g = self.gate(approx)
            new_a = g*approx + (1-g)*up_a  # Gated approximation
            new_d = g*detail_accum + (1-g)*up_d  # Gated details
            
            # Cross-scale feature fusion
            # Concatenate approximation and detail for processing
            sb = torch.cat([new_a,new_d],dim=1).unsqueeze(2)  # [B, 2C, 1, T]
            # Apply CAFFN with cross-scale attention
            out2 = sb + self.res_scales[i] * self.sub_ffn[i](sb, prev_feats)
            out2 = out2.squeeze(2)
            
            # Update for next iteration
            approx, detail_accum = out2[:,:C], out2[:,C:]
            prev_feats.append(sb)
            bands.append(detail_accum)
        
        # Add final approximation band
        bands.append(approx)
        
        # Concatenate all frequency bands to form Spec(X)
        return torch.cat(bands,dim=1)  # [B, (max_level+1)*C, T]