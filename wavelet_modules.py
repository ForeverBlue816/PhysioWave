"""
Wavelet Processing Modules
Contains learnable wavelet filters, adaptive wavelet selectors, and other components
for multi-scale time-frequency analysis of physiological signals.

This module implements the core wavelet decomposition pipeline described in the paper,
including learnable filters that adapt to signal characteristics and cross-scale
feature fusion mechanisms.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def load_wavelet_kernel(wave_name, kernel_size, mode="interp"):
    """
    Load and initialize wavelet kernel coefficients from PyWavelets.

    Args:
        wave_name: Name of the wavelet (e.g., 'db6', 'sym4', 'coif3')
        kernel_size: Target size for the convolution kernel
        mode: how a filter shorter or longer than ``kernel_size`` is fitted.

            ``'interp'`` resamples the taps linearly, rescaling to preserve
            their sum. This is what the original PhysioWave did and it is kept
            as the default so existing results reproduce, but it does not
            produce a wavelet filter bank. Stretching a filter in time
            compresses it in frequency, so the half-band cutoff moves: sym4
            fitted to 16 taps cuts at 0.203*pi instead of 0.5*pi, and the
            power-complementary condition |H_lo|^2 + |H_hi|^2 = 2 fails by
            1.999 out of a maximum possible 2.0 -- there is a band of the
            spectrum that neither the lowpass nor the highpass passes. It also
            costs orthonormality (||h_lo||^2 falls to 0.396) and every
            wavelet ends up with a different effective group delay, which
            matters because AdaptiveWaveletSelector mixes several of them.

            ``'pad'`` centres the native taps in a ``kernel_size`` window and
            pads with zeros. Zeros contribute nothing to any of the defining
            sums, so ||h||^2, sum(h), the shifted inner products and the
            frequency response are all exactly the native filter's: measured
            PR error 0.0000 and cutoff 0.500 for every orthogonal family. It
            requires ``len(native) <= kernel_size``.

    Filters whose native length already equals ``kernel_size`` are untouched by
    either mode -- db8 and sym8 are 16 taps, so at the default kernel size they
    are exact under both.

    Returns:
        dec_lo_t: Low-pass filter coefficients (approximation)
        dec_hi_t: High-pass filter coefficients (detail)
    """
    wave = pywt.Wavelet(wave_name)
    dec_lo = wave.dec_lo  # Low-pass decomposition filter
    dec_hi = wave.dec_hi  # High-pass decomposition filter

    dec_lo_t = torch.tensor(dec_lo, dtype=torch.float)
    dec_hi_t = torch.tensor(dec_hi, dtype=torch.float)

    if mode == "pad":
        n = dec_lo_t.numel()
        if n > kernel_size:
            raise ValueError(
                f"{wave_name} has {n} taps, which does not fit in kernel_size="
                f"{kernel_size}. Raise the kernel size or choose a shorter "
                f"wavelet; padding cannot shorten a filter without destroying it."
            )
        # Centred, so filters of different native lengths share a group delay.
        off = (kernel_size - n) // 2
        lo = torch.zeros(kernel_size); lo[off:off + n] = dec_lo_t
        hi = torch.zeros(kernel_size); hi[off:off + n] = dec_hi_t
        return lo, hi
    if mode != "interp":
        raise ValueError(f"mode must be 'interp' or 'pad', got {mode!r}")

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
    def __init__(self, in_ch=8, kernel_size=16, wave_init='db6', separate_per_channel=True,
                 init_mode='interp'):
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
        low_init, high_init = load_wavelet_kernel(wave_init, kernel_size, init_mode)
        
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
    def __init__(self, in_ch=8, wavelet_names=None, kernel_size=16, separate_per_channel=True,
                 init_mode='interp'):
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
            LearnableWaveFilter(in_ch, kernel_size, wname, separate_per_channel, init_mode)
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
        self.attn = nn.MultiheadAttention(embed_dims, num_heads=heads_dividing(embed_dims),
                                          batch_first=True)
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
            # Apply cross-scale attention.
            #
            # need_weights=False is load bearing, not a tidy-up. Left at its
            # default of True, nn.MultiheadAttention materialises the full
            # [B, heads, len(q), len(ks)] score matrix and takes the unfused
            # path; the weights are then discarded on the next line. That
            # matrix is quadratic in the window length, which is harmless at
            # the 256 samples of an sEMG window and fatal at the 3000 of a 30 s
            # EEG epoch: at the third level, len(q)=3000 against len(ks)=6000,
            # so B=64 with 4 heads needs 17.2 GiB for one level alone.
            #
            # Measured, B=8 and T=3000, activations saved for backward:
            #     need_weights=True    2200.6 MiB
            #     need_weights=False      3.7 MiB
            # with the outputs agreeing to 2.8e-07, i.e. fp32 reassociation.
            attn_out, _ = self.attn(q, ks, ks, need_weights=False)
            attn_out = attn_out.view(B,H,W,C).permute(0,3,1,2)
            # Add scaled attention output
            out = out + self.scale * attn_out
        return out


def heads_dividing(width, requested=4):
    """Largest head count no greater than ``requested`` that divides ``width``.

    Both attention blocks below split a width that comes from the electrode
    count, not from a power-of-two embedding dimension, so a montage the author
    did not try can make the split impossible: 14 sEMG channels against the
    hardcoded 4 heads leaves 3.5 per head and the reshape throws. Falling back
    to the nearest workable count keeps every channel layout runnable, at the
    cost of fewer heads on widths with awkward factors.
    """
    for h in range(min(requested, width), 0, -1):
        if width % h == 0:
            return h
    return 1


class MultiHeadGate(nn.Module):
    """
    Multi-Head Gating Mechanism
    
    Implements adaptive gating for soft wavelet decomposition.
    Uses multi-head attention over channel statistics to compute
    gate values that blend original and upsampled signals.
    """
    def __init__(self, in_ch, num_heads=4):
        super().__init__()
        self.num_heads = heads_dividing(in_ch, num_heads)
        if self.num_heads != num_heads:
            logging.getLogger(__name__).info(
                "MultiHeadGate: %d channels is not divisible by %d heads; using %d.",
                in_ch, num_heads, self.num_heads)
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
                 use_separate_channel=True, ffn_ratio=4., ffn_kernel_size=5, ffn_drop=0.1,
                 init_mode='interp'):
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
        self.selector = AdaptiveWaveletSelector(in_channels, wavelet_names, kernel_size,
                                                use_separate_channel, init_mode)
        
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

class ScaleFold(nn.Module):
    """Collapse the scale axis of ``Spec(X)``: ``[B, (J+1)*C, T] -> [B, C, T]``.

    ``SoftGateWaveletDecomp`` emits its bands scale-major -- band 0 for every
    channel, then band 1, and so on -- so the backbone sees ``(J+1)`` copies of
    every channel row and pays ``(J+1)^2`` in attention for them. Folding puts
    the bands back into one row per channel before patching, which is the
    synthesis half of an analysis/synthesis pair rather than a pooling trick.

    The modes form a ladder, each a single change on the one before:

    ``none``      pass through; the legacy model.
    ``mean``      plain average. Not the inverse transform in the strict sense
                  -- that is ``sum_s g_s * up(b_s)`` with a synthesis filter per
                  scale (Mallat 1989) -- but it is the unweighted reconstruction
                  and the honest baseline for everything above it.
    ``learned``   ``sum_s w[s,c] b[s,c,t]``, unconstrained. Free to grow, go
                  negative, or shrink the signal as a whole, and it carries one
                  parameter per (scale, channel) so it cannot transfer to a
                  different channel count.
    ``softmax``   the same weights through a softmax over scales. Convex, so
                  the fold is an interpolation of the bands and the output
                  magnitude stays in the bands' range.
    ``dynamic``   weights predicted per (channel, time-block) from the bands'
                  own statistics, by an MLP shared across channels. This is
                  best-basis selection (Coifman & Wickerhauser 1992) made
                  differentiable: the cost function is learned instead of being
                  Shannon entropy, and the choice is soft instead of a tree
                  search. Being shared across channels, it is also the only
                  mode with no channel-shaped parameter.

    Two options attach to ``dynamic``:

    ``synthesis_kernel``  a short depthwise filter per scale, delta-initialised
        and shared across channels -- the learnable ``g_s`` above. The bands
        have already been through soft gating and *nearest* upsampling, so they
        are not phase-aligned with each other; a scalar weight cannot fix that
        and a 3- or 5-tap filter can.
    ``shrinkage``  soft-thresholding per scale before the fold, with the
        threshold learned and the noise level estimated by MAD (Donoho 1995).
        The threshold starts at ~0.0025 sigma, i.e. off, because in sEMG the
        high-frequency detail is partly signal and hard shrinkage would remove
        it; letting the data decide is the point.

    Everything is initialised so that the module *is* ``mean`` at step 0:
    delta filters, zero thresholds, zero-initialised output layer and zero
    scale logits (so alpha is uniform), and ``gamma`` gating the deviation from
    the mean. Training therefore departs from the reconstruction rather than
    starting somewhere arbitrary.
    """

    MODES = ('none', 'mean', 'learned', 'softmax', 'dynamic')

    def __init__(self, mode='none', num_scales=4, in_channels=8, patch_len=64,
                 synthesis_kernel=0, synthesis_norm=False, share_channels=False,
                 shrinkage=False, scale_dropout=0.0,
                 gamma_init=0.1, hidden=16, eps=1e-6):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"scale_fold must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.num_scales = num_scales
        self.patch_len = int(patch_len or 0)
        self.scale_dropout = float(scale_dropout)
        self.synthesis_norm = bool(synthesis_norm)
        self.eps = eps

        # Reported back to the caller after every forward, never consumed here.
        # alpha_mean is for logging -- a fold that has collapsed onto one band
        # is indistinguishable from a healthy one by the loss alone -- and
        # reg_loss is the KL(alpha || uniform) the trainer may add.
        self.alpha_mean = None
        self.reg_loss = None
        self.alpha_std_time = None
        self.alpha_std_chan = None
        self.alpha_chan = None
        # Set True to have the last forward's full [B,C,N,S] survive in
        # alpha_blocks; scripts/alpha_probe.py is what reads it.
        self.keep_alpha = False
        self.alpha_blocks = None

        if mode in ('learned', 'softmax'):
            # Uniform average at init: for 'learned' that is literally 1/S, for
            # 'softmax' it is equal logits.
            init = 1.0 / num_scales if mode == 'learned' else 0.0
            # share_channels collapses the weight to one number per scale. The
            # trained synthesis filters turn out to differ across scales mostly
            # in their DC gain, which is exactly what this expresses -- with 4
            # parameters instead of 4*C, and without the per-channel freedom
            # that the static modes could be overfitting.
            width = 1 if share_channels else in_channels
            self.scale_weight = nn.Parameter(torch.full((num_scales, width), init))

        if mode == 'dynamic':
            self.stat_norm = nn.LayerNorm(num_scales)
            self.mlp = nn.Sequential(
                nn.Linear(3 * num_scales, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_scales),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            self.scale_logits = nn.Parameter(torch.zeros(num_scales))
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # Conditioning the bands is orthogonal to how they are then combined, so
        # it is available to every folding mode. Keeping it welded to 'dynamic'
        # would make the two inseparable: a gain from `dynamic + synthesis`
        # could be the mixture or the filters, and no run could tell you which.
        if mode != 'none' and synthesis_kernel:
            if synthesis_kernel % 2 == 0:
                raise ValueError("synthesis_kernel must be odd to stay centred")
            self.synth = nn.Conv1d(num_scales, num_scales, synthesis_kernel,
                                   padding=synthesis_kernel // 2,
                                   groups=num_scales, bias=False)
            with torch.no_grad():
                self.synth.weight.zero_()
                self.synth.weight[:, 0, synthesis_kernel // 2] = 1.0
        else:
            self.synth = None
        # softplus(-6) = 0.0025, so the threshold is off but differentiable.
        self.tau = (nn.Parameter(torch.full((num_scales,), -6.0))
                    if (shrinkage and mode != 'none') else None)

    def reset_fold_parameters(self):
        """Restore the initialisation that makes this module equal ``mean``.

        Separate from ``__init__`` because the owning model runs a generic
        ``apply(_init_weights)`` after construction, and its nn.Linear branch
        overwrites the zero output layer below. The model calls this afterwards;
        keeping the definition here means there is only one of it.

        The delta-initialised synthesis filters and the zero scale logits are
        not touched by that sweep -- Conv1d and bare Parameters are not
        nn.Linear -- but they are restored here too, so this is the whole
        statement of "starts as mean" in one place.
        """
        with torch.no_grad():
            if self.mode == 'dynamic':
                nn.init.zeros_(self.mlp[-1].weight)
                nn.init.zeros_(self.mlp[-1].bias)
                self.scale_logits.zero_()
            if self.synth is not None:
                k = self.synth.weight.shape[-1]
                self.synth.weight.zero_()
                self.synth.weight[:, 0, k // 2] = 1.0
            if self.tau is not None:
                self.tau.fill_(-6.0)

    # -- statistics -------------------------------------------------------- #
    def _block_stats(self, bands):
        """``[B,S,C,T] -> ([B,C,N,3,S], N, L)``: log-RMS, entropy, kurtosis.

        The block length is the patcher's time patch, so one weight is decided
        per token the backbone will later see. Whole-window statistics
        (``patch_len=0``) give one weight per window instead, which is the
        weaker "sample-dynamic" variant.
        """
        B, S, C, T = bands.shape
        L = self.patch_len if self.patch_len else T
        L = min(L, T)
        N = (T + L - 1) // L
        pad = N * L - T
        x = F.pad(bands, (0, pad), mode='replicate') if pad else bands
        x = x.reshape(B, S, C, N, L)

        power = x.pow(2)
        rms = power.mean(-1).clamp_min(self.eps).sqrt()
        energy = rms.log()                                        # [B,S,C,N]

        # Time-concentration of the band's energy inside the block. A burst
        # scores low, a stationary band scores near 1.
        p = power / power.sum(-1, keepdim=True).clamp_min(self.eps)
        entropy = -(p * p.clamp_min(self.eps).log()).sum(-1) / math.log(L)

        d = x - x.mean(-1, keepdim=True)
        var = d.pow(2).mean(-1).clamp_min(self.eps)
        # log1p keeps a heavy-tailed statistic from dominating the MLP input.
        kurt = torch.log1p((d.pow(4).mean(-1) / var.pow(2)).clamp_min(0.0))

        q = torch.stack([energy, entropy, kurt], dim=1)           # [B,3,S,C,N]
        q = q.permute(0, 3, 4, 1, 2).contiguous()                 # [B,C,N,3,S]
        # Normalising each statistic *across scales* is what makes this a
        # comparison between bands rather than an absolute-amplitude readout,
        # and is what lets one MLP serve every channel and every subject.
        return self.stat_norm(q), N, L

    def _alpha(self, bands, channel_scale_bias=None):
        """``[B,S,C,T] -> [B,S,C,T]`` mixing weights, summing to 1 over scales.

        ``channel_scale_bias`` is an optional ``[B,C,S]`` or ``[C,S]`` prior on
        which band a *channel* tends to want, added to the logits the block
        statistics produce. It broadcasts over the block axis: the prior says
        what this derivation is like, the statistics say what this moment is
        like, and the softmax combines them. The caller has already scaled it by
        whatever gate it owns -- this module does not know about channels.
        """
        B, S, C, T = bands.shape
        q, N, L = self._block_stats(bands)
        logits = self.mlp(q.flatten(-2)) + self.scale_logits    # [B,C,N,S]
        if channel_scale_bias is not None:
            b = channel_scale_bias
            if b.dim() == 2:                                   # [C,S] -> [B,C,S]
                b = b.unsqueeze(0).expand(B, -1, -1)
            if b.shape != (B, C, S):
                raise ValueError(
                    f"channel_scale_bias must be [C,S] or [B,C,S] with C={C} "
                    f"S={S}, got {tuple(channel_scale_bias.shape)}")
            logits = logits + b.unsqueeze(2)                    # [B,C,1,S]

        # Reported and regularised on the *undropped* weights. Dropout makes
        # alpha deliberately non-uniform, so measuring the KL after it would
        # have the penalty pushing back against the regulariser it is paired
        # with, and would report a spread the model never uses at inference.
        alpha = logits.softmax(-1)                              # [B,C,N,S]
        a = alpha.detach()
        self.alpha_mean = a.mean(dim=(0, 1, 2))
        # The mean above is the *marginal*: it averages over batch, channel and
        # time block, so a weight that swings hard from block to block and one
        # frozen at 1/S report the same [0.25, 0.25, 0.25, 0.25]. The whole
        # claim of this mode is that the weight is decided per block, and the
        # marginal is silent about it. These two are what carry it:
        #
        #   alpha_std_time  spread across time blocks, within one (sample,
        #                   channel). Zero means the fold is static, whatever
        #                   the marginal says.
        #   alpha_std_chan  spread across channels at fixed time. Distinguishes
        #                   "this recording favours a band" from "this moment
        #                   in this channel does".
        #
        # Both are per scale, mean-reduced over the axes they are not measuring,
        # and cost two reductions on a tensor already in hand.
        self.alpha_std_time = (a.std(dim=2) if N > 1 else
                               torch.zeros_like(a[:, :, 0])).mean(dim=(0, 1))
        self.alpha_std_chan = (a.std(dim=1) if C > 1 else
                               torch.zeros_like(a[:, 0])).mean(dim=(0, 1))
        # [C, S]: what each channel, on average, asks for. Kept separately from
        # alpha_mean because a channel prior's whole purpose is to make the two
        # derivations want different bands, and a marginal over channels is
        # exactly the axis that would hide it. Two reductions, no storage.
        self.alpha_chan = a.mean(dim=(0, 2))
        # KL(alpha || uniform); zero weights contribute zero, hence the clamp.
        self.reg_loss = (alpha * (alpha.clamp_min(self.eps).log()
                                  + math.log(S))).sum(-1).mean()
        # The full field, for the offline probe. Off by default: [B,C,N,S] is
        # small next to the activations but there is no reason to hold it for
        # every step of a run that will never look at it.
        self.alpha_blocks = a if self.keep_alpha else None

        if self.training and self.scale_dropout > 0:
            keep = torch.rand_like(logits) >= self.scale_dropout
            # Dropping every scale would leave nothing to fold; those rows keep
            # all of them, so the renormalisation below is always well posed.
            keep = keep | ~keep.any(-1, keepdim=True)
            alpha = logits.masked_fill(~keep, float('-inf')).softmax(-1)

        alpha = alpha.permute(0, 3, 1, 2)                        # [B,S,C,N]
        return alpha.repeat_interleave(L, dim=-1)[..., :T]

    # -- band conditioning ------------------------------------------------- #
    def _shrink(self, bands):
        B, S, C, T = bands.shape
        sigma = bands.abs().median(dim=-1, keepdim=True).values / 0.6745
        tau = F.softplus(self.tau).view(1, S, 1, 1)
        return torch.sign(bands) * (bands.abs() - tau * sigma).clamp_min(0.0)

    def _synthesise(self, bands):
        B, S, C, T = bands.shape
        x = bands.permute(0, 2, 1, 3).reshape(B * C, S, T)
        w = self.synth.weight
        if self.synthesis_norm:
            # Fix H(0) = 1 so the filter can only reshape a band, never rescale
            # it. Trained without this, the taps move mostly in gain, so the
            # two effects are not separable from a single run.
            g = w.sum(dim=-1, keepdim=True)
            w = w / torch.where(g.abs() < self.eps, torch.full_like(g, self.eps), g)
        y = F.conv1d(x, w, padding=self.synth.padding[0], groups=S)
        return y.view(B, C, S, T).permute(0, 2, 1, 3)

    # -- forward ----------------------------------------------------------- #
    def forward(self, spec, channel_scale_bias=None):
        self.alpha_mean, self.reg_loss = None, None
        self.alpha_std_time = self.alpha_std_chan = self.alpha_blocks = None
        self.alpha_chan = None
        if self.mode == 'none':
            if channel_scale_bias is not None:
                raise ValueError(
                    "channel_scale_bias needs scale_fold='dynamic'; mode is "
                    "'none', which has no mixing weights to bias.")
            return spec
        if channel_scale_bias is not None and self.mode != 'dynamic':
            raise ValueError(
                f"channel_scale_bias needs scale_fold='dynamic', got "
                f"{self.mode!r}. The static modes have one weight per (scale, "
                f"channel) already and no logits for a prior to enter.")

        B, FC, T = spec.shape
        S = self.num_scales
        if FC % S != 0:
            raise ValueError(f"{FC} rows is not a multiple of {S} bands")
        # Scale-major, so [B, S, C, T] is the correct grouping; [B, C, S, T]
        # would silently mix bands belonging to different channels.
        bands = spec.view(B, S, FC // S, T)

        proc = bands
        if self.tau is not None:
            proc = self._shrink(proc)
        if self.synth is not None:
            proc = self._synthesise(proc)

        if self.mode == 'mean':
            return proc.mean(dim=1)
        if self.mode == 'learned':
            return (proc * self.scale_weight.view(1, S, -1, 1)).sum(dim=1)
        if self.mode == 'softmax':
            w = self.scale_weight.softmax(dim=0)
            return (proc * w.view(1, S, -1, 1)).sum(dim=1)

        alpha = self._alpha(bands, channel_scale_bias)   # statistics read raw bands
        base = proc.mean(dim=1)
        return base + self.gamma * ((alpha * proc).sum(dim=1) - base)

    def extra_repr(self):
        return (f"mode={self.mode}, scales={self.num_scales}, "
                f"patch_len={self.patch_len or 'window'}, "
                f"synthesis_norm={self.synthesis_norm}, "
                f"scale_dropout={self.scale_dropout}")
