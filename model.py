"""
BERT-style Wavelet Transformer Main Model
Refactored version with modular design and clear component separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_modules import ScaleFold, SoftGateWaveletDecomp
from transformer_modules import PatchEmbed, PositionEmbedding, TransformerEncoder
from head_modules import ClassificationHead, ReconstructionHead, RegressionHead, LinearHead


class BERTWaveletTransformer(nn.Module):
    """
    BERT-style Wavelet Transformer Main Model
    
    Modular Design:
    1. Wavelet Decomposition Module: SoftGateWaveletDecomp
    2. Patch Embedding: PatchEmbed  
    3. Position Encoding: PositionEmbedding
    4. Transformer Encoder: TransformerEncoder
    5. Task Heads: Various Head modules
    """
    def __init__(self,
                 # Wavelet parameters
                 in_channels=8, 
                 max_level=3,
                 wave_kernel_size=16,
                 wavelet_names=None,
                 use_separate_channel=True,
                 wave_init_mode='interp',   # see wavelet_modules.load_wavelet_kernel
                 # Patch embedding parameters
                 patch_size=(1,20),
                 embed_dim=128,
                 # Transformer parameters
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 rope_dim=None,
                 # Transformer block variants. The defaults reproduce the
                 # original block exactly, so each is an ablation row.
                 norm='layernorm',      # 'layernorm' | 'rmsnorm'
                 ffn='mlp',             # 'mlp' | 'swiglu'
                 qk_norm=False,
                 # Fold the scale axis out of Spec(X) before patching, so the
                 # backbone sees C*S tokens instead of (J+1)*C*S. Everything
                 # else -- the per-channel wavelet filters, the basis selector,
                 # the 2-D position embedding, the full attention, the pooling
                 # and the head -- is untouched, which is what makes this a
                 # single-variable change against the legacy numbers.
                 # Channel metadata. Off by default, so a model built without
                 # these arguments is the one that existed before them.
                 channel_encoding='none',   # none|id|signed|hybrid
                 channel_injection='none',  # none|token|fold|dual
                 channel_embed_dim=64,
                 channel_fold_gate_init=0.0,
                 channel_token_gate_init=0.0,
                 channel_vocab_size=None,
                 scale_fold='none',     # see wavelet_modules.ScaleFold
                 fold_patch_len=None,   # None -> patch_size[1]; 0 -> whole window
                 fold_synthesis=0,      # odd kernel for a per-scale synthesis filter
                 fold_synthesis_norm=False,   # fix its DC gain, isolating shape
                 fold_share_channels=False,   # static weights become [S, 1]
                 fold_shrinkage=False,  # learned soft-threshold before folding
                 fold_scale_dropout=0.0,
                 fold_gamma=0.1,
                 # Position encoding parameters
                 use_pos_embed=True,
                 pos_embed_type='2d',
                 # Masking parameters
                 masking_strategy='frequency_guided',
                 importance_ratio=0.6,
                 mask_ratio=0.15,
                 # Task head parameters
                 task_type=None,  # 'classification', 'regression', 'pretrain'
                 num_classes=None,
                 output_dim=None,
                 head_config=None,
                 pooling='mean'):
        super().__init__()
        
        # Save configuration
        self.in_channels = in_channels
        self.max_level = max_level
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_pos_embed = use_pos_embed
        self.pos_embed_type = pos_embed_type
        self.masking_strategy = masking_strategy
        self.importance_ratio = importance_ratio
        self.mask_ratio = mask_ratio
        self.task_type = task_type
        
        # Calculate patch dimension
        self.patch_dim = patch_size[0] * patch_size[1]
        
        # 1. Wavelet decomposition module
        self.wavelet_decomp = SoftGateWaveletDecomp(
            in_channels=in_channels,
            max_level=max_level,
            kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
            init_mode=wave_init_mode,
            ffn_ratio=4.0,
            ffn_kernel_size=3,
            ffn_drop=0.1
        )
        
        # 1b. Scale folding. Spec(X) is [B, (J+1)*C, T] laid out scale-major:
        # J detail bands then the approximation, each [B, C, T] (see
        # wavelet_modules.SoftGateWaveletDecomp.forward). Reducing over the
        # scale axis is a learned generalisation of the inverse transform --
        # the bands are recombined into one row per channel rather than each
        # becoming its own row of tokens.
        # The dynamic modes decide one weight per time block, and the block is
        # the patcher's time patch by default so that one weight backs one
        # token. Detaching the two would give the backbone tokens whose scale
        # mix changes partway through.
        self.scale_fold = scale_fold
        self.fold = ScaleFold(
            mode=scale_fold,
            num_scales=max_level + 1,
            in_channels=in_channels,
            patch_len=patch_size[1] if fold_patch_len is None else fold_patch_len,
            synthesis_kernel=fold_synthesis,
            synthesis_norm=fold_synthesis_norm,
            share_channels=fold_share_channels,
            shrinkage=fold_shrinkage,
            scale_dropout=fold_scale_dropout,
            gamma_init=fold_gamma,
        )

        # 2. Patch embedding module
        self.patch_embed = PatchEmbed(
            input_channels=1,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # 3. Position encoding module
        if use_pos_embed:
            self.pos_embed = PositionEmbedding(
                embed_dim=embed_dim,
                pos_type=pos_embed_type
            )
        else:
            self.pos_embed = None
        
        # 4. MASK token (for pretraining)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 5. Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            rope_dim=rope_dim,
            norm=norm,
            ffn=ffn,
            qk_norm=qk_norm
        )
        
        # 6. Task head modules
        self.task_heads = nn.ModuleDict()
        
        # Pretraining reconstruction head
        self.task_heads['pretrain'] = ReconstructionHead(
            embed_dim=embed_dim,
            patch_dim=self.patch_dim,
            hidden_dims=[embed_dim],
            dropout=dropout
        )
        
        # Add corresponding head based on task type
        if task_type == 'classification' and num_classes is not None:
            head_config = head_config or {}
            self.task_heads['classification'] = ClassificationHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                hidden_dims=head_config.get('hidden_dims'),
                dropout=head_config.get('dropout', dropout),
                pooling=head_config.get('pooling', pooling)
            )
        
        elif task_type == 'regression' and output_dim is not None:
            head_config = head_config or {}
            self.task_heads['regression'] = RegressionHead(
                embed_dim=embed_dim,
                output_dim=output_dim,
                hidden_dims=head_config.get('hidden_dims'),
                dropout=head_config.get('dropout', dropout),
                pooling=head_config.get('pooling', pooling),
                output_activation=head_config.get('output_activation')
            )
        
        elif task_type == 'linear' and output_dim is not None:
            head_config = head_config or {}
            self.task_heads['linear'] = LinearHead(
                embed_dim=embed_dim,
                output_dim=output_dim,
                pooling=head_config.get('pooling', pooling),
                use_norm=head_config.get('use_norm', False)
            )
        
        # Weight initialization
        self.apply(self._init_weights)
        # apply() walks every submodule, so the generic nn.Linear branch above
        # overwrites the zero-initialised output layer of ScaleFold's dynamic
        # MLP with trunc_normal_(std=0.02). That silently breaks the property
        # the mode is built on: with a non-zero output layer the mixing weights
        # are already a function of the band statistics at step 0, so the fold
        # does not start as the plain mean and the ladder
        # none -> mean -> ... -> dynamic stops being single-variable.
        #
        # Measured before this line existed: alpha spanned [0.2487, 0.2512] at
        # initialisation and its spread across time blocks was 4e-4 to 6e-4 --
        # small, but the same order as what a trained fold has to be
        # distinguished from. Restoring the module's own initialisation is left
        # to the module, so there is one definition of it.
        for m in self.modules():
            if isinstance(m, ScaleFold):
                m.reset_fold_parameters()

        # Everything channel-related is constructed HERE, after the legacy
        # modules have been built and after apply() has swept them. Two reasons,
        # and both are load-bearing for the ablation:
        #
        #   * Constructing a module draws from the global RNG. Building these
        #     earlier would shift every legacy draw, and the variants would no
        #     longer share an initialisation to be compared at.
        #   * apply(_init_weights) rewrites every nn.Linear it walks. It already
        #     did that to ScaleFold's MLP; these would be next.
        #
        # A test pins the first point by hashing the legacy parameters across
        # all six variants at one seed.
        self._build_channel_modules(
            channel_encoding, channel_injection, channel_embed_dim,
            channel_fold_gate_init, channel_token_gate_init, channel_vocab_size,
            embed_dim, max_level, scale_fold)

    def _build_channel_modules(self, encoding, injection, embed_dim_c,
                               fold_gate_init, token_gate_init, vocab_size,
                               embed_dim, max_level, scale_fold):
        from channel_embedding import ChannelEncoder

        if encoding not in ChannelEncoder.MODES:
            raise ValueError(f"channel_encoding must be one of "
                             f"{ChannelEncoder.MODES}, got {encoding!r}")
        if injection not in ('none', 'token', 'fold', 'dual'):
            raise ValueError(f"channel_injection must be none|token|fold|dual, "
                             f"got {injection!r}")
        if (encoding == 'none') != (injection == 'none'):
            raise ValueError(
                f"channel_encoding={encoding!r} and channel_injection="
                f"{injection!r} disagree: a code with nowhere to go, or an "
                f"injection with no code. Both 'none', or neither.")
        if injection in ('fold', 'dual') and scale_fold != 'dynamic':
            raise ValueError(
                f"channel_injection={injection!r} biases the dynamic fold's "
                f"logits, and scale_fold is {scale_fold!r}. The static modes "
                f"have no logits for a prior to enter.")

        self.channel_encoding = encoding
        self.channel_injection = injection
        self.channel_embed_dim = embed_dim_c
        if encoding == 'none':
            self.channel_encoder = None
            return

        self.channel_encoder = ChannelEncoder(encoding, embed_dim_c,
                                              vocab_size=vocab_size)
        num_scales = max_level + 1
        # The gates start at zero and the projections do not. A zero gate makes
        # the branch's output zero, so the backbone is untouched at step 0; a
        # zero projection as well would put a zero on both sides of the product
        # and neither would ever receive gradient. This way the gate has
        # gradient immediately, and once it moves the projection does too.
        if injection in ('fold', 'dual'):
            self.channel_to_scale = nn.Linear(embed_dim_c, num_scales)
            self.channel_fold_gate = nn.Parameter(torch.tensor(float(fold_gate_init)))
        if injection in ('token', 'dual'):
            self.channel_to_token = nn.Linear(embed_dim_c, embed_dim)
            self.channel_token_gate = nn.Parameter(torch.tensor(float(token_gate_init)))
        self.reset_channel_parameters()

    def reset_channel_parameters(self):
        """Initialise the channel modules. Never called by apply()."""
        if self.channel_encoder is None:
            return
        self.channel_encoder.reset_channel_parameters()
        with torch.no_grad():
            for name in ('channel_to_scale', 'channel_to_token'):
                proj = getattr(self, name, None)
                if proj is not None:
                    nn.init.normal_(proj.weight, std=0.02)
                    nn.init.zeros_(proj.bias)

    def channel_parameter_names(self):
        """Every parameter this feature added, for whitelists and optimiser groups."""
        if self.channel_encoder is None:
            return ()
        names = [n for n, _ in self.named_parameters()
                 if n.startswith(('channel_encoder.', 'channel_to_scale.',
                                  'channel_to_token.'))
                 or n in ('channel_fold_gate', 'channel_token_gate')]
        return tuple(names)

    def channel_gate_values(self):
        """``(fold gate, token gate)`` as floats, or ``None`` where absent."""
        f = getattr(self, 'channel_fold_gate', None)
        t = getattr(self, 'channel_token_gate', None)
        return (None if f is None else float(f.detach()),
                None if t is None else float(t.detach()))

    def _channel_code(self, channel_meta):
        """``[C, Dc]`` or ``None``. Raises when the code is needed and absent."""
        if self.channel_encoder is None:
            return None
        return self.channel_encoder(channel_meta)

    def _init_weights(self, m):
        """Weight initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def initialize_weights(self):
        """Initialize special weights"""
        nn.init.normal_(self.mask_token, std=.02)
    
    def add_task_head(self, task_name, head_module):
        """Dynamically add task head"""
        self.task_heads[task_name] = head_module
    
    @staticmethod
    def _mask_budget(L, mask_ratio, valid_token_mask):
        """How many tokens to mask, counted over the tokens eligible to be masked.

        With a padded montage the eligible count is smaller than L, and taking
        the budget from L would push the ratio *over the valid tokens* well past
        what was asked for -- TDBRAIN's 26 real channels in 32 slots turn a
        requested 0.5 into 0.615. Counted over the valid tokens the requested
        ratio means the same thing on every route.

        The count is the per-sample minimum so that one budget serves the batch;
        within a route every sample carries the same montage, so the minimum is
        also the common value and nothing is lost to it.
        """
        if valid_token_mask is None:
            return int(L * mask_ratio)
        eligible = int(valid_token_mask.sum(dim=1).min().item())
        return max(1, int(eligible * mask_ratio)) if eligible > 0 else 0

    def frequency_guided_masking(self, tokens, mask_ratio, importance_ratio=0.6,
                                 valid_token_mask=None, generator=None):
        """Frequency-domain importance-based masking strategy.

        ``valid_token_mask`` is ``[B, L]`` or ``None``. Where it is False the
        token belongs to a padded channel slot -- a montage of 26 electrodes
        carried in 32 rows -- and must never be selected: it is not a
        measurement, so reconstructing it would be scoring the model on zeros.
        Excluded by driving the score to -inf rather than by shrinking the
        sequence, so the token positions the encoder sees do not shift.

        ``generator`` makes the random half of the score reproducible, which is
        what lets a validation mask stay fixed across epochs.
        """
        B, L, D = tokens.shape
        num_mask = self._mask_budget(L, mask_ratio, valid_token_mask)
        if num_mask == 0:
            return torch.zeros(B, L, device=tokens.device, dtype=torch.bool)

        # Calculate frequency domain importance
        tokens_reshaped = tokens.permute(0, 2, 1)
        tokens_fft = torch.abs(torch.fft.rfft(tokens_reshaped, dim=2))
        importance_scores = torch.sum(tokens_fft, dim=1)

        # Interpolate to original length
        importance_full = F.interpolate(
            importance_scores.unsqueeze(1), size=L,
            mode='linear', align_corners=True
        ).squeeze(1)

        # Mix randomness and importance
        if generator is None:
            random_noise = torch.rand(B, L, device=tokens.device)
        else:
            random_noise = torch.rand(B, L, device=generator.device,
                                      generator=generator).to(tokens.device)
        combined_scores = importance_ratio * importance_full + (1 - importance_ratio) * random_noise

        if valid_token_mask is not None:
            combined_scores = combined_scores.masked_fill(~valid_token_mask,
                                                          float('-inf'))

        # Select positions with highest scores for masking
        _, mask_indices = torch.topk(combined_scores, num_mask, dim=1)

        # Create mask
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, mask_indices, True)

        return mask

    def random_masking(self, tokens, mask_ratio, valid_token_mask=None,
                       generator=None):
        """Random masking strategy. ``valid_token_mask`` as above."""
        B, L, D = tokens.shape
        num_mask = self._mask_budget(L, mask_ratio, valid_token_mask)
        if num_mask == 0:
            return torch.zeros(B, L, device=tokens.device, dtype=torch.bool)

        if valid_token_mask is None:
            # Unchanged from the original: one permutation shared by the batch.
            if generator is None:
                perm = torch.randperm(L, device=tokens.device)
            else:
                perm = torch.randperm(L, generator=generator,
                                      device=generator.device).to(tokens.device)
            mask_indices = perm[:num_mask].unsqueeze(0).repeat(B, 1)
        else:
            # A per-sample draw, because which positions are eligible is a
            # per-sample fact. Ranking uniform noise with the ineligible
            # positions at -inf picks num_mask of the eligible ones uniformly.
            if generator is None:
                noise = torch.rand(B, L, device=tokens.device)
            else:
                noise = torch.rand(B, L, device=generator.device,
                                   generator=generator).to(tokens.device)
            noise = noise.masked_fill(~valid_token_mask, float('-inf'))
            mask_indices = torch.topk(noise, num_mask, dim=1).indices

        # Create mask
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, mask_indices, True)

        return mask

    def apply_masking(self, tokens, mask):
        """Apply masking: replace masked positions with [MASK] token"""
        B, L, D = tokens.shape
        
        # Clone tokens
        masked_tokens = tokens.clone()
        
        # Replace masked positions with [MASK] token
        mask_token_expanded = self.mask_token.expand(B, L, D)
        masked_tokens[mask] = mask_token_expanded[mask]
        
        return masked_tokens

    def patchify(self, imgs):
        """Convert images to patches"""
        B, C, F, T = imgs.shape
        p_f, p_t = self.patch_size
        assert F % p_f == 0 and T % p_t == 0
        
        f = F // p_f
        t = T // p_t
        
        x = imgs.reshape(shape=(B, C, f, p_f, t, p_t))
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(shape=(B, f * t, p_f * p_t * C))
        return x

    def prepare_tokens(self, x, channel_code=None):
        """Prepare tokens and add position encoding.

        ``channel_code`` is ``[C, Dc]``: one vector per EEG channel, injected
        after patching and before the position embedding, so the position
        embedding and RoPE both see exactly the sequence they saw before plus a
        per-channel offset.
        """
        B, C, F, T = x.shape
        
        # Patch embedding
        tokens = self.patch_embed(x)
        _, L, D = tokens.shape

        if channel_code is not None:
            tokens = self._inject_channel_tokens(tokens, channel_code, F, T)
        
        # Add position encoding
        if self.pos_embed is not None:
            if self.pos_embed_type == '2d':
                p_f, p_t = self.patch_size
                patches_per_freq = F // p_f
                patches_per_time = T // p_t
                tokens = self.pos_embed(tokens, freq_size=patches_per_freq, time_size=patches_per_time)
            else:
                tokens = self.pos_embed(tokens)
        
        return tokens

    def fold_scales(self, wave_spec):
        """``[B, (J+1)*C, T] -> [B, C, T]``, or a pass-through when disabled."""
        return self.fold(wave_spec)

    def scale_fold_reg(self):
        """KL(alpha || uniform) from the last forward, or ``None``.

        Only the dynamic modes produce one. Left for the trainer to add: a fold
        that has collapsed onto a single band is a *better* fit to the training
        set and the task loss will not object, so the pressure to keep using
        several scales has to come from outside it.
        """
        return self.fold.reg_loss

    def scale_fold_alpha(self):
        """Mean mixing weight per scale from the last forward, or ``None``."""
        return self.fold.alpha_mean

    def scale_fold_spread(self):
        """``(std over time blocks, std over channels)`` per scale, or ``None``.

        The mean returned by :meth:`scale_fold_alpha` averages over batch,
        channel and time block, so a fold that swings from block to block and
        one frozen at 1/S both report a flat vector. These say which it is:
        alpha_std_time is the spread the "per block" claim rests on, and it is
        zero for a static fold whatever the mean looks like.
        """
        return self.fold.alpha_std_time, self.fold.alpha_std_chan

    def scale_fold_per_channel(self):
        """``[C, S]`` mean mixing weight per channel from the last forward."""
        return self.fold.alpha_chan

    def scale_fold_blocks(self):
        """Last forward's full ``[B, C, N, S]``, or ``None`` unless enabled.

        Set ``model.fold.keep_alpha = True`` first; scripts/alpha_probe.py does.
        """
        return self.fold.alpha_blocks

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints written before ScaleFold existed keep the static weights
        # at the model root. Without this they would land in the "unexpected
        # keys" list, the fold would silently stay at its uniform init, and the
        # run would look like a reproduction that had quietly lost its fold.
        old = prefix + 'scale_weight'
        new = prefix + 'fold.scale_weight'
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _inject_channel_tokens(self, tokens, code, n_rows, n_samples):
        """``[B, C*P, D]`` + one vector per channel, broadcast over its own P.

        PatchEmbed produces the sequence channel-major and time-minor:
        Conv2d over ``[B, 1, C, T]`` gives ``[B, D, C, T/p]``, ``flatten(2)``
        walks C then P, and ``transpose`` puts D last -- so token ``c*P + p``
        belongs to channel ``c``. Reshaping to ``[B, C, P, D]`` is therefore the
        semantic view and not a reinterpretation, and a code added on the C axis
        reaches that channel's P patches and no others. A sentinel test pins it.
        """
        B, L, D = tokens.shape
        p_f, p_t = self.patch_size
        C, P = n_rows // p_f, n_samples // p_t
        if C * P != L:
            raise RuntimeError(f"token count {L} is not C*P = {C}*{P}")
        if code.dim() == 2:
            code = code.unsqueeze(0)                       # [1, C, Dc]
        if code.shape[-2] != C:
            raise ValueError(
                f"channel code has {code.shape[-2]} channels, the folded "
                f"spectrogram has {C} rows")
        delta = self.channel_to_token(code)                # [*, C, D]
        delta = torch.tanh(self.channel_token_gate) * delta
        tokens = tokens.reshape(B, C, P, D) + delta.unsqueeze(-2)
        return tokens.reshape(B, L, D)

    def _channel_scale_bias(self, code):
        """``[C, S]`` prior on the fold's logits, already gated."""
        return torch.tanh(self.channel_fold_gate) * self.channel_to_scale(code)

    def forward_features(self, x, channel_meta=None):
        """Extract features (encoder part)"""
        code = self._channel_code(channel_meta)
        fold_bias = token_code = None
        if code is not None:
            if self.channel_injection in ('fold', 'dual'):
                fold_bias = self._channel_scale_bias(code)
            if self.channel_injection in ('token', 'dual'):
                token_code = code

        # 1. Wavelet decomposition
        wave_spec = self.fold(self.wavelet_decomp(x), channel_scale_bias=fold_bias)
        wave_2d = wave_spec.unsqueeze(1)
        
        # 2. Patch embedding and position encoding
        tokens = self.prepare_tokens(wave_2d, channel_code=token_code)
        
        # 3. Transformer encoding
        features = self.encoder(tokens)
        
        return features

    def valid_token_mask(self, channel_meta, n_rows, n_samples, batch_size,
                         device):
        """``[B, C*P]`` True where the token is a real measurement, or ``None``.

        ``channel_meta['valid_channel_mask']`` is per channel; a channel is
        either measured or it is a padded slot for its whole window, so the
        per-token answer is that flag repeated over the channel's P patches.
        The repeat has to be ``repeat_interleave`` and not ``repeat``: the
        sequence is channel-major, token ``c*P + p`` belonging to channel ``c``,
        and the wrong one would silently mark a scattering of real tokens
        invalid instead.
        """
        if channel_meta is None:
            return None
        valid = channel_meta.get('valid_channel_mask')
        if valid is None:
            return None
        p_f, p_t = self.patch_size
        C, P = n_rows // p_f, n_samples // p_t
        valid = valid.to(device=device, dtype=torch.bool).reshape(-1)
        if valid.numel() != C:
            raise ValueError(
                f"valid_channel_mask has {valid.numel()} entries, the folded "
                f"spectrogram has {C} rows")
        return valid.repeat_interleave(P).unsqueeze(0).expand(batch_size, -1)

    def forward_pretrain(self, x, mask_ratio=None, channel_meta=None,
                         mask_generator=None):
        """Pretraining forward pass.

        ``channel_meta`` follows the same contract as :meth:`forward_features`:
        the code is built once per montage and enters at the sites named by
        ``channel_injection``. Nothing about the objective changes -- the target
        is still the folded wavelet patch, and the code conditions only what the
        encoder is given, never what it is asked to reproduce.

        Padded channel slots are excluded from mask selection, which is also how
        they leave the loss: the reconstruction term is taken over masked tokens
        only, so a token that is never masked is never scored.

        ``mask_generator`` fixes the random half of the mask, so a validation
        sample can be given the same mask at every epoch.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        code = self._channel_code(channel_meta)
        fold_bias = token_code = None
        if code is not None:
            if self.channel_injection in ('fold', 'dual'):
                fold_bias = self._channel_scale_bias(code)
            if self.channel_injection in ('token', 'dual'):
                token_code = code

        # Wavelet decomposition
        wave_spec = self.fold(self.wavelet_decomp(x), channel_scale_bias=fold_bias)
        wave_2d = wave_spec.unsqueeze(1)

        # Patch embedding and position encoding
        tokens = self.prepare_tokens(wave_2d, channel_code=token_code)

        # Get original patches as reconstruction target
        target_patches = self.patchify(wave_2d)

        valid_tokens = self.valid_token_mask(
            channel_meta, wave_2d.shape[2], wave_2d.shape[3],
            tokens.shape[0], tokens.device)

        # Select mask positions
        if self.masking_strategy == 'frequency_guided':
            mask = self.frequency_guided_masking(
                tokens, mask_ratio, self.importance_ratio,
                valid_token_mask=valid_tokens, generator=mask_generator)
        else:  # 'random'
            mask = self.random_masking(
                tokens, mask_ratio, valid_token_mask=valid_tokens,
                generator=mask_generator)

        # Apply masking
        masked_tokens = self.apply_masking(tokens, mask)

        # Encoder processing
        encoded_tokens = self.encoder(masked_tokens)

        # Reconstruction head prediction
        pred_patches = self.task_heads['pretrain'](encoded_tokens)

        return pred_patches, mask, target_patches

    def forward_downstream(self, x, task_name, channel_meta=None):
        """Downstream task forward pass"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task head '{task_name}' not found. Available: {list(self.task_heads.keys())}")
        
        # Extract features
        features = self.forward_features(x, channel_meta=channel_meta)
        
        # Task head prediction
        output = self.task_heads[task_name](features)
        
        return output

    def forward(self, x, task='features', mask_ratio=None, task_name=None,
                channel_meta=None, mask_generator=None):
        """
        Unified forward pass interface
        
        Args:
            x: [B, C, T] - Input time series signal
            task: 'features', 'pretrain', 'downstream'
            mask_ratio: Masking ratio (for pretraining)
            task_name: Downstream task name
            channel_meta: dict of numeric tensors describing the montage, or
                None. Required whenever channel_encoding is not 'none'; the
                encoder raises rather than quietly running without it. Names are
                resolved to ids in preprocessing -- nothing here parses a string.
            
        Returns:
            Different results based on task
        """
        if channel_meta is not None and self.channel_encoder is None:
            raise ValueError(
                "channel_meta was given but this model was built with "
                "channel_encoding='none'; it would be silently ignored.")
        if task == 'features':
            return self.forward_features(x, channel_meta=channel_meta)
        elif task == 'pretrain':
            return self.forward_pretrain(x, mask_ratio,
                                         channel_meta=channel_meta,
                                         mask_generator=mask_generator)
        elif task == 'downstream':
            if task_name is None:
                raise ValueError("task_name must be specified for downstream tasks")
            return self.forward_downstream(x, task_name, channel_meta=channel_meta)
        else:
            # Compatibility with old interface
            if task == 'classify' and 'classification' in self.task_heads:
                return self.forward_downstream(x, 'classification',
                                               channel_meta=channel_meta)
            elif task in self.task_heads:
                return self.forward_downstream(x, task, channel_meta=channel_meta)
            else:
                raise ValueError(f"Unknown task: {task}")


# Convenience constructor functions
def create_wavelet_classifier(in_channels=8, max_level=3, embed_dim=256, depth=8, 
                             num_heads=8, num_classes=2, **kwargs):
    """Create wavelet classifier"""
    return BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        task_type='classification',
        num_classes=num_classes,
        **kwargs
    )


def create_wavelet_regressor(in_channels=8, max_level=3, embed_dim=256, depth=8,
                            num_heads=8, output_dim=1, **kwargs):
    """Create wavelet regressor"""
    return BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        task_type='regression',
        output_dim=output_dim,
        **kwargs
    )


def create_wavelet_pretrain_model(in_channels=8, max_level=3, embed_dim=256, depth=8,
                                 num_heads=8, **kwargs):
    """Create pretraining model"""
    return BERTWaveletTransformer(
        in_channels=in_channels,
        max_level=max_level,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        task_type='pretrain',
        **kwargs
    )