"""
BERT-style Wavelet Transformer Main Model
Refactored version with modular design and clear component separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_modules import SoftGateWaveletDecomp
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
                 # Patch embedding parameters
                 patch_size=(1,20),
                 embed_dim=128,
                 # Transformer parameters
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 rope_dim=None,
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
            ffn_ratio=4.0,
            ffn_kernel_size=3,
            ffn_drop=0.1
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
            rope_dim=rope_dim
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
    
    def frequency_guided_masking(self, tokens, mask_ratio, importance_ratio=0.6):
        """Frequency-domain importance-based masking strategy"""
        B, L, D = tokens.shape
        num_mask = int(L * mask_ratio)

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
        random_noise = torch.rand(B, L, device=tokens.device)
        combined_scores = importance_ratio * importance_full + (1 - importance_ratio) * random_noise

        # Select positions with highest scores for masking
        _, mask_indices = torch.topk(combined_scores, num_mask, dim=1)
        
        # Create mask
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, mask_indices, True)
        
        return mask

    def random_masking(self, tokens, mask_ratio):
        """Random masking strategy"""
        B, L, D = tokens.shape
        num_mask = int(L * mask_ratio)
        
        # Randomly select mask positions
        mask_indices = torch.randperm(L, device=tokens.device)[:num_mask].unsqueeze(0).repeat(B, 1)
        
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

    def prepare_tokens(self, x):
        """Prepare tokens and add position encoding"""
        B, C, F, T = x.shape
        
        # Patch embedding
        tokens = self.patch_embed(x)
        _, L, D = tokens.shape
        
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

    def forward_features(self, x):
        """Extract features (encoder part)"""
        # 1. Wavelet decomposition
        wave_spec = self.wavelet_decomp(x)
        wave_2d = wave_spec.unsqueeze(1)
        
        # 2. Patch embedding and position encoding
        tokens = self.prepare_tokens(wave_2d)
        
        # 3. Transformer encoding
        features = self.encoder(tokens)
        
        return features

    def forward_pretrain(self, x, mask_ratio=None):
        """Pretraining forward pass"""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Wavelet decomposition
        wave_spec = self.wavelet_decomp(x)
        wave_2d = wave_spec.unsqueeze(1)
        
        # Patch embedding and position encoding
        tokens = self.prepare_tokens(wave_2d)
        
        # Get original patches as reconstruction target
        target_patches = self.patchify(wave_2d)
        
        # Select mask positions
        if self.masking_strategy == 'frequency_guided':
            mask = self.frequency_guided_masking(tokens, mask_ratio, self.importance_ratio)
        else:  # 'random'
            mask = self.random_masking(tokens, mask_ratio)
        
        # Apply masking
        masked_tokens = self.apply_masking(tokens, mask)
        
        # Encoder processing
        encoded_tokens = self.encoder(masked_tokens)
        
        # Reconstruction head prediction
        pred_patches = self.task_heads['pretrain'](encoded_tokens)
        
        return pred_patches, mask, target_patches

    def forward_downstream(self, x, task_name):
        """Downstream task forward pass"""
        if task_name not in self.task_heads:
            raise ValueError(f"Task head '{task_name}' not found. Available: {list(self.task_heads.keys())}")
        
        # Extract features
        features = self.forward_features(x)
        
        # Task head prediction
        output = self.task_heads[task_name](features)
        
        return output

    def forward(self, x, task='features', mask_ratio=None, task_name=None):
        """
        Unified forward pass interface
        
        Args:
            x: [B, C, T] - Input time series signal
            task: 'features', 'pretrain', 'downstream'
            mask_ratio: Masking ratio (for pretraining)
            task_name: Downstream task name
            
        Returns:
            Different results based on task
        """
        if task == 'features':
            return self.forward_features(x)
        elif task == 'pretrain':
            return self.forward_pretrain(x, mask_ratio)
        elif task == 'downstream':
            if task_name is None:
                raise ValueError("task_name must be specified for downstream tasks")
            return self.forward_downstream(x, task_name)
        else:
            # Compatibility with old interface
            if task == 'classify' and 'classification' in self.task_heads:
                return self.forward_downstream(x, 'classification')
            elif task in self.task_heads:
                return self.forward_downstream(x, task)
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