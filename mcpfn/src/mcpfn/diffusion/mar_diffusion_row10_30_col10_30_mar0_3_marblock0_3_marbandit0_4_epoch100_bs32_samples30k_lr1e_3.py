"""
MAR Diffusion Model Implementation
Model: diffusion-row-10-30-col-10-30-mar0.3-marblock0.3-marbandit0.4-epoch100-bs32-samples30k-lr1e-3

This module implements a Discrete Denoising Diffusion Probabilistic Model (D3PM) for generating
missingness patterns in tabular data. The model is trained on three types of missingness:
- Bandit-based missingness (40% weight)
- MAR (Missing At Random) missingness (30% weight)  
- Block MAR missingness (30% weight)

The model uses a fully convolutional architecture with both discrete (missingness type) and
continuous (numeric matrix) conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
from pathlib import Path

# Configuration for the diffusion model
mar_diffusion_config = {
    'missingness_type': 'bandit',  # 'bandit', 'mar', 'block_mar', or None for random
    'device': 'cpu',
    'target_shape': None,  # (height, width) or None to use input shape
    'num_samples': 1,
    'pad_mode': 'pad_crop'  # 'pad_crop' or 'tile'
}

class FlexibleConvBlock(nn.Module):
    """Fully convolutional block that can handle any input size"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(out_channels // 8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(out_channels // 8, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)
        self.activation = nn.LeakyReLU()
        
        # Optional residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        
        return x + residual

class FlexibleUpBlock(nn.Module):
    """Fully convolutional upsampling block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(out_channels // 8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(out_channels // 8, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)
        self.activation = nn.LeakyReLU()
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        
        return x + residual

class ConvolutionalSelfAttention(nn.Module):
    """Convolutional self-attention to replace transformers"""
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Multi-head attention using convolutions
        self.q_conv = nn.Conv1d(d_model, d_model, 1)
        self.k_conv = nn.Conv1d(d_model, d_model, 1)
        self.v_conv = nn.Conv1d(d_model, d_model, 1)
        self.out_conv = nn.Conv1d(d_model, d_model, 1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        # x shape: (B, seq_len, d_model)
        residual = x
        
        # Multi-head attention
        x_norm = self.norm1(x)
        x_transposed = x_norm.transpose(1, 2)  # (B, d_model, seq_len)
        
        q = self.q_conv(x_transposed).transpose(1, 2)  # (B, seq_len, d_model)
        k = self.k_conv(x_transposed).transpose(1, 2)  # (B, seq_len, d_model)
        v = self.v_conv(x_transposed).transpose(1, 2)  # (B, seq_len, d_model)
        
        # Reshape for multi-head attention
        B, seq_len, d_model = q.shape
        q = q.reshape(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, seq_len, d_model)
        
        # Output projection
        attn_output = self.out_conv(attn_output.transpose(1, 2)).transpose(1, 2)
        
        # Residual connection
        x = residual + attn_output
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x 

class FullyConvolutionalX0Model(nn.Module):
    """Fully convolutional model that can handle any input size with both discrete and continuous conditioning"""
    
    def __init__(self, n_channel: int, N: int = 2, num_classes: int = 4) -> None:
        super(FullyConvolutionalX0Model, self).__init__()
        self.N = N
        
        # === 1. FLEXIBLE CONVOLUTIONAL BLOCKS ===
        # These can handle any input size
        self.conv1 = FlexibleConvBlock(n_channel, 16)
        self.conv2 = FlexibleConvBlock(16, 32)
        self.conv3 = FlexibleConvBlock(32, 64)
        self.conv4 = FlexibleConvBlock(64, 128)
        self.conv5 = FlexibleConvBlock(128, 256)
        self.conv6 = FlexibleConvBlock(256, 512)
        
        # === 2. ADAPTIVE POOLING ===
        # Pool to fixed size for global processing
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_pool_spatial = nn.AdaptiveAvgPool2d((4, 4))
        
        # === 3. CONVOLUTIONAL SELF-ATTENTION (REPLACES TRANSFORMERS) ===
        self.self_attention_1 = ConvolutionalSelfAttention(512, 8)
        self.self_attention_2 = ConvolutionalSelfAttention(512, 8)
        self.self_attention_3 = ConvolutionalSelfAttention(64, 8)
        
        # === 4. ADAPTIVE UPSAMPLING ===
        self.up1 = FlexibleUpBlock(512, 256)
        self.up2 = FlexibleUpBlock(256, 128)
        self.up3 = FlexibleUpBlock(128, 64)
        self.up4 = FlexibleUpBlock(64, 32)
        self.up5 = FlexibleUpBlock(32, 16)
        
        # === 5. DISCRETE CONDITIONING EMBEDDINGS (ADAPTIVE) ===
        self.cond_embedding = nn.Embedding(num_classes, 512)  # Match max channel dimension
        
        # === 6. CONTINUOUS CONDITIONING PROJECTIONS ===
        # Projection layers for continuous matrix to different channel dimensions
        self.cont_proj_16 = nn.Conv2d(1, 16, 1)
        self.cont_proj_32 = nn.Conv2d(1, 32, 1)
        self.cont_proj_64 = nn.Conv2d(1, 64, 1)
        self.cont_proj_128 = nn.Conv2d(1, 128, 1)
        self.cont_proj_256 = nn.Conv2d(1, 256, 1)
        self.cont_proj_512 = nn.Conv2d(1, 512, 1)
        
        # === 7. TIMESTEP EMBEDDING (ADAPTIVE) ===
        # Separate timestep embeddings for each channel dimension
        self.temb_16 = nn.Linear(32, 16)   # For 16-channel layers
        self.temb_32 = nn.Linear(32, 32)   # For 32-channel layers
        self.temb_64 = nn.Linear(32, 64)   # For 64-channel layers
        self.temb_128 = nn.Linear(32, 128) # For 128-channel layers
        self.temb_256 = nn.Linear(32, 256) # For 256-channel layers
        self.temb_512 = nn.Linear(32, 512) # For 512-channel layers
        
        # === 8. FINAL OUTPUT ===
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)
    
    def forward(self, x, t, cond, cont_matrix) -> torch.Tensor:
        # Store original size for final output
        original_size = x.shape[2:]
        
        # === 1. INPUT PREPROCESSING (ANY SIZE) ===
        x = (2 * x.float() / self.N) - 1.0
        
        # === 2. TIMESTEP EMBEDDING (ADAPTIVE) ===
        t = t.float().reshape(-1, 1) / 1000
        t_features = [torch.sin(t * 3.1415 * 2**i) for i in range(16)] + [
            torch.cos(t * 3.1415 * 2**i) for i in range(16)
        ]
        tx = torch.cat(t_features, dim=1).to(x.device)
        
        # Create timestep embeddings for each channel dimension
        t_emb_16 = self.temb_16(tx).unsqueeze(-1).unsqueeze(-1)   # (B, 16, 1, 1)
        t_emb_32 = self.temb_32(tx).unsqueeze(-1).unsqueeze(-1)   # (B, 32, 1, 1)
        t_emb_64 = self.temb_64(tx).unsqueeze(-1).unsqueeze(-1)   # (B, 64, 1, 1)
        t_emb_128 = self.temb_128(tx).unsqueeze(-1).unsqueeze(-1) # (B, 128, 1, 1)
        t_emb_256 = self.temb_256(tx).unsqueeze(-1).unsqueeze(-1) # (B, 256, 1, 1)
        t_emb_512 = self.temb_512(tx).unsqueeze(-1).unsqueeze(-1) # (B, 512, 1, 1)
        
        # === 3. DISCRETE CONDITIONING EMBEDDING ===
        # Get spatial dimensions dynamically
        B, C, H, W = x.shape
        
        # Create adaptive conditioning using direct slicing
        cond_emb = self.cond_embedding(cond)  # (B, 512) - already max channel dimension
        cond_emb = cond_emb.unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        cond_emb = F.interpolate(cond_emb, size=(H, W), mode='bilinear', align_corners=False)  # (B, 512, H, W)
        
        # Adaptive conditioning function: slice + interpolate (no projection needed)
        def get_adaptive_conditioning(cond_emb, target_channels):
            # Step 1: Slice to target channels (direct slicing, no projection)
            sliced = cond_emb[:, :target_channels, :, :]  # (B, target_channels, H, W)
            return sliced
        
        # Get adaptive conditioning for each layer
        cond_emb_16 = get_adaptive_conditioning(cond_emb, 16)  # (B, 16, H, W)
        cond_emb_32 = get_adaptive_conditioning(cond_emb, 32)  # (B, 32, H, W)
        cond_emb_64 = get_adaptive_conditioning(cond_emb, 64)  # (B, 64, H, W)
        cond_emb_128 = get_adaptive_conditioning(cond_emb, 128)  # (B, 128, H, W)
        cond_emb_256 = get_adaptive_conditioning(cond_emb, 256)  # (B, 256, H, W)
        cond_emb_512 = get_adaptive_conditioning(cond_emb, 512)  # (B, 512, H, W)
        
        # Global conditioning (fixed size)
        cond_emb_global = self.cond_embedding(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        
        # === 4. CONTINUOUS CONDITIONING PROCESSING ===
        # Normalize continuous matrix
        cont_matrix = (cont_matrix - cont_matrix.mean()) / (cont_matrix.std() + 1e-8)
        
        # Project continuous matrix to different channel dimensions
        cont_emb_16 = self.cont_proj_16(cont_matrix)  # (B, 16, H, W)
        cont_emb_32 = self.cont_proj_32(cont_matrix)  # (B, 32, H, W)
        cont_emb_64 = self.cont_proj_64(cont_matrix)  # (B, 64, H, W)
        cont_emb_128 = self.cont_proj_128(cont_matrix)  # (B, 128, H, W)
        cont_emb_256 = self.cont_proj_256(cont_matrix)  # (B, 256, H, W)
        cont_emb_512 = self.cont_proj_512(cont_matrix)  # (B, 512, H, W)
        
        # Global continuous conditioning (fixed size)
        cont_emb_global = F.adaptive_avg_pool2d(cont_matrix, (1, 1))  # (B, 1, 1, 1)
        cont_emb_global = self.cont_proj_512(cont_emb_global)  # (B, 512, 1, 1)
        
        # === 5. FLEXIBLE CONVOLUTIONAL PROCESSING WITH BOTH CONDITIONINGS ===
        # Process at original size with adaptive conditioning and proper timestep embeddings
        x1 = self.conv1(x) + t_emb_16 + cond_emb_16 + cont_emb_16
        x2 = self.conv2(x1) + t_emb_32 + cond_emb_32 + cont_emb_32
        x3 = self.conv3(x2) + t_emb_64 + cond_emb_64 + cont_emb_64
        x4 = self.conv4(x3) + t_emb_128 + cond_emb_128 + cont_emb_128
        x5 = self.conv5(x4) + t_emb_256 + cond_emb_256 + cont_emb_256
        x6 = self.conv6(x5) + t_emb_512 + cond_emb_512 + cont_emb_512
        
        # === 6. GLOBAL FEATURES (FIXED SIZE) ===
        # Pool to fixed size for global processing
        global_features = self.global_pool(x6)  # Shape: (B, 512, 1, 1)
        global_features = global_features + t_emb_512 + cond_emb_global + cont_emb_global
        
        # === 7. CONVOLUTIONAL SELF-ATTENTION (REPLACES TRANSFORMERS) ===
        # Reshape for attention
        global_features_flat = global_features.squeeze(-1).squeeze(-1)  # (B, 512)
        global_features_attended = self.self_attention_1(global_features_flat.unsqueeze(1))  # (B, 1, 512)
        global_features_attended = global_features_attended.squeeze(1)  # (B, 512)
        global_features_attended = global_features_attended.unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        
        # === 8. SPATIAL FEATURES (ADAPTIVE SIZE) ===
        # Pool to medium size for spatial processing
        spatial_features = self.global_pool_spatial(x6)  # Shape: (B, 512, 4, 4)
        # Create adaptive conditioning for spatial features
        cond_emb_spatial_4x4 = cond_emb  # (B, 512, H, W) - already correct size
        cond_emb_spatial_4x4 = F.interpolate(cond_emb_spatial_4x4, size=(4, 4), mode='bilinear', align_corners=False)  # (B, 512, 4, 4)
        # Create timestep embedding for spatial features (512 channels)
        t_emb_spatial_4x4 = F.interpolate(t_emb_512, size=(4, 4), mode='bilinear', align_corners=False)  # (B, 512, 4, 4)
        # Create continuous embedding for spatial features
        cont_emb_spatial_4x4 = F.interpolate(cont_emb_512, size=(4, 4), mode='bilinear', align_corners=False)  # (B, 512, 4, 4)
        spatial_features = spatial_features + t_emb_spatial_4x4 + cond_emb_spatial_4x4 + cont_emb_spatial_4x4
        
        # Spatial attention
        spatial_features_flat = spatial_features.flatten(2).transpose(1, 2)  # (B, 16, 512)
        spatial_features_attended = self.self_attention_2(spatial_features_flat)  # (B, 16, 512)
        spatial_features_attended = spatial_features_attended.transpose(1, 2).reshape_as(spatial_features)  # (B, 512, 4, 4)
        
        # === 9. ADAPTIVE UPSAMPLING ===
        # Combine global and spatial features
        combined_features = global_features_attended + spatial_features_attended
        
        # Upsample back to original size
        y = F.interpolate(combined_features, size=original_size, mode='bilinear', align_corners=False)
        
        # Progressive upsampling
        y = self.up1(y)
        y = self.up2(y)
        y = self.up3(y)
        y = self.up4(y)
        y = self.up5(y)
        
        # === 10. FINAL OUTPUT ===
        y = self.final(y)
        
        # Reshape to match expected output format
        y = y.reshape(y.shape[0], -1, self.N, *x.shape[2:]).transpose(2, 3).transpose(3, 4).contiguous()
        
        return y 

class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative

        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_one_step_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        # Handle shape mismatch - if x_0_logits has different spatial dimensions than x_t
        if len(x_0_logits.shape) != len(x_t.shape) + 1:
            # Reshape x_0_logits to match expected format
            if len(x_0_logits.shape) == 4 and len(x_t.shape) == 3:
                # x_0_logits is (B, H, W, N), x_t is (B, C, H, W)
                x_0_logits = x_0_logits.permute(0, 3, 1, 2)  # (B, N, H, W)
                x_0_logits = x_0_logits.permute(0, 2, 3, 1)  # (B, H, W, N)
            elif len(x_0_logits.shape) == 5 and len(x_t.shape) == 4:
                # x_0_logits is (B, C, H, W, N), x_t is (B, C, H, W)
                x_0_logits = x_0_logits.squeeze(1)  # Remove extra channel dim
        
        # Ensure shapes match
        if x_0_logits.shape[:-1] != x_t.shape:
            print(f"Shape mismatch: x_0_logits={x_0_logits.shape}, x_t={x_t.shape}")
            # Try to reshape x_0_logits to match x_t
            if len(x_0_logits.shape) == 4 and len(x_t.shape) == 4:
                # x_0_logits: (B, H, W, N), x_t: (B, C, H, W)
                x_0_logits = x_0_logits.permute(0, 3, 1, 2)  # (B, N, H, W)
                x_0_logits = x_0_logits.permute(0, 2, 3, 1)  # (B, H, W, N)
                x_t = x_t.squeeze(1)  # (B, H, W)
        
        assert x_0_logits.shape[:-1] == x_t.shape, f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=1) * (
            torch.log_softmax(dist1 + self.eps, dim=1)
            - torch.log_softmax(dist2 + self.eps, dim=1)
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_0, t, cond, cont_matrix):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        predicted_x0_logits = self.x0_model(x_0, t, cond, cont_matrix)

        return predicted_x0_logits

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, cont_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        x_t = self.q_sample(
            x, t, torch.rand((*x.shape, self.num_classses), device=x.device)
        )
        # x_t is same shape as x
        assert x_t.shape == x.shape, print(
            f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
        )
        # we use hybrid loss.

        predicted_x0_logits = self.model_predict(x_t, t, cond, cont_matrix)

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * vb_loss + ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, cont_matrix, noise):

        predicted_x0_logits = self.model_predict(x, t, cond, cont_matrix)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample(self, x, cond=None, cont_matrix=None):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, cont_matrix, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x

    def sample_with_image_sequence(self, x, cond=None, cont_matrix=None, stride=10):
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, cont_matrix, torch.rand((*x.shape, self.num_classses), device=x.device)
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images

class MARDiffusionModel(nn.Module):
    """Wrapper class for the MAR Diffusion Model"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.n_channel = config.get('n_channel', 1)
        self.N = config.get('N', 2)
        self.num_classes = config.get('num_classes', 3)
        self.diffusion_steps = config.get('diffusion_steps', 1000)
        self.hybrid_loss_coeff = config.get('hybrid_loss_coeff', 0.0)
        
        # Create the X0 model
        self.x0_model = FullyConvolutionalX0Model(
            n_channel=self.n_channel,
            N=self.N,
            num_classes=self.num_classes
        )
        
        # Create D3PM wrapper
        self.d3pm = D3PM(
            x0_model=self.x0_model,
            n_T=self.diffusion_steps,
            num_classes=self.N,  # Use N (binary values), not num_classes (missingness types)
            forward_type="uniform",
            hybrid_loss_coeff=self.hybrid_loss_coeff
        )
    
    def forward(self, x: torch.Tensor, x_cond: torch.Tensor, 
                class_cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass for training"""
        return self.d3pm.model_predict(x, timesteps, class_cond, x_cond)
    
    def generate(self, shape: Tuple[int, int], x_cond: torch.Tensor, 
                 missingness_type: Optional[str] = None, num_samples: int = 1) -> torch.Tensor:
        """Generate missingness patterns"""
        height, width = shape
        device = next(self.parameters()).device
        
        # Map missingness type to class index
        if missingness_type is None:
            class_idx = torch.randint(0, self.num_classes, (num_samples,), device=device)
        else:
            class_mapping = {'bandit': 0, 'mar': 1, 'block_mar': 2}
            class_idx = torch.tensor([class_mapping[missingness_type]] * num_samples, device=device)
        
        # Start with random discrete tokens (integer for D3PM)
        x = torch.randint(0, self.N, (num_samples, 1, height, width), device=device, dtype=torch.long)
        
        # Generate samples using D3PM
        with torch.no_grad():
            samples = self.d3pm.sample(x, class_idx, x_cond)
        
        # Convert to binary mask
        mask = (samples.squeeze(1) == 1).float()
        
        return mask

class MARDiffusion:
    """MAR Diffusion missingness pattern generator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.diffusion_config = mar_diffusion_config.copy()
        self.diffusion_config.update(config)
        
        # Model configuration
        model_config = {
            'n_channel': 1,
            'N': 2,
            'num_classes': 3,  # bandit, mar, block_mar
            'diffusion_steps': 1000,
            'hybrid_loss_coeff': 0.0
        }
        
        # Initialize model
        self.model = MARDiffusionModel(model_config).to(self.device)
        
        # Load pretrained weights
        self._load_model_weights()
        
        # Set to evaluation mode
        self.model.eval()
    
    def _load_model_weights(self):
        """Load pre-trained model weights"""
        weights_path = Path(__file__).parent / "models" / "diffusion-row-10-30-col-10-30-mar0.3-marblock0.3-marbandit0.4-epoch100-bs32-samples30k-lr1e-3" / "model_final.pth"
        
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            try:
                self.model.x0_model.load_state_dict(state_dict, strict=True)
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Warning: Could not load all weights: {e}")
                self.model.x0_model.load_state_dict(state_dict, strict=False)
                print("Loaded weights with partial matching")
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    def _induce_missingness(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate missingness pattern using the diffusion model.
        
        Args:
            X: Input tensor of shape (height, width)
            
        Returns:
            Tensor with missing values (NaN) applied
        """
        height, width = X.shape
        
        # Use target_shape if specified, otherwise use X shape
        if self.diffusion_config['target_shape'] is not None:
            target_height, target_width = self.diffusion_config['target_shape']
        else:
            target_height, target_width = height, width
        
        # Check if matrix size is within training range
        if not (10 <= target_height <= 30 and 10 <= target_width <= 30):
            warnings.warn(f"Matrix size ({target_height}, {target_width}) outside training range (10-30). "
                         "Results may be suboptimal.")
        
        # Prepare X conditioning tensor
        X_cond = X.unsqueeze(0).unsqueeze(0)  # (1, 1, height, width)
        X_cond = X_cond.to(self.device)
        
        # Resize X_cond to target shape if needed
        if (target_height, target_width) != (height, width):
            X_cond = F.interpolate(X_cond, size=(target_height, target_width), mode='bilinear', align_corners=False)
        
        # Generate missingness pattern
        with torch.no_grad():
            missingness_mask = self.model.generate(
                shape=(target_height, target_width),
                x_cond=X_cond,
                missingness_type=self.diffusion_config['missingness_type'],
                num_samples=self.diffusion_config['num_samples']
            )
        
        # Convert to boolean mask and apply missingness
        mask = missingness_mask[0].bool()  # Remove batch dimension
        
        # If target shape differs from X shape, resize mask
        if (target_height, target_width) != (height, width):
            if self.diffusion_config['pad_mode'] == 'pad_crop':
                # Simple crop/pad to match X shape
                if target_height >= height and target_width >= width:
                    mask = mask[:height, :width]
                else:
                    # Pad if target is smaller
                    pad_h = max(0, height - target_height)
                    pad_w = max(0, width - target_width)
                    mask = F.pad(mask, (0, pad_w, 0, pad_h))
                    mask = mask[:height, :width]
            else:  # tile mode
                # Tile the mask to cover X shape
                mask = mask.repeat(height // target_height + 1, width // target_width + 1)[:height, :width]
        
        X_missing = X.clone()
        X_missing[mask] = torch.nan
        
        return X_missing

def create_mar_diffusion_pattern(config: Dict) -> MARDiffusion:
    """Factory function to create MAR Diffusion pattern"""
    return MARDiffusion(config)




# # Example usage:
# from mcpfn.diffusion.mar_diffusion_row10_30_col10_30_mar0_3_marblock0_3_marbandit0_4_epoch100_bs32_samples30k_lr1e_3 import MARDiffusion
# import torch

# # Create model
# config = {
#     'missingness_type': 'bandit',  # or 'mar', 'block_mar'
#     'device': 'cuda',
#     'target_shape': None,
#     'num_samples': 1,
#     'pad_mode': 'pad_crop'
# }

# mar_diffusion = MARDiffusion(config)

# # Generate missingness
# X = torch.randn(20, 25).to('cuda')
# X_missing = mar_diffusion._induce_missingness(X)