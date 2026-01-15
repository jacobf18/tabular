import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm
import os
import importlib.resources as resources
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from torch.optim import AdamW
from torch.nn.attention import SDPBackend, sdpa_kernel
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def torch_nanmean(
    x: torch.Tensor,
    nan_mask: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: bool = False,
    include_inf: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean of a tensor over a given dimension, ignoring NaNs.

    Designed for stability: If all inputs are NaN, the mean will be 0.0.

    Args:
        x: The input tensor.
        nan_mask: The mask of nan values.
        axis: The dimension to reduce.
        return_nanshare: If True, also return the proportion of NaNs.
        include_inf: If True, treat infinity as NaN for the purpose of the calculation.

    Returns:
        The mean of the input tensor, ignoring NaNs. If `return_nanshare` is True,
        returns a tuple containing the mean and the share of NaNs.
    """
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    if return_nanshare:
        return value / num, 1.0 - (num / x.shape[axis])
    return value / num.clip(min=1.0)

class TabImputeModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int, num_cls: int = 1):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size, num_cls)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        self.num_cls = num_cls

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        src = self.feature_encoder(src)

        # repeatedly applies the transformer block on (B,R,C,E)
        for block in self.transformer_blocks:
            src = block(src)
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        return self.decoder(src[:,self.num_cls:,self.num_cls:,:]) # remove the cls token
    
class SinusoidalColumnEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=100, damping_factor=0.1):
        super().__init__()
        # Pre-compute the positional encodings once to save speed
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(max_len) / embedding_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)
        self.damping_factor = damping_factor

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, num_rows, num_cols, embedding_size)
        """
        _, _, num_cols, _ = x.shape
        
        # Slice the pre-computed embeddings to the current number of columns
        # Shape: (1, 1, num_cols, embedding_size) for broadcasting
        col_embeddings = self.pe[:num_cols, :].unsqueeze(0).unsqueeze(0)
        
        return x + col_embeddings * self.damping_factor
    
import torch
import torch.nn as nn
import math

class SinusoidalRowEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=100, damping_factor=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.damping_factor = damping_factor
        
        # 1. Store the frequency term (div_term) as a buffer.
        # We MUST use the original max_len to define the curve's 'slope'.
        # If we change this later, the embeddings for pos 0-100 would change, 
        # confusing the model.
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float() * (-math.log(max_len) / embedding_size)
        )
        self.register_buffer('div_term', div_term)
        
        # 2. Pre-compute the initial PE cache
        # We verify if `pe` exists in forward, but initializing it here is good practice.
        self.register_buffer('pe', self._generate_pe(max_len))

    def _generate_pe(self, length):
        """
        Generates positional embeddings for positions [0, length).
        Uses the stored self.div_term to ensure consistency.
        """
        # Ensure we generate on the correct device/dtype
        pe = torch.zeros(length, self.embedding_size, device=self.div_term.device, dtype=self.div_term.dtype)
        position = torch.arange(0, length, dtype=self.div_term.dtype, device=self.div_term.device).unsqueeze(1)
        
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, num_rows, num_cols, embedding_size)
        """
        _, num_rows, _, _ = x.shape
        current_max_len = self.pe.size(0)

        # 3. Dynamic Extrapolation
        # If input is longer than our cache, extend the cache
        if num_rows > current_max_len:
            # Generate ONLY the new needed positions (e.g., from 100 to 150)
            # This is more efficient than regenerating the whole matrix
            new_positions = torch.arange(
                current_max_len, num_rows, 
                dtype=torch.float, 
                device=self.pe.device
            ).unsqueeze(1)
            
            new_pe = torch.zeros(
                num_rows - current_max_len, 
                self.embedding_size, 
                device=self.pe.device
            )
            
            new_pe[:, 0::2] = torch.sin(new_positions * self.div_term)
            new_pe[:, 1::2] = torch.cos(new_positions * self.div_term)
            
            # Concatenate and update the buffer so next time it's fast
            self.pe = torch.cat([self.pe, new_pe], dim=0)

        # Slice the embeddings to the current number of rows
        # Shape: (1, num_rows, 1, embedding_size) for broadcasting
        row_embeddings = self.pe[:num_rows, :].unsqueeze(0).unsqueeze(2).to(x.device).to(x.dtype)
        
        return x + row_embeddings * self.damping_factor

class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int, num_cls: int = 1):
        """ 
        Creates the embedding layer with support for multiple CLS tokens.
        
        Args:
            embedding_size: Size of the embedding vector
            num_cls: Number of CLS tokens to use (K). 
                     Creates a K-wide border on the top and left.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.num_cls = num_cls

        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size)
        )
        
        # Note: Damping factor 0.1 helps prevent embeddings from drowning out the data
        self.row_embedding = SinusoidalRowEmbedding(embedding_size, damping_factor=0.1)
        self.column_embedding = SinusoidalColumnEmbedding(embedding_size, damping_factor=0.1)
            
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        
        # 1. Row CLS Tokens (Left Side)
        # We need K tokens for every row. 
        # Shape: (1, 1, K, E) -> Will expand to (Batch, Rows, K, E)
        self.row_cls_token = nn.Parameter(torch.randn(1, 1, num_cls, embedding_size))
        
        # 2. Col CLS Tokens (Top Side)
        # We need K tokens for every column.
        # Shape: (1, K, 1, E) -> Will expand to (Batch, K, Cols, E)
        self.col_cls_token = nn.Parameter(torch.randn(1, num_cls, 1, embedding_size))
        
        # 3. Corner Tokens (Top-Left Intersection)
        # This is now a K x K block of tokens
        self.corner_token = nn.Parameter(torch.randn(1, num_cls, num_cls, embedding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (torch.Tensor) shape (batch_size, num_rows, num_features)
        Returns:
            (torch.Tensor) shape (batch_size, num_rows + K, num_features + K, embedding_size)
        """
        # --- 1. Embed Data ---
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, 0.0, x)
        
        embedded_observed = self.observed_linear_layer(x.unsqueeze(-1))
        
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(embedded_observed)
        
        B, R, C = x.shape
        mask_expanded = self.mask_token.expand(B, R, C, self.embedding_size)
        embedded = torch.where(nan_mask_expanded, mask_expanded, embedded_observed)
        
        # Add positions (only to the data part)
        embedded = self.column_embedding(embedded) + self.row_embedding(embedded)
        
        # --- 2. Expand CLS Tokens ---
        K = self.num_cls
        
        # Corner (Top-Left): (B, K, K, E)
        corner = self.corner_token.expand(B, K, K, self.embedding_size)
        
        # Col CLS (Top-Right): (B, K, C, E)
        col_cls = self.col_cls_token.expand(B, K, C, self.embedding_size)
        
        # Row CLS (Bottom-Left): (B, R, K, E)
        row_cls = self.row_cls_token.expand(B, R, K, self.embedding_size)
        
        # --- 3. Concatenate Grid ---
        
        # Top Strip: [Corner | Col_CLS] -> Shape (B, K, K+C, E)
        top_block = torch.cat([corner, col_cls], dim=2)
        
        # Bottom Strip: [Row_CLS | Data] -> Shape (B, R, K+C, E)
        bottom_block = torch.cat([row_cls, embedded], dim=2)
        
        # Full Grid: [Top / Bottom] -> Shape (B, K+R, K+C, E)
        out = torch.cat([top_block, bottom_block], dim=1)
        
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        
        self.gelu = nn.GELU()
        
        # Store attention weights for inspection
        self._last_feature_attention_weights = None
        self._last_datapoint_attention_weights = None

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # --- BLOCK 1: Feature Attention (Pre-Norm) ---
        src_norm = self.norm1(src) # Norm BEFORE Attention
        
        src_flat = src_norm.reshape(batch_size*rows_size, col_size, embedding_size)
        # Note: If batch_first=True, inputs are (Batch, Seq, Emb)
        attn_output = self.self_attention_between_features(src_flat, src_flat, src_flat, need_weights=True)
        src_att = attn_output[0]
        # Store attention weights: shape is (batch*rows, num_heads, seq_len, seq_len) or (batch*rows, seq_len, seq_len)
        if len(attn_output) > 1 and attn_output[1] is not None:
            self._last_feature_attention_weights = attn_output[1]
        src_att = src_att.reshape(batch_size, rows_size, col_size, embedding_size)
        
        src = src + src_att # Residual connection
        
        # --- BLOCK 2: Datapoint Attention (Pre-Norm) ---
        src_norm = self.norm2(src) # Norm BEFORE Attention
        
        # Reshape for Datapoint Attention
        # We treat (Batch * Cols) as the "Batch" dimension for the attention module
        # and 'Rows' as the sequence length.
        src_t = src.transpose(1, 2)
        src_flat = src_t.reshape(batch_size * col_size, rows_size, embedding_size)
        
        # --- CREATE DIAGONAL MASK ---
        # Shape: (Rows, Rows)
        # 0.0 means "Attend", -inf means "Ignore"
        # diag_mask = torch.zeros((rows_size, rows_size), device=src.device, dtype=src.dtype)
        # diag_mask.fill_diagonal_(float('-inf'))
        
        # Apply Attention with the Mask
        # Note: attn_mask in nn.MultiheadAttention expects (Seq_Len, Seq_Len)
        # or (Batch*Num_Heads, Seq_Len, Seq_Len). 
        # Passing (Rows, Rows) works and broadcasts to all batches.
        # src_flat = src_flat * 2.0
        attn_output = self.self_attention_between_datapoints(src_flat, src_flat, src_flat, need_weights=True)
        src_att = attn_output[0]
        # Store attention weights
        if len(attn_output) > 1 and attn_output[1] is not None:
            self._last_datapoint_attention_weights = attn_output[1]
        src_att = src_att.reshape(batch_size, col_size, rows_size, embedding_size)
        src_att = src_att.transpose(2, 1) # Back to (Batch, Row, Col, Emb)
        
        src = src + src_att # Residual
        
        # --- BLOCK 3: MLP (Pre-Norm) ---
        src_norm = self.norm3(src) # Norm BEFORE MLP
        src_mlp = self.linear2(self.gelu(self.linear1(src_norm)))
        src = src + src_mlp # Residual
        
        return src
    
    def get_attention_weights(self, attention_type: str = "features"):
        """
        Get the attention weights from the last forward pass.
        
        Args:
            attention_type: Either "features" or "datapoints" to specify which attention weights to return.
        
        Returns:
            Attention weights tensor, or None if not available.
        """
        if attention_type == "features":
            return self._last_feature_attention_weights
        elif attention_type == "datapoints":
            return self._last_datapoint_attention_weights
        else:
            raise ValueError(f"attention_type must be 'features' or 'datapoints', got {attention_type}")

class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """ Initializes the linear layers for use in the forward """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, num_outputs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to the embeddings to get the logits

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embedding_size)
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_outputs)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    # Increase model capacity significantly to ensure memorization
    num_attention_heads = 32
    embedding_size = 32 * num_attention_heads
    mlp_hidden_size = 1024
    num_cls = 8
    num_layers = 12
    model = TabImputeModel(
        embedding_size=embedding_size,  # Increased from 256
        num_attention_heads=num_attention_heads,  # Increased from 8
        mlp_hidden_size=mlp_hidden_size,  # Increased from 512
        num_layers=num_layers,  # Increased from 12
        num_outputs=5000,
        num_cls=num_cls,
    ).to('cuda')
    model.train()
    model = model.to(torch.bfloat16)
    torch.compile(model)
    
    with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as path:
        borders = torch.load(path).to(torch.device('cuda'))
        bar_distribution = FullSupportBarDistribution(borders=borders)
    
    # Keep borders in float32 for numerical stability (they're buffers, not trainable)
    # The bar_distribution can accept bfloat16 logits while keeping borders in float32
    
    opt = AdamW(model.parameters(), lr=1e-4)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    exit()
    
    true_values_npy = np.random.randn(1, 5,5)
    print(true_values_npy)
    missing_npy = true_values_npy.copy()
    missing_npy[np.random.rand(*missing_npy.shape) < 0.2] = np.nan
    print(missing_npy)
    true_values = torch.from_numpy(true_values_npy).to('cuda').to(torch.bfloat16)
    
    test_batch = torch.from_numpy(missing_npy).to('cuda').to(torch.bfloat16)
    
    for i in range(30):
        opt.zero_grad()
        
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        preds = model(test_batch)
        loss = bar_distribution(logits=preds, y=true_values)
        missing_loss = loss[torch.isnan(test_batch)].mean()
        
        missing_loss.backward()
        opt.step()
        
        print(f"Step {i}: , Missing loss: {missing_loss.item():.4f}")