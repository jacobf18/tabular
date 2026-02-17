import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm
import os
import importlib.resources as resources
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from torch.optim import AdamW
from tabimpute.model.positional import SinusoidalRowEmbedding, SinusoidalColumnEmbedding, LinearPositionalEmbedding

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int, 
                 num_cls: int):
        """ 
        Initializes the feature/target encoder, transformer stack and decoder.
        
        Args:
            embedding_size: Size of the embedding vector
            num_attention_heads: Number of attention heads
            mlp_hidden_size: Hidden size for MLP layers
            num_layers: Number of transformer layers
            num_outputs: Number of output classes
            num_cls: Fixed number of CLS tokens to use for both rows and columns.
        """
        super().__init__()
        
        self.num_cls = num_cls
        self.feature_encoder = FeatureEncoder(embedding_size, num_cls)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        
        src = self.feature_encoder(src)

        # repeatedly applies the transformer block on (B,R,C,E)
        for block in self.transformer_blocks:
            src = block(src)
        
        # Use the CLS token counts that FeatureEncoder computed and stored
        num_cls_rows = self.feature_encoder.last_num_cls_rows
        num_cls_cols = self.feature_encoder.last_num_cls_cols
        
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        return self.decoder(src[:,num_cls_rows:,num_cls_cols:,:]) # remove the cls tokens

class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int, num_cls: int):
        """ 
        Creates the embedding layer with a fixed number of CLS tokens.
        
        Args:
            embedding_size: Size of the embedding vector
            num_cls: Fixed number of CLS tokens to use for both rows and columns.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.num_cls = num_cls
        
        # Store the last computed CLS token counts (set during forward pass)
        self.last_num_cls_rows = None
        self.last_num_cls_cols = None

        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size)
        )
        
        # Note: Damping factor 0.1 helps prevent embeddings from drowning out the data
        self.row_embedding = SinusoidalRowEmbedding(embedding_size, damping_factor=0.1)
        self.column_embedding = SinusoidalColumnEmbedding(embedding_size, damping_factor=0.1)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        
        # Store fixed CLS tokens as embeddings
        # 1. Row CLS Tokens (Left Side)
        self.row_cls_embedding = nn.Embedding(num_cls, embedding_size)
        
        # 2. Col CLS Tokens (Top Side)
        self.col_cls_embedding = nn.Embedding(num_cls, embedding_size)
        
        # 3. Corner Tokens (Top-Left Intersection) - stored as 2D embedding
        # Shape: (max_cls, max_cls, E) -> can be interpolated to any (K_rows, K_cols)
        # We'll use a combination of row and col embeddings for the corner
        # For simplicity, we can use outer product or just tile row/col embeddings

    def _compute_num_cls(self, size: int) -> int:
        """Return fixed number of CLS tokens."""
        return self.num_cls

    def _get_cls_tokens(self, num_cls: int, embedding: nn.Embedding, device: torch.device, dtype: torch.dtype):
        """
        Get CLS tokens of the requested size, interpolating or tiling as needed.
        
        Args:
            num_cls: Desired number of CLS tokens
            embedding: The embedding layer to use
            device: Device to create tokens on
            dtype: Data type for tokens
            
        Returns:
            Tensor of shape (num_cls, embedding_size)
        """
        indices = torch.arange(num_cls, device=device, dtype=torch.long)
        return embedding(indices).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (torch.Tensor) shape (batch_size, num_rows, num_features)
        Returns:
            (torch.Tensor) shape (batch_size, num_rows + K_rows, num_features + K_cols, embedding_size)
        """
        # --- 1. Embed Data ---
        nan_mask = torch.isnan(x)
        # Preserve dtype to avoid promoting to float32
        x = torch.where(nan_mask, torch.zeros_like(x), x)
        
        embedded_observed = self.observed_linear_layer(x.unsqueeze(-1))
        
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(embedded_observed)
        
        B, R, C = x.shape
        mask_expanded = self.mask_token.expand(B, R, C, self.embedding_size)
        embedded = torch.where(nan_mask_expanded, mask_expanded, embedded_observed)
        
        # Add positions (only to the data part)
        embedded = self.column_embedding(embedded) + self.row_embedding(embedded)
        
        # --- 2. Compute CLS token counts (fixed or dynamic) ---
        # K_rows: number of CLS tokens for the TOP (scales with number of rows if dynamic, fixed if num_cls is set)
        # K_cols: number of CLS tokens for the LEFT (scales with number of columns if dynamic, fixed if num_cls is set)
        K_rows = self._compute_num_cls(R)
        K_cols = self._compute_num_cls(C)
        
        # Store the computed counts so TabImputeModel can use them without recomputing
        self.last_num_cls_rows = K_rows
        self.last_num_cls_cols = K_cols
        
        # In fixed mode, both K_rows and K_cols will be equal to num_cls
        
        # --- 3. Generate CLS Tokens ---
        device = x.device
        dtype = embedded.dtype
        
        # Row CLS tokens go on the LEFT side (before columns), so use K_cols
        # Shape: (K_cols, E) -> (B, R, K_cols, E)
        row_cls_tokens = self._get_cls_tokens(K_cols, self.row_cls_embedding, device, dtype)
        row_cls = row_cls_tokens.unsqueeze(0).unsqueeze(0).expand(B, R, K_cols, self.embedding_size)
        
        # Col CLS tokens go on the TOP side (before rows), so use K_rows
        # Shape: (K_rows, E) -> (B, K_rows, C, E)
        col_cls_tokens = self._get_cls_tokens(K_rows, self.col_cls_embedding, device, dtype)
        col_cls = col_cls_tokens.unsqueeze(0).unsqueeze(2).expand(B, K_rows, C, self.embedding_size)
        
        # Corner tokens: Use combination of row and col CLS tokens
        # row_cls_tokens are for LEFT (K_cols), col_cls_tokens are for TOP (K_rows)
        # Corner shape: (B, K_rows, K_cols, E)
        # For corner[i, j]: combine col_cls_tokens[i] (top) with row_cls_tokens[j] (left)
        row_corner = row_cls_tokens.unsqueeze(0).expand(K_rows, K_cols, self.embedding_size)  # (K_rows, K_cols, E)
        col_corner = col_cls_tokens.unsqueeze(1).expand(K_rows, K_cols, self.embedding_size)  # (K_rows, K_cols, E)
        # Combine them (using addition for simplicity)
        corner = (row_corner + col_corner).unsqueeze(0).expand(B, K_rows, K_cols, self.embedding_size)
        
        # --- 4. Concatenate Grid ---
        
        # Top Strip: [Corner | Col_CLS] -> Shape (B, K_rows, K_cols+C, E)
        top_block = torch.cat([corner, col_cls], dim=2)
        
        # Bottom Strip: [Row_CLS | Data] -> Shape (B, R, K_cols+C, E)
        bottom_block = torch.cat([row_cls, embedded], dim=2)
        
        # Full Grid: [Top / Bottom] -> Shape (B, K_rows+R, K_cols+C, E)
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
        attn_output = self.self_attention_between_features(src_flat, src_flat, src_flat, need_weights=False)
        src_att = attn_output[0]
        # # Store attention weights: shape is (batch*rows, num_heads, seq_len, seq_len) or (batch*rows, seq_len, seq_len)
        # if len(attn_output) > 1 and attn_output[1] is not None:
        #     self._last_feature_attention_weights = attn_output[1]
        src_att = src_att.reshape(batch_size, rows_size, col_size, embedding_size)
        
        src = src + src_att # Residual connection
        
        # --- BLOCK 2: Datapoint Attention (Pre-Norm) ---
        src_norm = self.norm2(src) # Norm BEFORE Attention
        
        # Reshape for Datapoint Attention
        # We treat (Batch * Cols) as the "Batch" dimension for the attention module
        # and 'Rows' as the sequence length.
        src_t = src_norm.transpose(1, 2)
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
        attn_output = self.self_attention_between_datapoints(src_flat, src_flat, src_flat, need_weights=False)
        src_att = attn_output[0]
        # # Store attention weights
        # if len(attn_output) > 1 and attn_output[1] is not None:
        #     self._last_datapoint_attention_weights = attn_output[1]
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
    ).to(torch.bfloat16).to('cuda')
    model.train()
    # model = torch.compile(model)
    
    with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as path:
        borders = torch.load(path).to(torch.device('cuda'))
        bar_distribution = FullSupportBarDistribution(borders=borders)
    
    # Keep borders in float32 for numerical stability (they're buffers, not trainable)
    # The bar_distribution can accept bfloat16 logits while keeping borders in float32
    
    opt = AdamW(model.parameters(), lr=1e-4)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # exit()
    
    size = (20, 20)
    true_values_npy = np.random.randn(1, size[0], size[1])
    print(true_values_npy)
    missing_npy = true_values_npy.copy()
    missing_npy[np.random.rand(*missing_npy.shape) < 0.2] = np.nan
    print(missing_npy)
    true_values = torch.from_numpy(true_values_npy).to('cuda').to(torch.bfloat16)
    
    test_batch = torch.from_numpy(missing_npy).to('cuda').to(torch.bfloat16)
    
    import time
    
    time_start = time.time()
    for i in range(50):
        opt.zero_grad()
        
        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        preds = model(test_batch)
        loss = bar_distribution(logits=preds, y=true_values)
        missing_loss = loss[torch.isnan(test_batch)].mean()
        
        missing_loss.backward()
        opt.step()
        
        print(f"Step {i}: , Missing loss: {missing_loss.item():.4f}")
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")