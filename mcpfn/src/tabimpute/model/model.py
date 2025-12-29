import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm
import os
import importlib.resources as resources
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from torch.optim import AdamW

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
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = FeatureEncoder(embedding_size)
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
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        return self.decoder(src)

class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """ Creates the embedding layer that we will use to embed our features. """
        super().__init__()
        self.observed_linear_layer = nn.Linear(1, embedding_size)
        self.missing_linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            single_eval_pos: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        nan_mask = torch.isnan(x)
        # fill missing values with mean of the training data
        mean = torch_nanmean(x, nan_mask=nan_mask, axis=1)
        
        # fill missing values with mean of the training data along axis 1
        # Expand mean from (batch_size, num_features) to (batch_size, num_rows, num_features)
        mean_expanded = mean.unsqueeze(1).expand_as(x)
        x = torch.where(nan_mask, mean_expanded, x)
        
        embedded_observed = self.observed_linear_layer(x.unsqueeze(-1)) # (batch_size, num_rows, num_features, embedding_size)
        embedded_missing = self.missing_linear_layer(x.unsqueeze(-1)) # (batch_size, num_rows, num_features, embedding_size)
        
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(embedded_observed).to(torch.float32)
        
        return embedded_observed * (1-nan_mask_expanded) + embedded_missing * nan_mask_expanded
        
    
class MaskEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x.unsqueeze(-1))

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
        # attention between features
        src = src.reshape(batch_size*rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0]+src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size*col_size, rows_size, embedding_size)
        # training data attends to itself
        src_left = self.self_attention_between_datapoints(src, src, src)[0]
        src = src_left+src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = self.linear2(self.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src

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
    model = TabImputeModel(
        embedding_size=24,  # Increased from 256
        num_attention_heads=8,  # Increased from 8
        mlp_hidden_size=256,  # Increased from 512
        num_layers=12,  # Increased from 12
        num_outputs=5000,
    ).to('cuda')
    
    with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as path:
        borders = torch.load(path).to(torch.device('cuda'))
        bar_distribution = FullSupportBarDistribution(borders=borders)
    
    model.train()
    opt = AdamW(model.parameters(), lr=1e-3)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    true_values = torch.randn(1, 10, 10).to('cuda')
    
    test_batch = true_values.clone()
    
    rand_indices = torch.randint(0, 10, (10,))
    test_batch[0, :, rand_indices] = torch.nan
    
    for i in range(30):
        opt.zero_grad()
        
        preds = model(test_batch)
        loss = bar_distribution(logits=preds, y=true_values)
        missing_loss = loss[torch.isnan(test_batch)].mean()
        
        missing_loss.backward()
        opt.step()
        
        print(f"Step {i}: , Missing loss: {missing_loss.item()}")