import torch
import torch.nn.functional as F
from torch import nn

from tabimpute.model.positional import SinusoidalColumnEmbedding, SinusoidalRowEmbedding

try:
    from torchvision.ops import StochasticDepth as TorchvisionStochasticDepth
except ImportError:
    TorchvisionStochasticDepth = None


BEST_TRANSFER_MODEL_CONFIG = {
    # Centered on the downstream-best trial12/local14 family.
    "embedding_size": 768,
    "num_attention_heads": 16,
    "mlp_hidden_size": 1536,
    "num_layers": 8,
    "num_cls": 12,
    "use_rope": True,
    "rope_base": 10000.0,
    "rope_fraction": 0.75,
    "use_absolute_positional_embeddings": False,
    "positional_damping_factor": 0.1,
    # The first stable run was too damped for 8 layers, so keep regularization
    # present but materially lighter for the next anchor.
    "attention_dropout": 0.01,
    "ffn_dropout": 0.05,
    "drop_path_rate": 0.02,
    "residual_scale_init": 0.20,
    "rms_norm_eps": 1e-6,
    "embedding_dropout": 0.0,
}

BEST_TRANSFER_OPTIMIZER_CONFIG = {
    "optimizer_name": "adamw",
    "scheduler_name": "warmup_cosine",
    "lr": 2e-4,
    "weight_decay": 0.03,
    "grad_clip_norm": 0.5,
    "warmup_ratio": 0.06,
    "min_lr_ratio": 0.05,
}


RMSNorm = nn.RMSNorm


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.impl = None
        if TorchvisionStochasticDepth is not None:
            self.impl = TorchvisionStochasticDepth(drop_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl is not None:
            return self.impl(x)

        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=torch.float32).bernoulli_(
            keep_prob
        )
        if keep_prob > 0.0:
            mask.div_(keep_prob)
        return x * mask.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE cache for a single axis."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_to_subset(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_index: int,
    rotary_dim: int,
) -> torch.Tensor:
    if start_index >= x.size(2):
        return x

    prefix = x[:, :, :start_index, :]
    target = x[:, :, start_index:, :]

    target_rot = target[..., :rotary_dim]
    target_pass = target[..., rotary_dim:]

    rotated = (target_rot * cos) + (_rotate_half(target_rot) * sin)
    tail = torch.cat([rotated, target_pass], dim=-1)
    return torch.cat([prefix, tail], dim=2) if start_index > 0 else tail


class AxialRoPESelfAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
    ):
        super().__init__()
        if embedding_size % nhead != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by nhead ({nhead})"
            )

        self.embedding_size = embedding_size
        self.nhead = nhead
        self.head_dim = embedding_size // nhead
        self.dropout = dropout
        self.use_rope = use_rope

        rotary_dim = int(self.head_dim * rope_fraction)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rotary_dim = max(0, min(self.head_dim, rotary_dim))
        self.rope = (
            RotaryEmbedding(self.rotary_dim, base=rope_base)
            if self.use_rope and self.rotary_dim > 0
            else None
        )

        self.q_proj = nn.Linear(embedding_size, embedding_size)
        self.k_proj = nn.Linear(embedding_size, embedding_size)
        self.v_proj = nn.Linear(embedding_size, embedding_size)
        self.out_proj = nn.Linear(embedding_size, embedding_size)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.nhead, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, rope_start_index: int = 0) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        if self.rope is not None and rope_start_index < q.size(2):
            rope_seq_len = q.size(2) - rope_start_index
            cos, sin = self.rope.get_cos_sin(rope_seq_len, q.device, q.dtype)
            q = _apply_rope_to_subset(q, cos, sin, rope_start_index, self.rotary_dim)
            k = _apply_rope_to_subset(k, cos, sin, rope_start_index, self.rotary_dim)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            x.size(0), x.size(1), self.embedding_size
        )
        return self.out_proj(attn_out)


class FeatureEncoderStable(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_cls: int,
        use_absolute_positional_embeddings: bool = False,
        positional_damping_factor: float = 0.1,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_cls = num_cls
        self.use_absolute_positional_embeddings = use_absolute_positional_embeddings

        self.last_num_cls_rows = None
        self.last_num_cls_cols = None

        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
        )

        if self.use_absolute_positional_embeddings:
            self.row_embedding = SinusoidalRowEmbedding(
                embedding_size, damping_factor=positional_damping_factor
            )
            self.column_embedding = SinusoidalColumnEmbedding(
                embedding_size, damping_factor=positional_damping_factor
            )
        else:
            self.row_embedding = None
            self.column_embedding = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        self.missingness_embedding = nn.Embedding(2, embedding_size)
        self.row_cls_embedding = nn.Embedding(num_cls, embedding_size)
        self.col_cls_embedding = nn.Embedding(num_cls, embedding_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def _compute_num_cls(self, _: int) -> int:
        return self.num_cls

    def _get_cls_tokens(
        self,
        num_cls: int,
        embedding: nn.Embedding,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        indices = torch.arange(num_cls, device=device, dtype=torch.long)
        return embedding(indices).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x_clean = torch.where(nan_mask, torch.zeros_like(x), x)

        embedded_observed = self.observed_linear_layer(x_clean.unsqueeze(-1))
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(embedded_observed)

        batch_size, rows, cols = x.shape
        mask_expanded = self.mask_token.expand(batch_size, rows, cols, self.embedding_size)
        embedded = torch.where(nan_mask_expanded, mask_expanded, embedded_observed)

        missingness_ids = nan_mask.to(torch.long)
        embedded = embedded + self.missingness_embedding(missingness_ids)

        if self.row_embedding is not None and self.column_embedding is not None:
            embedded = self.row_embedding(embedded)
            embedded = self.column_embedding(embedded)

        k_rows = self._compute_num_cls(rows)
        k_cols = self._compute_num_cls(cols)
        self.last_num_cls_rows = k_rows
        self.last_num_cls_cols = k_cols

        device = x.device
        dtype = embedded.dtype

        row_cls_tokens = self._get_cls_tokens(k_cols, self.row_cls_embedding, device, dtype)
        row_cls = row_cls_tokens.unsqueeze(0).unsqueeze(0).expand(
            batch_size, rows, k_cols, self.embedding_size
        )

        col_cls_tokens = self._get_cls_tokens(k_rows, self.col_cls_embedding, device, dtype)
        col_cls = col_cls_tokens.unsqueeze(0).unsqueeze(2).expand(
            batch_size, k_rows, cols, self.embedding_size
        )

        row_corner = row_cls_tokens.unsqueeze(0).expand(k_rows, k_cols, self.embedding_size)
        col_corner = col_cls_tokens.unsqueeze(1).expand(k_rows, k_cols, self.embedding_size)
        corner = (row_corner + col_corner).unsqueeze(0).expand(
            batch_size, k_rows, k_cols, self.embedding_size
        )

        top_block = torch.cat([corner, col_cls], dim=2)
        bottom_block = torch.cat([row_cls, embedded], dim=2)
        encoded = torch.cat([top_block, bottom_block], dim=1)
        return self.embedding_dropout(encoded)


class TransformerEncoderLayerStable(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path: float = 0.0,
        residual_scale_init: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
    ):
        super().__init__()
        self.self_attention_between_features = AxialRoPESelfAttention(
            embedding_size=embedding_size,
            nhead=nhead,
            dropout=attention_dropout,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
        )
        self.self_attention_between_datapoints = AxialRoPESelfAttention(
            embedding_size=embedding_size,
            nhead=nhead,
            dropout=attention_dropout,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
        )

        self.norm1 = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.norm2 = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.norm3 = RMSNorm(embedding_size, eps=rms_norm_eps)

        self.ff_gate = nn.Linear(embedding_size, mlp_hidden_size)
        self.ff_value = nn.Linear(embedding_size, mlp_hidden_size)
        self.ff_out = nn.Linear(mlp_hidden_size, embedding_size)
        self.ff_dropout = nn.Dropout(ffn_dropout)

        self.feature_drop_path = DropPath(drop_path)
        self.datapoint_drop_path = DropPath(drop_path)
        self.ffn_drop_path = DropPath(drop_path)

        self.feature_residual_scale = nn.Parameter(
            torch.full((1,), residual_scale_init)
        )
        self.datapoint_residual_scale = nn.Parameter(
            torch.full((1,), residual_scale_init)
        )
        self.ffn_residual_scale = nn.Parameter(torch.full((1,), residual_scale_init))

    def forward(
        self,
        src: torch.Tensor,
        num_cls_rows: int,
        num_cls_cols: int,
    ) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        src_norm = self.norm1(src)
        src_flat = src_norm.reshape(batch_size * rows_size, col_size, embedding_size)
        src_att = self.self_attention_between_features(
            src_flat,
            rope_start_index=num_cls_cols,
        )
        src_att = src_att.reshape(batch_size, rows_size, col_size, embedding_size)
        src = src + self.feature_drop_path(self.feature_residual_scale * src_att)

        src_norm = self.norm2(src)
        src_t = src_norm.transpose(1, 2)
        src_flat = src_t.reshape(batch_size * col_size, rows_size, embedding_size)
        src_att = self.self_attention_between_datapoints(
            src_flat,
            rope_start_index=num_cls_rows,
        )
        src_att = src_att.reshape(batch_size, col_size, rows_size, embedding_size)
        src_att = src_att.transpose(2, 1)
        src = src + self.datapoint_drop_path(
            self.datapoint_residual_scale * src_att
        )

        src_norm = self.norm3(src)
        src_mlp = F.silu(self.ff_gate(src_norm)) * self.ff_value(src_norm)
        src_mlp = self.ff_dropout(src_mlp)
        src_mlp = self.ff_out(src_mlp)
        return src + self.ffn_drop_path(self.ffn_residual_scale * src_mlp)


class DecoderStable(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(embedding_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, num_outputs),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_drop_path_schedule(num_layers: int, drop_path_rate: float) -> list[float]:
    if num_layers <= 0:
        return []
    if num_layers == 1:
        return [drop_path_rate]
    return torch.linspace(0.0, drop_path_rate, steps=num_layers).tolist()


class TabImputeModelStable(nn.Module):
    """
    Drop-in sibling of model_new with:
    - RMSNorm in place of LayerNorm
    - attention dropout + FFN dropout
    - per-layer stochastic depth
    - learned residual branch scaling
    """

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        num_cls: int,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_absolute_positional_embeddings: bool = False,
        positional_damping_factor: float = 0.1,
        attention_dropout: float = 0.05,
        ffn_dropout: float = 0.10,
        drop_path_rate: float = 0.05,
        residual_scale_init: float = 0.10,
        rms_norm_eps: float = 1e-6,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.feature_encoder = FeatureEncoderStable(
            embedding_size=embedding_size,
            num_cls=num_cls,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            embedding_dropout=embedding_dropout,
        )

        drop_path_rates = _build_drop_path_schedule(num_layers, drop_path_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderLayerStable(
                    embedding_size=embedding_size,
                    nhead=num_attention_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    drop_path=drop_path_rates[layer_idx],
                    residual_scale_init=residual_scale_init,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    rope_fraction=rope_fraction,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.decoder = DecoderStable(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.feature_encoder(src)

        num_cls_rows = self.feature_encoder.last_num_cls_rows
        num_cls_cols = self.feature_encoder.last_num_cls_cols

        for block in self.transformer_blocks:
            src = block(src, num_cls_rows=num_cls_rows, num_cls_cols=num_cls_cols)

        return self.decoder(src[:, num_cls_rows:, num_cls_cols:, :])


class TabImputeModel(TabImputeModelStable):
    """Drop-in class name compatibility for training code imports."""


def build_best_transfer_model(num_outputs: int = 5000, **overrides) -> TabImputeModel:
    model_cfg = dict(BEST_TRANSFER_MODEL_CONFIG)
    model_cfg.update(overrides)
    return TabImputeModel(num_outputs=num_outputs, **model_cfg)


__all__ = [
    "BEST_TRANSFER_MODEL_CONFIG",
    "BEST_TRANSFER_OPTIMIZER_CONFIG",
    "TabImputeModelStable",
    "TabImputeModel",
    "build_best_transfer_model",
]
