"""Transient cell queries with observed row/column cross-attention."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class CellQueryLayer(nn.Module):
    def __init__(self, width, heads, axis_balanced=False):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.kv = nn.Linear(width, 2 * width, bias=False)
        self.output = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.axis_balanced = axis_balanced
        if axis_balanced:
            self.axis_gate = nn.Linear(width, 3)
            nn.init.zeros_(self.axis_gate.weight)
            nn.init.zeros_(self.axis_gate.bias)
            self.axis_blend = nn.Parameter(torch.zeros(()))

    def forward(self, query, memory, allowed, row_length=None):
        n, length, width = memory.shape
        q = self.q(self.query_norm(query)).view(n, self.heads, 1, width // self.heads)
        key, value = self.kv(self.memory_norm(memory)).chunk(2, -1)
        key = key.reshape(n, length, self.heads, -1).transpose(1, 2)
        value = value.reshape(n, length, self.heads, -1).transpose(1, 2)
        context = F.scaled_dot_product_attention(q, key, value, attn_mask=allowed[:, None, None, :])
        if self.axis_balanced:
            if row_length is None or not 0 < row_length < length - 2:
                raise ValueError("Separate-axis attention needs row and column memory lengths")
            contexts, available = [], []
            for start, end in ((0, row_length), (row_length, length - 2), (length - 2, length)):
                mask = allowed[:, start:end]
                valid = mask.any(-1)
                # Supply a dummy key for empty axes, then remove its contribution.
                safe_mask = mask.clone()
                safe_mask[:, 0] |= ~valid
                part = F.scaled_dot_product_attention(q, key[:, :, start:end], value[:, :, start:end],
                    attn_mask=safe_mask[:, None, None, :])
                contexts.append(part * valid[:, None, None, None])
                available.append(valid)
            weights = self.axis_gate(self.query_norm(query)).float().masked_fill(
                ~torch.stack(available, -1), float("-inf")).softmax(-1)
            balanced = sum(part * weights[:, i, None, None, None].to(part.dtype)
                           for i, part in enumerate(contexts))
            # Exact legacy function at warm start; learn the correction continuously.
            context = context + self.axis_blend.tanh() * (balanced - context)
        query = query + self.output(context.reshape(n, width))
        return query + self.ff(self.ff_norm(query))


class CellQueryDecoder(nn.Module):
    def __init__(self, latent_size, feature_size, width=128, heads=4, layers=2,
                 chunk_size=128, checkpoint_chunks=True, axis_balanced=False):
        super().__init__()
        if min(width, heads, layers, chunk_size) <= 0 or width % heads:
            raise ValueError("Positive query dimensions required; width must be divisible by heads")
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.row_proj = nn.Linear(latent_size, width)
        self.col_proj = nn.Linear(latent_size, width)
        self.query_proj = nn.Linear(feature_size, width)
        self.value_proj = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.role = nn.Parameter(torch.zeros(4, width))
        self.layers = nn.ModuleList([CellQueryLayer(width, heads, axis_balanced) for _ in range(layers)])
        self.output = nn.Linear(width, feature_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _chunk(self, features, rows, cols, src, batch, row, col):
        row_values = src[batch, row, :]
        col_values = src[batch, :, col]
        values = torch.cat((row_values, col_values), -1)
        observed = torch.isfinite(values)
        safe = torch.where(observed, values, torch.zeros_like(values)).float()
        compressed = safe.sign() * safe.abs().log1p()
        value_features = torch.stack((safe.clamp(-6, 6), compressed, compressed.abs()), -1)
        value_state = self.value_proj(value_features.to(features.dtype))
        row_memory = cols[batch] + rows[batch, row, None] + self.role[0]
        col_memory = rows[batch] + cols[batch, col, None] + self.role[1]
        memory = torch.cat((row_memory, col_memory), 1) + value_state
        # Always-valid shared tokens also make empty rows/columns well-defined.
        shared = torch.stack((rows.mean(1)[batch] + self.role[2],
                              cols.mean(1)[batch] + self.role[3]), 1)
        memory = torch.cat((memory, shared), 1)
        allowed = F.pad(observed, (0, 2), value=True)
        query = self.query_proj(features)
        for layer in self.layers:
            query = (layer(query, memory, allowed, src.shape[2]) if layer.axis_balanced
                     else layer(query, memory, allowed))
        return self.output(query)

    def forward(self, features, row_state, col_state, src, indices):
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        if not flat.shape[0]:
            return torch.zeros_like(features)
        rows, cols = self.row_proj(row_state), self.col_proj(col_state)
        outputs = []
        for start in range(0, len(flat), self.chunk_size):
            end = start + self.chunk_size
            args = (flat[start:end], rows, cols, src, *(v[start:end] for v in indices))
            if self.training and torch.is_grad_enabled() and self.checkpoint_chunks:
                output = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                output = self._chunk(*args)
            outputs.append(output)
        return torch.cat(outputs, 0).reshape(shape)
