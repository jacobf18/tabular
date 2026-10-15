"""Experimental gradient-safe microbatching with full axial attention context.

Unlike inference streaming, this retains differentiable hidden grids and uses
recomputation, not in-place overwrites. The loss callable must return one loss
per target and must not mutate model state. Dropout draws can differ from dense
execution, but checkpoint recomputation preserves the chunked execution's RNG.
"""

from functools import partial

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from tabimpute.model.model_new_stable import TabImputeModelStable


def _recompute(function, *arguments, enabled):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *arguments, use_reentrant=False, preserve_rng_state=True)
    return function(*arguments)


def _streamed_block(hidden, *, block, cls_rows, cls_columns, attention_batch_size,
                    cell_batch_size, use_checkpoint):
    def feature_attention(current):
        return block.self_attention_between_features(
            block.norm1(current), rope_start_index=cls_columns,
        )

    def datapoint_attention(current):
        normalized = block.norm2(current).transpose(0, 1)
        return block.self_attention_between_datapoints(
            normalized, rope_start_index=cls_rows,
        ).transpose(0, 1)

    def feed_forward(current):
        normalized = block.norm3(current)
        projected = functional.silu(block.ff_gate(normalized)) * block.ff_value(normalized)
        return block.ff_out(block.ff_dropout(projected))

    attended = torch.stack([
        torch.cat([
            _recompute(feature_attention, matrix[start:start + attention_batch_size],
                       enabled=use_checkpoint)
            for start in range(0, matrix.shape[0], attention_batch_size)
        ], dim=0)
        for matrix in hidden.unbind(0)
    ], dim=0)
    hidden = hidden + block.feature_drop_path(block.feature_residual_scale * attended)
    attended = torch.stack([
        torch.cat([
            _recompute(datapoint_attention, matrix[:, start:start + attention_batch_size],
                       enabled=use_checkpoint)
            for start in range(0, matrix.shape[1], attention_batch_size)
        ], dim=1)
        for matrix in hidden.unbind(0)
    ], dim=0)
    hidden = hidden + block.datapoint_drop_path(block.datapoint_residual_scale * attended)
    projected = torch.cat([
        _recompute(feed_forward, current, enabled=use_checkpoint)
        for current in hidden.reshape(-1, hidden.shape[-1]).split(cell_batch_size)
    ], dim=0).reshape_as(hidden)
    return hidden + block.ffn_drop_path(block.ffn_residual_scale * projected)


def streamed_training_loss(model, values, targets, decode_mask, loss_function, *,
                           attention_batch_size=32, cell_batch_size=8192,
                           encoder_row_batch_size=32, use_checkpoint=True,
                           auxiliary_targets=None, decoder_cell_batch_size=None):
    """Return the mean masked loss without constructing a full output distribution.

    All selected cells contribute, including uneven final microbatches. Attention
    microbatch sizes bound independent sequences, never attended sequence length.
    Both block and microbatch checkpointing bound retained intermediates; full
    hidden grids and their gradients still occupy memory. No backward or optimizer
    step is performed here. Absolute positional embeddings are not supported.
    Optional auxiliary targets are detached and passed as the loss's third argument,
    aligned with the same selected cells in every checkpoint recomputation.
    """
    if not isinstance(model, TabImputeModelStable):
        raise TypeError("Streamed training requires the stable model")
    if min(attention_batch_size, cell_batch_size, encoder_row_batch_size) <= 0:
        raise ValueError("Microbatch sizes must be positive")
    if values.ndim != 3 or min(values.shape) <= 0 or targets.shape != values.shape:
        raise ValueError("Values and targets must be matching nonempty matrix batches")
    if decode_mask.dtype != torch.bool or decode_mask.shape != values.shape or not decode_mask.any():
        raise ValueError("A nonempty boolean decode mask matching values is required")
    if not torch.isfinite(targets[decode_mask]).all():
        raise ValueError("Selected targets must be finite")
    if auxiliary_targets is not None:
        if auxiliary_targets.shape != targets.shape or auxiliary_targets.device != targets.device:
            raise ValueError("Auxiliary targets must match the target grid and device")
        if not torch.isfinite(auxiliary_targets[decode_mask]).all():
            raise ValueError("Selected auxiliary targets must be finite")
        auxiliary_targets = auxiliary_targets.detach()
    decoder_cell_batch_size = cell_batch_size if decoder_cell_batch_size is None else decoder_cell_batch_size
    if decoder_cell_batch_size <= 0:
        raise ValueError("Decoder microbatch size must be positive")
    if model.feature_encoder.use_absolute_positional_embeddings:
        raise ValueError("Absolute positional embeddings are unsupported")
    if model.decoder is None:
        raise ValueError("A decoder is required")
    if any(attention.capture_attention_weights for block in model.transformer_blocks
           for attention in (block.self_attention_between_features, block.self_attention_between_datapoints)):
        raise ValueError("Disable attention capture before streamed training")

    encoder = model.feature_encoder
    cls_rows = encoder._compute_num_cls(values.shape[1])
    cls_columns = encoder._compute_num_cls(values.shape[2])
    encoded_rows = []
    for start in range(0, values.shape[1], encoder_row_batch_size):
        encoded = _recompute(encoder, values[:, start:start + encoder_row_batch_size],
                             enabled=use_checkpoint)
        encoded_rows.append(encoded if start == 0 else encoded[:, cls_rows:])
    hidden = torch.cat(encoded_rows, dim=1)
    del encoded, encoded_rows
    for block in model.transformer_blocks:
        run_block = partial(
            _streamed_block, block=block, cls_rows=cls_rows, cls_columns=cls_columns,
            attention_batch_size=attention_batch_size, cell_batch_size=cell_batch_size,
            use_checkpoint=use_checkpoint,
        )
        hidden = _recompute(run_block, hidden, enabled=use_checkpoint)

    features = hidden[:, cls_rows:, cls_columns:][decode_mask]
    selected_targets = targets[decode_mask]

    def decode_loss(current, truth, *auxiliary):
        logits = model.decoder(current)
        with torch.autocast(device_type=values.device.type, enabled=False):
            losses = loss_function(logits.float(), truth, *auxiliary)
            if losses.shape != truth.shape:
                raise ValueError("The loss function must return one loss per selected target")
            return losses.sum()

    chunks = [features.split(decoder_cell_batch_size), selected_targets.split(decoder_cell_batch_size)]
    if auxiliary_targets is not None:
        chunks.append(auxiliary_targets[decode_mask].split(decoder_cell_batch_size))
    losses = [_recompute(decode_loss, *arguments, enabled=use_checkpoint) for arguments in zip(*chunks)]
    return torch.stack(losses).sum() / selected_targets.numel()
