"""Inference-only streaming for the stable model, without shortening attention context.

The hidden grid stays on device, but encoder rows, attention batches, feed-forward
cells, and output distributions are temporary microbatches. Column normalization
precedes the axis transpose, and decoding retains the original strided feature
layout: changing these layouts changes fused-bias rounding under BF16 autocast.
Decoder batches contain at least two rows when available, including an overlapping
tail when needed, so cell_batch_size is a target rather than a strict cell bound.
"""

from importlib import resources

import numpy as np
import torch
from torch.nn import functional as functional

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable


@torch.inference_mode()
def streamed_point_predictions(
    model, values, distribution, *, attention_batch_size=32,
    cell_batch_size=8192, encoder_row_batch_size=32, prediction_stat="mean",
):
    if not isinstance(model, TabImputeModelStable):
        raise TypeError("Streaming requires the stable model")
    if any(module.training for module in model.modules()):
        raise ValueError("Streaming is inference-only; call eval() first")
    if min(attention_batch_size, cell_batch_size, encoder_row_batch_size) <= 0:
        raise ValueError("Streaming batch sizes must be positive")
    if prediction_stat not in {"mean", "median"}:
        raise ValueError("Unsupported prediction statistic")
    encoder = model.feature_encoder
    if encoder.use_absolute_positional_embeddings:
        raise ValueError("Streaming does not support absolute positional embeddings")
    if any(attention.capture_attention_weights for block in model.transformer_blocks
           for attention in (block.self_attention_between_features, block.self_attention_between_datapoints)):
        raise ValueError("Disable attention capture before streaming")
    if model.decoder is None or values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("A decoder and a nonempty batch of matrices are required")
    batch_size, rows, columns = values.shape
    cls_rows = encoder._compute_num_cls(rows)
    cls_columns = encoder._compute_num_cls(columns)
    hidden = None
    for start in range(0, rows, encoder_row_batch_size):
        stop = min(start + encoder_row_batch_size, rows)
        encoded = encoder(values[:, start:stop])
        if hidden is None:
            hidden = encoded.new_empty(batch_size, rows + cls_rows, columns + cls_columns, encoded.shape[-1])
            hidden[:, :cls_rows].copy_(encoded[:, :cls_rows])
        hidden[:, cls_rows + start:cls_rows + stop].copy_(encoded[:, cls_rows:])
        del encoded
    for block in model.transformer_blocks:
        for batch in range(batch_size):
            for start in range(0, hidden.shape[1], attention_batch_size):
                current = hidden[batch, start:start + attention_batch_size]
                attended = block.self_attention_between_features(block.norm1(current), rope_start_index=cls_columns)
                current.copy_(current + block.feature_drop_path(block.feature_residual_scale * attended))
                del attended
            for start in range(0, hidden.shape[2], attention_batch_size):
                current = hidden[batch, :, start:start + attention_batch_size]
                normalized = block.norm2(current).transpose(0, 1)
                if batch_size > 1:
                    normalized = normalized.contiguous()
                attended = block.self_attention_between_datapoints(normalized, rope_start_index=cls_rows).transpose(0, 1)
                current.copy_(current + block.datapoint_drop_path(block.datapoint_residual_scale * attended))
                del normalized, attended
        flat = hidden.view(-1, hidden.shape[-1])
        for start in range(0, flat.shape[0], cell_batch_size):
            current = flat[start:start + cell_batch_size]
            normalized = block.norm3(current)
            projected = functional.silu(block.ff_gate(normalized)) * block.ff_value(normalized)
            projected = block.ff_out(block.ff_dropout(projected))
            current.copy_(current + block.ffn_drop_path(block.ffn_residual_scale * projected))
            del normalized, projected
    predictions = torch.empty(batch_size, rows, columns, device=values.device, dtype=torch.float32)
    decode_rows = max(2, cell_batch_size // columns)
    for batch in range(batch_size):
        for start in range(0, rows, decode_rows):
            stop = min(start + decode_rows, rows)
            if stop - start == 1 and rows > 1:
                start -= 1
            features = hidden[batch:batch + 1, cls_rows + start:cls_rows + stop, cls_columns:]
            logits = model.decoder(features)
            with torch.autocast(device_type=values.device.type, enabled=False):
                predictions[batch:batch + 1, start:stop] = getattr(distribution, prediction_stat)(logits.float())
            del logits
    return predictions


def overlapping_spans(size, limit):
    if size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("Matrix and block sizes must be positive")
    if limit is None or limit >= size:
        return [(0, size)]
    spans = [(start, start + limit) for start in range(0, size - limit, limit)]
    spans.append((size - limit, size))
    return spans


class StreamedImputePFN:
    """Standalone point-imputation adapter; no dependency on research interface edits."""

    def __init__(self, *, device, checkpoint_path, json_config, max_num_rows=None, max_num_cols=None,
                 prediction_stat="mean", precision_mode="amp_bfloat16", model_family="new_stable",
                 entry_wise_features=False, max_num_chunks=1, preprocessors=None, verbose=False,
                 attention_batch_size=32, cell_batch_size=8192, encoder_row_batch_size=32):
        if entry_wise_features or model_family != "new_stable" or max_num_chunks != 1 or preprocessors is not None:
            raise ValueError("Streaming supports the stable model, one context chunk, and no preprocessing ensemble")
        if prediction_stat not in {"mean", "median"}:
            raise ValueError("Unsupported prediction statistic")
        if precision_mode not in {"float32", "amp_bfloat16", "amp_float16", "full_bfloat16"}:
            raise ValueError("Unsupported precision mode")
        self.device = device
        self.precision_mode = precision_mode
        self.prediction_stat = prediction_stat
        self.direct_dtype = torch.bfloat16 if precision_mode == "full_bfloat16" else torch.float32
        self.max_num_rows, self.max_num_cols = max_num_rows, max_num_cols
        self.model = TabImputeModelStable(**json_config["config"]).to(device=device, dtype=self.direct_dtype).eval()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as handle:
            borders = torch.load(handle, map_location=device, weights_only=True)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.streaming_options = {
            "attention_batch_size": attention_batch_size,
            "cell_batch_size": cell_batch_size,
            "encoder_row_batch_size": encoder_row_batch_size,
        }

    @torch.inference_mode()
    def _stream_points(self, values):
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.float16 if self.precision_mode == "amp_float16" else torch.bfloat16,
            enabled=self.precision_mode.startswith("amp_"),
        ):
            return streamed_point_predictions(
                self.model, values.to(self.direct_dtype), self.bar_distribution,
                prediction_stat=self.prediction_stat, **self.streaming_options,
            )

    def _run_model_direct(self, values, last_values):
        points = self._stream_points(values)
        last_points = None if last_values is None else self._stream_points(last_values)
        return points, last_points

    def impute(self, matrix, *, means=None, stds=None):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or min(matrix.shape) <= 0 or np.isinf(matrix).any():
            raise ValueError("Expected a nonempty matrix with finite observations or NaN")
        means = np.nanmean(matrix, axis=0) if means is None else np.asarray(means, dtype=np.float64)
        stds = np.nanstd(matrix, axis=0) if stds is None else np.asarray(stds, dtype=np.float64)
        if means.shape != (matrix.shape[1],) or stds.shape != means.shape:
            raise ValueError("Normalization arrays must match the item columns")
        means = np.where(np.isnan(means), 0, means)
        stds = np.where(np.isnan(stds), 1, stds)
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds < 0).any():
            raise ValueError("Invalid normalization statistics")
        normalized = (matrix - means) / (stds + 1e-16)
        totals = np.zeros_like(normalized)
        counts = np.zeros(matrix.shape, dtype=np.uint8)
        for row_start, row_stop in overlapping_spans(matrix.shape[0], self.max_num_rows):
            for col_start, col_stop in overlapping_spans(matrix.shape[1], self.max_num_cols):
                selection = np.s_[row_start:row_stop, col_start:col_stop]
                block = normalized[selection]
                values = torch.from_numpy(block.copy()).to(device=self.device, dtype=self.direct_dtype).unsqueeze(0)
                points = self._stream_points(values)[0].cpu().numpy()
                totals[selection] += np.where(np.isnan(block), points, block)
                counts[selection] += 1
        completed = (totals / counts) * (stds + 1e-16) + means
        completed[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
        return completed


class DenseReferenceImputer(StreamedImputePFN):
    @torch.inference_mode()
    def _stream_points(self, values):
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.float16 if self.precision_mode == "amp_float16" else torch.bfloat16,
            enabled=self.precision_mode.startswith("amp_"),
        ):
            logits = self.model(values.to(self.direct_dtype))
        return getattr(self.bar_distribution, self.prediction_stat)(logits.float())
