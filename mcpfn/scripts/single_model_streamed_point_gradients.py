"""Differentiate point metrics through the exact checkpointed decoder forward."""

from functools import partial

import torch

from single_model_point_metric_gradients import linearization, point_metric_terms
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.streamed_training import _recompute, _streamed_block


def streamed_point_predictions(model, criterion, values, decode_mask, *, attention_batch_size=32,
                               cell_batch_size=8192, encoder_row_batch_size=32,
                               decoder_cell_batch_size=2048, use_checkpoint=True):
    if not isinstance(model, TabImputeModelStable):
        raise TypeError("Streamed point gradients require the original stable network")
    if min(attention_batch_size, cell_batch_size, encoder_row_batch_size, decoder_cell_batch_size) <= 0:
        raise ValueError("All microbatch sizes must be positive")
    if (values.ndim != 3 or min(values.shape) < 1 or decode_mask.shape != values.shape
            or decode_mask.dtype != torch.bool or decode_mask.device != values.device or not decode_mask.any()):
        raise ValueError("Require nonempty aligned values and a boolean query mask")
    if model.feature_encoder.use_absolute_positional_embeddings or model.decoder is None:
        raise ValueError("Require the original decoder without absolute positional embeddings")
    if any(attention.capture_attention_weights for block in model.transformer_blocks
           for attention in (block.self_attention_between_features, block.self_attention_between_datapoints)):
        raise ValueError("Disable attention capture before streamed point gradients")
    encoder = model.feature_encoder
    cls_rows = encoder._compute_num_cls(values.shape[1])
    cls_columns = encoder._compute_num_cls(values.shape[2])
    encoded_rows = []
    for start in range(0, values.shape[1], encoder_row_batch_size):
        encoded = _recompute(encoder, values[:, start:start + encoder_row_batch_size], enabled=use_checkpoint)
        encoded_rows.append(encoded if start == 0 else encoded[:, cls_rows:])
    hidden = torch.cat(encoded_rows, dim=1)
    del encoded_rows, encoded
    for block in model.transformer_blocks:
        run_block = partial(_streamed_block, block=block, cls_rows=cls_rows, cls_columns=cls_columns,
                            attention_batch_size=attention_batch_size, cell_batch_size=cell_batch_size,
                            use_checkpoint=use_checkpoint)
        hidden = _recompute(run_block, hidden, enabled=use_checkpoint)
    features = hidden[:, cls_rows:, cls_columns:][decode_mask]

    def decode_points(current):
        logits = model.decoder(current)
        with torch.autocast(device_type=values.device.type, enabled=False):
            points = criterion.mean(logits.float())
        if points.shape != current.shape[:-1]:
            raise ValueError("The criterion must produce one point mean per decoded cell")
        return points

    selected = torch.cat([_recompute(decode_points, current, enabled=use_checkpoint)
                          for current in features.split(decoder_cell_batch_size)])
    if not torch.isfinite(selected).all():
        raise FloatingPointError("Nonfinite point predictions")
    return selected.new_zeros(values.shape).masked_scatter(decode_mask, selected)


def anchored_metric_loss(prediction, mask, record):
    linearization(prediction.detach(), mask, record)
    selected = prediction[mask].double()
    return record["value"] + ((selected - selected.detach()) * record["gradient"][mask]).sum()


def streamed_point_metric_losses(model, criterion, episode, **options):
    prediction = streamed_point_predictions(model, criterion, episode.source, episode.mask, **options)
    records = point_metric_terms(prediction.detach(), episode)
    losses = {name: None if record is None else anchored_metric_loss(prediction, episode.mask, record)
              for name, record in records.items()}
    return prediction, records, losses
