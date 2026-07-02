from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from tabimpute.model.model_compressed_cross import make_chunk_spans

_TRAINER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_compressed_cross_rank10.py"
)
_SPEC = importlib.util.spec_from_file_location("train_compressed_cross_rank10", _TRAINER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TRAINER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAINER)
train_one_step = _TRAINER.train_one_step
train_one_random_chunk_step = _TRAINER.train_one_random_chunk_step
resize_compressor_query_tensor = _TRAINER._resize_compressor_query_tensor


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward_target_chunk(
        self,
        batch_x,
        *,
        chunk_rows,
        chunk_cols,
        target_row_idx,
        target_col_idx,
    ):
        row_span = make_chunk_spans(batch_x.shape[1], chunk_rows)[target_row_idx]
        col_span = make_chunk_spans(batch_x.shape[2], chunk_cols)[target_col_idx]
        return self.value.expand(
            batch_x.shape[0],
            row_span.end - row_span.start,
            col_span.end - col_span.start,
            1,
        )


class _SquaredErrorCriterion:
    def __call__(self, logits, target):
        return (logits[..., 0] - target).square()

    def median(self, logits):
        return logits[..., 0]


def test_train_one_step_uses_explicit_loss_mask_when_present():
    model = _FakeModel()
    criterion = _SquaredErrorCriterion()
    batch_x = torch.zeros(1, 4, 4)
    batch_target = torch.ones(1, 4, 4)
    loss_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    loss_mask[:, 1, 2] = True

    _, metrics = train_one_step(
        model=model,
        criterion=criterion,
        batch_x=batch_x,
        batch_target=batch_target,
        loss_mask=loss_mask,
        chunk_rows=2,
        chunk_cols=2,
    )

    assert metrics["loss_missing"] == 1.0
    assert metrics["rmse_missing"] == 1.0
    assert metrics["exposed_holdout_rmse"] == 1.0
    assert model.value.grad is not None
    assert float(model.value.grad) < 0.0


def test_random_chunk_step_finds_chunk_with_explicit_loss_mask():
    model = _FakeModel()
    criterion = _SquaredErrorCriterion()
    batch_x = torch.zeros(1, 4, 4)
    batch_target = torch.ones(1, 4, 4)
    loss_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    loss_mask[:, 3, 3] = True

    _, metrics = train_one_random_chunk_step(
        model=model,
        criterion=criterion,
        batch_x=batch_x,
        batch_target=batch_target,
        loss_mask=loss_mask,
        chunk_rows=2,
        chunk_cols=2,
    )

    assert metrics["loss_missing"] == 1.0
    assert metrics["rmse_missing"] == 1.0
    assert metrics["target_row_idx"] == 1.0
    assert metrics["target_col_idx"] == 1.0


def test_random_chunk_step_averages_multiple_target_gradients():
    torch.manual_seed(0)
    model = _FakeModel()
    criterion = _SquaredErrorCriterion()
    batch_x = torch.zeros(1, 4, 4)
    batch_target = torch.ones(1, 4, 4)
    loss_mask = torch.ones(1, 4, 4, dtype=torch.bool)

    _, metrics = train_one_random_chunk_step(
        model=model,
        criterion=criterion,
        batch_x=batch_x,
        batch_target=batch_target,
        loss_mask=loss_mask,
        chunk_rows=2,
        chunk_cols=2,
        num_target_chunks=2,
    )

    assert metrics["targets_sampled"] == 2.0
    assert metrics["loss_missing"] == 1.0
    assert torch.isclose(model.value.grad, torch.tensor(-2.0))


def test_query_resize_preserves_existing_queries_and_fills_new_queries():
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
    target = torch.zeros(1, 5, 1, 4)

    resized = resize_compressor_query_tensor(
        source,
        target,
        query_axis=1,
    )

    assert resized.shape == target.shape
    assert torch.equal(resized[:, :3], source)
    expected_mean = source.mean(dim=1, keepdim=True)
    assert torch.equal(resized[:, 3:], expected_mean.expand(1, 2, 1, 4))
