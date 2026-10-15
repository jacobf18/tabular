from contextlib import nullcontext
import json

import numpy as np
import pytest
import torch

import dwip_recsys_experiment as experiment
import evaluate_dwip_recsys_candidate as candidate
from evaluate_dwip_recsys import user_rating_metrics


class ContextMeanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.shapes = []

    def forward(self, source, decode_mask):
        assert not torch.is_grad_enabled()
        assert torch.isnan(source[decode_mask]).all()
        self.shapes.append(tuple(source.shape[1:]))
        return source.nanmean().expand(int(decode_mask.sum()), 1) + self.anchor * 0


class IdentityCriterion:
    def mean(self, logits):
        return logits[:, 0]


def configure_toy_split(monkeypatch):
    observed = np.array([[1., 2., 3., 4., np.nan], [2., 3., 4., 5., 1.], [np.nan, 4., 5., 1., 2.]])
    users, items, truth = np.array([0, 2]), np.array([4, 0]), np.array([4., 2.])
    monkeypatch.setattr(experiment, "DATASETS", {"toy": {"folder": "toy"}})
    monkeypatch.setattr(experiment, "load_split", lambda folder, partition: (observed, None, (users, items, truth)))
    monkeypatch.setattr(experiment, "inference_components", lambda context, mode: (np.zeros(5), np.ones(5), np.zeros(3)))
    monkeypatch.setattr(experiment, "compute_context", nullcontext)
    return observed, users, items, truth


@pytest.mark.parametrize("rows,cols", [(2, 3), (3, 5), (2, 5)])
def test_rectangular_query_validation_matches_overlapping_tail_aggregation(monkeypatch, rows, cols):
    observed, users, items, truth = configure_toy_split(monkeypatch)
    model = ContextMeanModel()
    actual = experiment.rating_validation(model, IdentityCriterion(), block_rows=rows, block_cols=cols)["toy"]
    sums, counts = np.zeros(len(users)), np.zeros(len(users))
    for row_start, row_end in experiment.block_spans(observed.shape[0], rows):
        for col_start, col_end in experiment.block_spans(observed.shape[1], cols):
            selected = (users >= row_start) & (users < row_end) & (items >= col_start) & (items < col_end)
            sums[selected] += np.float32(np.nanmean(observed[row_start:row_end, col_start:col_end]))
            counts[selected] += 1
    expected = user_rating_metrics(users, truth, sums / counts)
    assert actual == pytest.approx(expected)
    assert model.shapes and all(shape == (rows, cols) for shape in model.shapes)
    assert model.anchor.grad is None and model.anchor.item() == 0


def test_default_geometry_is_identical_to_explicit_legacy_window(monkeypatch):
    configure_toy_split(monkeypatch)
    model = ContextMeanModel()
    actual = experiment.rating_validation(model, IdentityCriterion())
    explicit = experiment.rating_validation(model, IdentityCriterion(), block_rows=1000, block_cols=1000)
    assert actual == explicit


@pytest.mark.parametrize("rows,cols", [(0, 1000), (1000, -1)])
def test_invalid_evaluation_windows_rejected(rows, cols):
    with pytest.raises(ValueError, match="positive evaluation block"):
        experiment.rating_validation(ContextMeanModel(), IdentityCriterion(), block_rows=rows, block_cols=cols)


@pytest.fixture
def context_receipt(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best_validation_checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint")
    (tmp_path / "config.json").write_text(json.dumps({"arguments": {"normalization": "two_way_user_scale"}}))
    legacy = {"toy": {"MAE": .9, "RMSE": .95, "users": 2}}
    selected = {"step": 1, "score": .95, "validation_sha256": "selected-validation"}
    (tmp_path / "validation_1.json").write_text(json.dumps({"datasets": legacy}))
    monkeypatch.setattr(candidate, "verify_selection", lambda path: selected)
    monkeypatch.setattr(candidate, "DATASETS", {"toy": {"folder": "toy"}})
    monkeypatch.setattr(candidate, "OUTPUT", tmp_path)
    (tmp_path / "toy").mkdir()
    split = tmp_path / "toy/split.npz"
    split.write_bytes(b"fixed split")
    baseline = tmp_path / "mf_baseline.json"
    baseline.write_text(json.dumps({"validation": {"toy": {"MAE": 1., "RMSE": 1.}}}))
    screen = tmp_path / "screen.json"
    snapshot = screen.with_suffix(".source_snapshot")
    snapshot.mkdir()
    sources = {}
    for name in ("screen_dwip_recsys_context_windows.py", "dwip_recsys_experiment.py", "recsys_rating_episode.py",
                 "model_new_stable.py", "bar_distribution.py", "interface.py"):
        path = snapshot / name
        path.write_text(name)
        sources[str(path)] = candidate.sha256(path)
    report = {"checkpoint_sha256": candidate.sha256(checkpoint), "weights_bitwise_unchanged": True,
              "gradients_absent": True, "legacy_scores_reproduced": True, "outer_evaluated": False, "axis": "users",
              "normalization": "two_way_user_scale", "selection": selected, "baseline_sha256": candidate.sha256(baseline),
              "split_sha256": {"toy": candidate.sha256(split)}, "source_sha256": sources,
              "registered_windows": [[1000, 1000], [2000, 1000]], "best_inner_window": [2000, 1000],
              "candidates": [{"block_shape": [1000, 1000], "datasets": legacy, "worst_inner_rating_ratio": .95},
                             {"block_shape": [2000, 1000], "datasets": {"toy": {"MAE": .85, "RMSE": .88, "users": 2}},
                              "worst_inner_rating_ratio": .88}]}
    screen.write_text(json.dumps(report))
    return checkpoint, screen, tmp_path


def test_context_receipt_selects_only_registered_inner_winner(context_receipt):
    checkpoint, screen, root = context_receipt
    shape, selection = candidate.context_window_selection(checkpoint, screen, root)
    assert shape == [2000, 1000] and selection["worst_inner_rating_ratio"] == .88
    assert selection["screen_sha256"] == candidate.sha256(screen)
    assert candidate.context_window_selection(checkpoint, None, root) == ([1000, 1000], None)


@pytest.mark.parametrize("alteration", ["checkpoint", "baseline", "split", "gradient", "score", "winner", "source",
                                       "legacy", "normalization", "outer", "nan"])
def test_invalid_context_receipt_rejected(context_receipt, alteration):
    checkpoint, screen, root = context_receipt
    report = json.loads(screen.read_text())
    if alteration == "checkpoint":
        report["checkpoint_sha256"] = "changed"
    elif alteration == "baseline":
        report["baseline_sha256"] = "changed"
    elif alteration == "split":
        report["split_sha256"]["toy"] = "changed"
    elif alteration == "gradient":
        report["gradients_absent"] = False
    elif alteration == "score":
        report["candidates"][1]["worst_inner_rating_ratio"] = .7
    elif alteration == "winner":
        report["best_inner_window"] = [1000, 2000]
    elif alteration == "source":
        (screen.with_suffix(".source_snapshot") / "interface.py").write_text("changed")
    elif alteration == "legacy":
        report["candidates"][0]["datasets"]["toy"]["MAE"] = .8
    elif alteration == "normalization":
        report["normalization"] = "column"
    elif alteration == "outer":
        report["outer_evaluated"] = True
    elif alteration == "nan":
        report["candidates"][1]["datasets"]["toy"]["MAE"] = float("nan")
    screen.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        candidate.context_window_selection(checkpoint, screen, root)
