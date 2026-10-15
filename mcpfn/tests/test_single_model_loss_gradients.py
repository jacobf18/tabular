import math
from types import SimpleNamespace

import pytest
import torch

import diagnose_single_model_loss_gradients as diagnostic
import diagnose_single_model_full_gradients as full_diagnostic
from recsys_evaluation_common import write_json
from screen_single_matrixformer_initializers import cache_episode
from single_model_distillation import Episode


def test_gradient_geometry_detects_exact_cancellation_and_domain_conflict():
    point = torch.tensor([2., 0., 0.])
    total = torch.tensor([1., 0., 0.])
    geometry = diagnostic.geometry(total, point)
    assert geometry["probability_norm"] == 1.
    assert geometry["point_probability_cosine"] == -1.
    assert geometry["probability_to_point_norm"] == .5
    cross = diagnostic.cross_geometry(point, -point)
    assert cross["cosine"] == -1. and cross["dot"] == -4.


def test_zero_and_orthogonal_gradients_have_finite_geometry():
    zeros = torch.zeros(3)
    result = diagnostic.geometry(zeros, zeros)
    assert result["point_probability_cosine"] is None
    point = torch.tensor([1., 0., 0.])
    total = torch.tensor([1., 2., 0.])
    result = diagnostic.geometry(total, point)
    assert result["probability_norm"] == 2. and result["point_probability_cosine"] == 0.
    assert result["total_norm"] == pytest.approx(math.sqrt(5.))


def test_crop_preserves_parent_scaling_original_masks_and_source():
    target = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
    mask = target % 2 == 0
    ratings = SimpleNamespace(means=torch.arange(6.), stds=torch.ones(6), row_offsets=torch.arange(5.), row_scales=torch.ones(5))
    episode = Episode(target.masked_fill(mask, torch.nan), target, mask,
                      {"domain": "recsys", "gradient_data": "synthetic", "rows": 5, "columns": 6}, ratings)
    cached = cache_episode(episode)
    cropped = diagnostic.cropped_case(cached, rows=3, columns=4)
    assert cached["source"].shape == (1, 5, 6)
    torch.testing.assert_close(cropped.target, target[:, :3, :4])
    torch.testing.assert_close(cropped.mask, mask[:, :3, :4])
    torch.testing.assert_close(cropped.ratings.means, torch.arange(4.))
    torch.testing.assert_close(cropped.ratings.row_offsets, torch.arange(3.))
    assert cropped.metadata["rows"] == 5
    cached["metadata"]["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        diagnostic.cropped_case(cached)


def test_full_context_screen_covers_both_domains_and_includes_large_imputation():
    assert len(full_diagnostic.CASE_INDICES) == 20
    assert tuple(range(16, 20)) == full_diagnostic.CASE_INDICES[8:12]
    assert full_diagnostic.CASE_INDICES[-8:] == tuple(range(20, 28))


def test_full_diagnostic_waits_for_terminal_campaign_without_restarting(tmp_path, monkeypatch):
    for state in ("complete", "not_promoted"):
        write_json(tmp_path / "status.json", {"state": state})
        assert full_diagnostic.campaign_ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="stopped"):
        full_diagnostic.campaign_ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "waiting_for_synthetic_confirmation", "pid": 123})
    monkeypatch.setattr(full_diagnostic.os, "kill", lambda *args: None)
    assert not full_diagnostic.campaign_ready(tmp_path)
    monkeypatch.setattr(full_diagnostic.os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(RuntimeError, match="process is missing"):
        full_diagnostic.campaign_ready(tmp_path)
