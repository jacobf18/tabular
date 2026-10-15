from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import audit_single_model_family_projection_training as audit


def raw_step(projected=True, step=1):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02)
    gradients = torch.tensor([[1., 0.], [-10., 1.]])
    for index in range(1, step + 1):
        record, snapshot = audit.training.update_model(model, optimizer, gradients, [.95, .05], projected, .01, raw=index == step)
    return record, snapshot


@pytest.mark.parametrize("projected,step", [(False, 1), (True, 1), (False, 100), (True, 100)])
def test_complete_raw_endpoint_independently_reconstructs(projected, step):
    record, snapshot = raw_step(projected, step)
    result = audit.check_snapshot(snapshot, record, [2], step, .02)
    assert result["step"] == step and result["gradient_vectors"] == 2
    assert result["combined_gradient_norm_accumulation_dtype"] == "float64"


@pytest.mark.parametrize("kind", ["gram", "weight", "clip", "adverse", "certificate", "gradient", "applied", "after", "moment", "count", "lr"])
def test_corrupted_endpoint_fails(kind):
    record, snapshot = raw_step()
    if kind == "gram":
        record["raw_gram"][0][0] += 1.
    elif kind == "weight":
        record["combination"]["weights"][0] += .1
    elif kind == "clip":
        record["combination"]["unweighted_task_clip"] = 1.
    elif kind == "adverse":
        record["actual_products_after"][0] = -.1
    elif kind == "certificate":
        record["projection"]["raw_gradient_coefficients"][0] += .1
    elif kind == "gradient":
        snapshot["task_gradients"][0, 0] += 1.
    elif kind == "applied":
        snapshot["applied_gradient"][0] += .1
    elif kind == "after":
        snapshot["after"][0] += .1
    elif kind == "moment":
        snapshot["optimizer_after"]["state"][0]["exp_avg"].fill_(1.)
    elif kind == "count":
        snapshot["optimizer_after"]["state"][0]["step"].fill_(2.)
    else:
        snapshot["optimizer_before"]["param_groups"][0]["lr"] = .02
    snapshot["record"] = deepcopy(record)
    with pytest.raises((AssertionError, ValueError)):
        audit.check_snapshot(snapshot, record, [2], 1, .02)


def test_long_gradient_norm_uses_fp64_without_changing_tolerance():
    generator = torch.Generator().manual_seed(91)
    values = torch.randn(2_000_000, generator=generator) * 1e-3
    measured = float(torch.linalg.vector_norm(values.double()))
    assert measured == pytest.approx(float(values.double().square().sum().sqrt()), rel=1e-12)


def episode_fixture():
    metadata = {"seed": 141100901, "gradient_data": "synthetic", "domain": "imputation"}
    mask = torch.tensor([[False, True], [True, False]])
    target = torch.ones((2, 2))
    episode = SimpleNamespace(source=target.masked_fill(mask, torch.nan), target=target, mask=mask, metadata=metadata, ratings=None)
    task = {"name": "legacy", "specification": {"seed": metadata["seed"], "domain": "imputation"}}
    payload = {"episode": audit.training.mixture.backend.cache_episode(episode), "teacher": target.clone(), "task": task, "routing": {"teacher": "fixture"}}
    logged = {"routing": payload["routing"], "losses": {"control": 1., "projected": 1.}, "diagnostics": {name: deepcopy(metadata) for name in audit.training.ARMS}}
    return payload, task, logged


@pytest.mark.parametrize("kind", [None, "seed", "teacher", "loss", "metadata", "mask"])
def test_training_episode_provenance_and_shared_targets(kind):
    payload, task, logged = episode_fixture()
    if kind is None:
        audit.check_episode(payload, task, logged)
        return
    if kind == "seed":
        payload["episode"]["metadata"]["seed"] += 1
    elif kind == "teacher":
        payload["teacher"][0, 0] = torch.nan
    elif kind == "loss":
        logged["losses"]["control"] = float("nan")
    elif kind == "metadata":
        logged["diagnostics"]["projected"]["gradient_data"] = "real"
    else:
        payload["episode"]["mask"].fill_(False)
    with pytest.raises(ValueError):
        audit.check_episode(payload, task, logged)


def test_partial_log_is_not_treated_as_completed_update(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}')
    with pytest.raises(ValueError):
        audit.read_rows(path)
