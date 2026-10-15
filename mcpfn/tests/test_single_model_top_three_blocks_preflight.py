import ast
import builtins
from copy import deepcopy
import inspect
import json
import os
import subprocess
import sys

import pytest
import torch

import preflight_single_model_top_three_blocks as preflight
import test_single_model_legacy_point_preflight as inherited
from recover_single_model_legacy_strata_replay import code_globals


def test_probe_changes_only_fresh_seeds_and_trainable_depth():
    assert preflight.PARENT == preflight.template.PARENT and preflight.PARENT_SHA256 == preflight.template.PARENT_SHA256
    assert preflight.PARENT != preflight.latest.training.OUTPUT / "selected_checkpoint.pth"
    assert preflight.RATE == 5e-6 and preflight.paired is preflight.template.paired
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert preflight.mixture is preflight.template.mixture and preflight.composite is preflight.template.composite
    assert preflight.PREFIXES == ("transformer_blocks.5.", "transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
    assert (preflight.DATA_SEED, preflight.GRADIENT_SEED) == (771100901, 772100901)
    fresh = preflight.training.alignment.fixed.seeds_in(preflight.batch())
    old = preflight.training.alignment.fixed.seeds_in([preflight.template.batch(), preflight.latest.training.batches(),
                                                      preflight.latest.training.confirmation_cases()])
    assert len(fresh) == 56 and not fresh.intersection(old)
    for current, original in zip(preflight.batch()["tasks"], preflight.template.batch()["tasks"]):
        previous = deepcopy(original)
        previous["gradient_seed"] = current["gradient_seed"]
        previous["specification"].get("structured", previous["specification"])["seed"] = current["specification"].get(
            "structured", current["specification"])["seed"]
        assert previous == current
    for name, function in vars(preflight).items():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(preflight):
            assert not code_globals(inspect.unwrap(function).__code__) - vars(preflight).keys() - vars(builtins).keys(), name
    for name in ("run", "audit_admitted", "replay_admitted", "save_raw", "audit_raw", "audit_admission"):
        assert inspect.unwrap(getattr(preflight, name)).__globals__ is vars(preflight)
    for node in ast.walk(ast.parse(preflight.ADAPTED_SOURCE)):
        if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(("top_three.", "paired.")):
            eval(compile(ast.Expression(node), "<runtime-attribute>", "eval"), vars(preflight))
    assert 'top_two.' not in preflight.ADAPTED_SOURCE


def test_identity_survives_a_fresh_interpreter():
    command = ("import json, runpy; module=runpy.run_path('scripts/preflight_single_model_top_three_blocks.py', run_name='identity'); "
               "print(json.dumps({'source':module['ADAPTED_SOURCE_SHA256'], 'batch':module['batch'](), "
               "'prefixes':module['PREFIXES'], 'storage':module['STORAGE_ACCOUNTING']}))")
    actual = subprocess.check_output([sys.executable, "-c", command], cwd=preflight.REPO,
        env={**os.environ, "PYTHONHASHSEED": "217", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    assert json.loads(actual) == {"source": preflight.ADAPTED_SOURCE_SHA256, "batch": preflight.batch(),
                                 "prefixes": list(preflight.PREFIXES), "storage": preflight.STORAGE_ACCOUNTING}


def latest():
    digest = preflight.SELECTED_V32_SHA256
    gate = {"passed": False, "confirmation_evaluated": True, "selected_step": 10, "cases": 284, "checkpoint_sha256": digest}
    return ({"complete": True, "gate": gate, "campaign": None, "review": None, "no_inference_ensemble": True},
            {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 61,
             "best": {"step": 10}, "prediction_grids": 1368, "independently_regenerated_training_tasks": 1764,
             "independently_regenerated_rating_contexts": 126, "independently_regenerated_structured_draws": 1536,
             "admission_attempts": {"data_rejected_steps": [51], "optimizer_attempts": 63},
             "raw_endpoint": {"adam": {"gradient_vectors": 84}, "optimizer_attempt_step": 64}},
            {"complete": True, "endpoint_grids": 56, "selected_grids": 152},
            {"complete": True, "cases": 284, "prediction_grids": 1136}, gate, digest)


@pytest.mark.parametrize("mutation", [None, "unfinished", "selection", "no_confirmation", "audit", "prefix", "published",
                                     "digest", "fabricated_tasks", "missing_noop", "optimizer_count", "endpoint"])
def test_requires_complete_v32_rejection_and_exact_admitted_task_inventory(mutation):
    result, training, replay, confirmation, gate, digest = latest()
    if mutation == "unfinished":
        training["complete_training_audit"] = False
    elif mutation == "selection":
        gate["selected_step"] = 64
    elif mutation == "no_confirmation":
        gate["confirmation_evaluated"] = False
    elif mutation == "audit":
        confirmation["prediction_grids"] = 1135
    elif mutation == "prefix":
        training["raw_endpoint"]["adam"]["gradient_vectors"] = 28
    elif mutation == "published":
        result["campaign"] = "other"
    elif mutation == "digest":
        digest = "changed"
    elif mutation == "fabricated_tasks":
        training["independently_regenerated_training_tasks"] = 1792
    elif mutation == "missing_noop":
        training["admission_attempts"]["data_rejected_steps"] = []
    elif mutation == "optimizer_count":
        training["admission_attempts"]["optimizer_attempts"] = 64
    elif mutation == "endpoint":
        training["raw_endpoint"]["optimizer_attempt_step"] = 63
    if mutation:
        with pytest.raises(ValueError):
            preflight.validate_latest(result, training, replay, confirmation, gate, digest)
    else:
        preflight.validate_latest(result, training, replay, confirmation, gate, digest)


@pytest.mark.parametrize("volume", [None, "checkpoint", "scratch"])
def test_expanded_artifact_accounting_keeps_original_reserves_and_budgets(monkeypatch, volume):
    assert preflight.STORAGE_ACCOUNTING["raw_gradient_bytes"] == 12108262992
    assert preflight.RESERVES == preflight.template.RESERVES
    assert preflight.WRITE_BUDGETS == preflight.template.WRITE_BUDGETS
    calls = []
    monkeypatch.setattr(preflight, "require_volume_space", lambda: calls.append(True))
    if volume:
        monkeypatch.setitem(preflight.WRITE_BUDGETS, volume, 1)
        with pytest.raises(ValueError, match="Expanded full-row"):
            preflight.require_space()
        assert not calls
    else:
        preflight.require_space()
        assert calls == [True]


@pytest.mark.parametrize("mutation", [None, "buffer", "signed_zero", "parameter", "inventory"])
def test_full_frozen_inventory_checks_parameters_and_selected_prefix_buffers(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    names = [f"transformer_blocks.5.parameter_{index}" for index in range(90)]
    initial = {name: torch.ones(1) for name in names}
    initial.update({f"frozen_{index}": torch.ones(1) for index in range(163)})
    buffer = "transformer_blocks.5.self_attention_between_features.rope.inv_freq"
    initial[buffer] = torch.zeros(1)
    current = deepcopy(initial)
    preflight.write_json(tmp_path / "domain_gradient_probe_v1/parameter_inventory.json", [[name, [1], 1] for name in names])
    current[names[0]].add_(1.)
    if mutation == "buffer":
        current[buffer].add_(1.)
    elif mutation == "signed_zero":
        current[buffer].fill_(-0.)
    elif mutation == "parameter":
        current["frozen_0"].add_(1.)
    elif mutation == "inventory":
        names.pop()
    if mutation:
        with pytest.raises(ValueError):
            preflight.frozen_inventory(initial, current, names)
    else:
        result = preflight.frozen_inventory(initial, current, names)
        assert result["frozen_tensors"] == 164 and result["changed_trainable_tensors"] == [names[0]]


def test_data_rejection_still_does_no_model_or_optimizer_work(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", preflight)
    inherited.test_rejected_admission_still_performs_no_model_or_optimizer_work(tmp_path, monkeypatch)


def test_actual_endpoint_preserves_every_raw_gradient_and_full_proposal_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", preflight)
    inherited.test_actual_proposal_endpoint_uses_full_alias_through_preflight_storage(tmp_path, monkeypatch)
