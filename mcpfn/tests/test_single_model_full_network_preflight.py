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

import preflight_single_model_full_network as preflight
import test_single_model_legacy_point_preflight as inherited
from recover_single_model_legacy_strata_replay import code_globals


def test_full_network_probe_keeps_parent_priors_protector_and_fresh_seed_scope():
    assert preflight.PARENT == preflight.template.PARENT == preflight.latest.training.PARENT
    assert preflight.PARENT_SHA256 == preflight.template.PARENT_SHA256
    assert preflight.RATE == 5e-6 and preflight.paired is preflight.template.paired
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert preflight.mixture is preflight.template.mixture and preflight.composite is preflight.template.composite
    assert preflight.PREFIXES == preflight.full_network.PREFIXES
    assert (preflight.DATA_SEED, preflight.GRADIENT_SEED) == (801100901, 802100901)
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
    for name in ("run", "audit_admitted", "replay_admitted", "save_raw", "audit_raw", "audit_admission", "frozen_inventory"):
        assert inspect.unwrap(getattr(preflight, name)).__globals__ is vars(preflight)
    for node in ast.walk(ast.parse(preflight.ADAPTED_SOURCE)):
        if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(("full_network.", "paired.")):
            eval(compile(ast.Expression(node), "<runtime-attribute>", "eval"), vars(preflight))
    assert 'top_three.' not in preflight.ADAPTED_SOURCE and 'top_two.' not in preflight.ADAPTED_SOURCE


def test_identity_survives_fresh_interpreter():
    command = ("import json, runpy; module=runpy.run_path('scripts/preflight_single_model_full_network.py', run_name='identity'); "
               "print(json.dumps({'source':module['ADAPTED_SOURCE_SHA256'], 'batch':module['batch'](), "
               "'prefixes':module['PREFIXES'], 'storage':module['STORAGE_ACCOUNTING']}))")
    actual = subprocess.check_output([sys.executable, "-c", command], cwd=preflight.REPO,
        env={**os.environ, "PYTHONHASHSEED": "233", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    assert json.loads(actual) == {"source": preflight.ADAPTED_SOURCE_SHA256, "batch": preflight.batch(),
                                 "prefixes": list(preflight.PREFIXES), "storage": preflight.STORAGE_ACCOUNTING}


def latest():
    gate = {"passed": False, "confirmation_evaluated": False, "selected_step": 0, "real_benchmark_success": False}
    return ({"complete": True, "gate": gate, "campaign": None, "review": None, "no_inference_ensemble": True},
            {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
             "accepted_updates": 64, "best": {"step": 0}, "prediction_grids": 1368, "top_three_block_training": True,
             "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
             "independently_regenerated_structured_draws": 1536,
             "admission_attempts": {"data_rejected_steps": [], "optimizer_attempts": 64},
             "raw_endpoint": {"adam": {"gradient_vectors": 84}, "optimizer_attempt_step": 64}},
            {"complete": True, "endpoint_grids": 56, "selected_grids": 0},
            {"endpoint_reproduced": True, "selected_validation_passed": False, "selected_step": 0}, gate)


@pytest.mark.parametrize("entry,path,value", [
    (0, ("complete",), False), (0, ("campaign",), "published"), (0, ("review",), "reviewed"),
    (1, ("complete_training_audit",), False), (1, ("attempted_batches",), 63),
    (1, ("accepted_updates",), 63), (1, ("best", "step"), 10), (1, ("prediction_grids",), 1367),
    (1, ("top_three_block_training",), False), (1, ("independently_regenerated_training_tasks",), 1764),
    (1, ("independently_regenerated_rating_contexts",), 126), (1, ("independently_regenerated_structured_draws",), 1535),
    (1, ("admission_attempts", "data_rejected_steps"), [51]),
    (1, ("admission_attempts", "optimizer_attempts"), 63),
    (1, ("raw_endpoint", "adam", "gradient_vectors"), 28), (1, ("raw_endpoint", "optimizer_attempt_step"), 63),
    (2, ("complete",), False), (2, ("endpoint_grids",), 55), (2, ("selected_grids",), 152),
    (3, ("endpoint_reproduced",), False), (3, ("selected_validation_passed",), True), (3, ("selected_step",), 10),
    (4, ("passed",), True), (4, ("confirmation_evaluated",), True), (4, ("selected_step",), 10),
    (4, ("real_benchmark_success",), True),
])
def test_require_complete_v33_audits_without_fabricated_selection_or_confirmation(entry, path, value):
    records = latest()
    preflight.validate_latest(*records)
    current = records[entry]
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = value
    with pytest.raises(ValueError):
        preflight.validate_latest(*records)


@pytest.mark.parametrize("volume", [None, "checkpoint", "scratch"])
def test_expanded_evidence_cannot_reuse_smaller_budgets_or_lower_reserves(monkeypatch, volume):
    assert preflight.STORAGE_ACCOUNTING["raw_gradient_bytes"] == 26207494656
    assert preflight.RESERVES == preflight.template.RESERVES
    assert preflight.WRITE_BUDGETS == {"checkpoint": 384 * 1024**2, "root": 16 * 1024**2, "scratch": 32 * 1024**3}
    calls = []
    monkeypatch.setattr(preflight, "require_volume_space", lambda: calls.append(True))
    if volume:
        monkeypatch.setitem(preflight.WRITE_BUDGETS, volume, preflight.template.WRITE_BUDGETS[volume])
        with pytest.raises(ValueError, match="Expanded full-row"):
            preflight.require_space()
        assert not calls
    else:
        preflight.require_space()
        assert calls == [True]


@pytest.mark.parametrize("mutation", [None, "buffer", "signed_zero", "inventory", "nonfinite", "shape"])
def test_full_parameter_inventory_preserves_every_buffer_and_original_tensor_layout(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    names = [f"feature_encoder.parameter_{index}" for index in range(238)]
    initial = {name: torch.ones(1) for name in names}
    initial.update({f"transformer_blocks.0.buffer_{index}": torch.zeros(1) for index in range(16)})
    current = deepcopy(initial)
    preflight.write_json(tmp_path / "domain_gradient_probe_v1/parameter_inventory.json", [[name, [1], 1] for name in names])
    current[names[0]].add_(1.)
    if mutation == "buffer":
        current["transformer_blocks.0.buffer_0"].add_(1.)
    elif mutation == "signed_zero":
        current["transformer_blocks.0.buffer_0"].fill_(-0.)
    elif mutation == "inventory":
        names.pop()
    elif mutation == "nonfinite":
        current[names[0]].fill_(float("nan"))
    elif mutation == "shape":
        current[names[0]] = torch.ones(2)
    if mutation:
        with pytest.raises(ValueError):
            preflight.frozen_inventory(initial, current, names)
    else:
        result = preflight.frozen_inventory(initial, current, names)
        assert result["frozen_tensors"] == 16 and result["changed_trainable_tensors"] == [names[0]]


def test_data_rejection_still_does_no_model_or_optimizer_work(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", preflight)
    inherited.test_rejected_admission_still_performs_no_model_or_optimizer_work(tmp_path, monkeypatch)


def test_actual_endpoint_preserves_every_raw_gradient_and_full_proposal_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", preflight)
    inherited.test_actual_proposal_endpoint_uses_full_alias_through_preflight_storage(tmp_path, monkeypatch)
