from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import single_model_fixed_half_trial as integration
import test_single_model_fixed_half_protection as probe_tests


def fixture(accepted=True):
    model, optimizer, gradients, state, before, record, snapshot = probe_tests.attempt(accepted=accepted)
    projection = integration.protection.previous._update_namespace["projection"]
    training = SimpleNamespace(paired=integration.protection, TRAINABLE_PARAMETERS=2,
        original=SimpleNamespace(alignment=SimpleNamespace(projection=projection)))
    return training, {"record": record, "snapshot": snapshot, "gradients": gradients}


@pytest.mark.parametrize("accepted", [False, True])
def test_independent_reconstruction_uses_the_contracted_candidate_even_after_rollback(accepted):
    training, payload = fixture(accepted)
    candidate, gradients = integration.reconstruct_candidate(training, payload, "cpu")
    assert torch.equal(candidate, payload["snapshot"]["contracted"])
    assert torch.equal(gradients, payload["gradients"])
    assert not torch.equal(candidate, payload["snapshot"]["uncontracted"])
    if not accepted:
        assert not torch.equal(candidate, payload["snapshot"]["after"])


@pytest.mark.parametrize("mutation", ["fraction", "legacy_namespace", "uncontracted", "contracted"])
def test_endpoint_replay_cannot_use_old_protection_or_corrupted_vectors(mutation):
    training, payload = fixture()
    if mutation == "fraction":
        payload["record"]["displacement_fraction"] = .25
    elif mutation == "legacy_namespace":
        training.paired = integration.protection.previous
    else:
        payload["snapshot"][mutation] = payload["snapshot"][mutation] + .01
    with pytest.raises(ValueError):
        integration.reconstruct_candidate(training, payload, "cpu")


def test_cpu_replay_check_rejects_uncontracted_or_rolled_back_parameters():
    training, payload = fixture(False)
    for name in ("uncontracted", "after"):
        with pytest.raises(ValueError):
            integration.check_replayed_candidate(payload, payload["snapshot"][name])
    integration.check_replayed_candidate(payload, payload["snapshot"]["contracted"])


@pytest.mark.parametrize("mutation", [None, "scope", "fraction", "parameters", "checked", "missing_endpoint"])
def test_qualification_requires_the_new_independent_optimizer_proof(mutation):
    training, payload = fixture()
    report = {"fixed_half_training": True, "displacement_fraction": .5,
        "raw_endpoint": {"optimizer_attempt_step": 64, "fixed_half_step": {"fraction": .5, "evaluated": True,
            "parameters": 2, "actual_saved_displacements_checked": True}}}
    if mutation == "scope":
        report["fixed_half_training"] = False
    elif mutation == "fraction":
        report["displacement_fraction"] = .25
    elif mutation == "parameters":
        report["raw_endpoint"]["fixed_half_step"]["parameters"] = 1
    elif mutation == "checked":
        report["raw_endpoint"]["fixed_half_step"]["actual_saved_displacements_checked"] = False
    elif mutation == "missing_endpoint":
        report["raw_endpoint"].pop("fixed_half_step")
    if mutation:
        with pytest.raises(ValueError):
            integration.require_report(training, report)
    else:
        integration.require_report(training, report)


def test_valid_projection_failure_and_all_data_rejections_remain_distinct():
    training, payload = fixture()
    report = {"fixed_half_training": True, "displacement_fraction": .5,
        "raw_endpoint": {"optimizer_attempt_step": 64, "fixed_half_step": {"fraction": .5, "evaluated": False}}}
    integration.require_report(training, report)
    report.update(raw_endpoint={"optimizer_attempt_step": None, "unchanged_initializer_checked": True,
        "accepted_updates": 0, "gradient_vectors": 0}, admission_attempts={"optimizer_attempts": 0})
    integration.require_report(training, report)
    forged = deepcopy(report)
    forged["admission_attempts"]["optimizer_attempts"] = 1
    with pytest.raises(ValueError):
        integration.require_report(training, forged)
