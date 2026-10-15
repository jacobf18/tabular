import builtins
import inspect

import pytest

import preflight_single_model_point_only as preflight
import test_single_model_legacy_point_preflight as inherited
from recover_single_model_legacy_strata_replay import code_globals


def test_new_protection_is_bound_without_changing_other_recipe_or_parent():
    assert preflight.PARENT == preflight.template.PARENT and preflight.PARENT_SHA256 == preflight.template.PARENT_SHA256
    assert preflight.PARENT != preflight.latest.training.OUTPUT / "selected_checkpoint.pth"
    assert preflight.RATE == 5e-6 and preflight.PREFIXES == preflight.template.PREFIXES
    assert preflight.composite is preflight.template.composite and preflight.mixture is preflight.template.mixture
    assert preflight.paired is not preflight.template.paired
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert (preflight.DATA_SEED, preflight.GRADIENT_SEED) == (741100901, 742100901)
    fresh = preflight.training.alignment.fixed.seeds_in(preflight.batch())
    old = preflight.training.alignment.fixed.seeds_in(preflight.template.batch())
    assert len(fresh) == 56 and not fresh.intersection(old)
    for name, function in vars(preflight).items():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(preflight):
            assert not code_globals(inspect.unwrap(function).__code__) - vars(preflight).keys() - vars(builtins).keys(), name
    for name in ("run", "audit_admitted", "replay_admitted", "save_raw", "audit_raw", "audit_admission"):
        assert inspect.unwrap(getattr(preflight, name)).__globals__ is vars(preflight)
    assert preflight.run.__globals__["paired"] is preflight.paired
    assert preflight.audit_admitted.__globals__["paired"] is preflight.paired


def latest():
    digest = preflight.SELECTED_V31_SHA256
    gate = {"passed": False, "confirmation_evaluated": True, "selected_step": 50, "cases": 284, "checkpoint_sha256": digest}
    return ({"complete": True, "gate": gate, "campaign": None, "review": None, "no_inference_ensemble": True},
            {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 63,
             "best": {"step": 50}, "prediction_grids": 1368, "independently_regenerated_training_tasks": 1792,
             "independently_regenerated_rating_contexts": 128, "independently_regenerated_structured_draws": 1536,
             "raw_endpoint": {"adam": {"gradient_vectors": 84}}},
            {"complete": True, "endpoint_grids": 56, "selected_grids": 152},
            {"complete": True, "cases": 284, "prediction_grids": 1136}, gate, digest)


@pytest.mark.parametrize("mutation", [None, "unfinished", "selection", "no_confirmation", "audit", "prefix", "published", "digest"])
def test_requires_complete_v31_rejection_without_reselection(mutation):
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
    if mutation:
        with pytest.raises(ValueError):
            preflight.validate_latest(result, training, replay, confirmation, gate, digest)
    else:
        preflight.validate_latest(result, training, replay, confirmation, gate, digest)


def test_actual_data_rejection_wrapper_does_no_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", preflight)
    inherited.test_rejected_admission_still_performs_no_model_or_optimizer_work(tmp_path, monkeypatch)


def test_actual_point_only_endpoint_uses_full_raw_gradient_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "preflight", preflight)
    inherited.test_actual_proposal_endpoint_uses_full_alias_through_preflight_storage(tmp_path, monkeypatch)
