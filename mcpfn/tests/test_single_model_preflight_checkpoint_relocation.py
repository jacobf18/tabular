import pytest

import relocate_single_model_completed_preflight_checkpoints as relocation


@pytest.mark.parametrize("updates", [0, 1])
def test_only_exact_never_selectable_endpoint_is_admitted(updates):
    payload = {"step": 1, "preflight_only": True, "selection_qualified": False,
               "gradient_updates_performed": updates, "config_sha256": "original", "model": {"weight": 1}, "optimizer": {}}
    current = {"state": "complete", "accepted": bool(updates)}
    relocation.checkpoint_identity(payload, current, (updates, "digest"), "original")


@pytest.mark.parametrize("field,value", [("step", 2), ("preflight_only", False), ("selection_qualified", True),
    ("gradient_updates_performed", 0), ("config_sha256", "changed"), ("model", {}), ("optimizer", None)])
def test_selected_changed_or_incomplete_model_is_rejected(field, value):
    payload = {"step": 1, "preflight_only": True, "selection_qualified": False,
               "gradient_updates_performed": 1, "config_sha256": "original", "model": {"weight": 1}, "optimizer": {}}
    payload[field] = value
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, {"state": "complete", "accepted": True}, (1, "digest"), "original")


def test_scope_preserves_recent_probe_and_all_selected_models_and_archives():
    assert set(relocation.OWNERS) == {"top_two_blocks_preflight_v1", "general_imputation_teacher_preflight_v1",
                                     "point_inclusive_preflight_v1"}
    assert "anchor_rehearsal_preflight_v1" not in relocation.OWNERS
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert "copying.copy_and_verify(row)" in relocation.MAIN_SOURCE
    assert "copying.publish(row)" in relocation.MAIN_SOURCE
    assert "verify_hashes(inputs)" in relocation.MAIN_SOURCE
    assert '"checkpoint_free_after"' in relocation.MAIN_SOURCE
    assert relocation.main.__globals__ is relocation.__dict__
    assert relocation.status.__globals__ is relocation.__dict__
