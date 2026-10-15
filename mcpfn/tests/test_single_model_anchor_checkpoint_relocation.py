import pytest

import relocate_single_model_anchor_preflight_checkpoints as relocation


def test_only_stopped_never_selectable_anchor_diagnostics_can_move():
    assert set(relocation.OWNERS) == {"anchor_rehearsal_preflight_v1", "anchor_small_step_preflight_v1",
                                     "legacy_strata_preflight_v1"}
    assert relocation.OWNERS["anchor_rehearsal_preflight_v1"][0] == 0
    assert relocation.OWNERS["anchor_small_step_preflight_v1"][0] == 1
    assert relocation.OWNERS["legacy_strata_preflight_v1"][0] == 1
    assert relocation.OUTPUT != relocation.template.OUTPUT
    assert relocation.DESTINATION != relocation.template.DESTINATION
    assert relocation.checkpoint_identity is relocation.template.checkpoint_identity
    assert relocation.main.__globals__ is relocation.__dict__
    assert relocation.inventory.__globals__ is relocation.__dict__
    assert relocation.require_idle.__globals__ is relocation.__dict__
    assert relocation.status.__globals__ is relocation.__dict__
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert "copying.copy_and_verify(row)" in relocation.MAIN_SOURCE
    assert "copying.publish(row)" in relocation.MAIN_SOURCE
    assert "verify_hashes(inputs)" in relocation.MAIN_SOURCE


@pytest.mark.parametrize("mutation", ["promoted", "active", "changed_updates"])
def test_preserve_rejected_or_accepted_endpoint_never_active_or_promoted(mutation):
    payload = {"step": 1, "preflight_only": True, "selection_qualified": False,
               "gradient_updates_performed": 1, "config_sha256": "original", "model": {"weight": 1}, "optimizer": {}}
    current = {"state": "complete", "accepted": True}
    if mutation == "promoted":
        payload["selection_qualified"] = True
    elif mutation == "active":
        current["state"] = "capturing_gradients"
    else:
        payload["gradient_updates_performed"] = 2
    with pytest.raises(ValueError):
        relocation.checkpoint_identity(payload, current, (1, "digest"), "original")
