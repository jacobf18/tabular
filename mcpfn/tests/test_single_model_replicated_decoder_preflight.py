import pytest

import preflight_single_model_replicated_decoder as preflight


def test_preflight_uses_v8_not_the_unqualified_v17_checkpoint():
    assert preflight.PARENT == preflight.training.alignment.fixed.PARENT
    assert preflight.PARENT != preflight.preceding.training.OUTPUT / "selected_checkpoint.pth"
    assert preflight.OUTPUT == preflight.ROOT / "replicated_decoder_preflight_v1"
    assert preflight.RATE == 2e-5
    assert len(preflight.batch()["tasks"]) == 28


def test_preflight_preserves_all_per_draw_protection_and_qualification_boundaries():
    source = preflight.ADAPTED_SOURCE
    assert 'training.guarded.protected_update(view, optimizer, gradients[:len(entries)]' in source
    assert 'checks.check_endpoint(payload, record' in source
    assert 'compare_nested(decision, record["finite"])' in source
    assert 'replicated.capture_batch(' in source
    assert 'len(manifest["tasks"]) != 28' in source
    assert 'len(report["rows"]) != 28' in source
    assert '"training_tasks": 28' in source
    assert '"preflight_only": True' in source
    assert '"selection_qualified": False' in source
    assert 'config["capture_source_sha256"] != replicated.CAPTURE_SOURCE_SHA256' in source


def test_full_evidence_reserve_is_checked_at_registration_and_execution(monkeypatch):
    class Usage:
        free = 26 * 1024**3 - 1

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage())
    with pytest.raises(OSError, match="18 GiB"):
        preflight.require_space()
    Usage.free += 1
    preflight.require_space()
    assert 'wait_for_gpu(output, True, False)\n    require_space()' in preflight.ADAPTED_SOURCE


@pytest.mark.parametrize("mutation", ["incomplete", "qualified", "unconfirmed", "live", "campaign", "review"])
def test_cannot_bypass_or_compete_with_the_preceding_pipeline(monkeypatch, mutation):
    result = {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True}, "campaign": None, "review": None}
    status = {"state": "complete"}
    if mutation == "incomplete":
        result["complete"] = False
    elif mutation == "qualified":
        result["gate"]["passed"] = True
    elif mutation == "unconfirmed":
        result["gate"]["confirmation_evaluated"] = False
    elif mutation == "live":
        status["state"] = "confirm"
    else:
        result[mutation] = "existing"
    monkeypatch.setattr(preflight, "read", lambda path: result if path.name == "result.json" else status)
    monkeypatch.setattr(preflight.preceding, "prerequisite", lambda path: pytest.fail("Never bypass the terminal gate"))
    with pytest.raises(ValueError, match="completed unsuccessful"):
        preflight.require_predecessor()
