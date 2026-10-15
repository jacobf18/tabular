from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import audit_single_model_cached_continuation as audit
import recover_single_model_cached_continuation as recovery


def records(count):
    return [{"step": step, "accepted": step not in (15, 27, 29, 56, 63), "seconds": 1.0,
             "peak_gib": 2.0, "projection": {"coefficients": [0.0, 1.0]}, "point_before": {"rmse": 1.0}}
            for step in range(1, count + 1)]


def test_requires_all_original_records_and_does_not_claim_terminal_match():
    reference, actual = records(63), records(64)
    for row in actual[50:]:
        row.update(seconds=3.0, peak_gib=4.0)
    result = recovery.verify_replayed_rows(actual, reference)
    assert result == {"exact_prefix_records": 50, "exact_replayed_records": 13,
                      "reconstructed_terminal_step": 64, "terminal_outcome_matched_to_original": False}
    with pytest.raises(ValueError, match="Only original51-63"):
        recovery.compare_record(actual[-1], actual[-1])


@pytest.mark.parametrize("step", [1, 50, 51, 56, 63])
def test_even_tiny_scientific_differences_stop_reconstruction(step):
    reference, actual = records(63), records(64)
    actual[step - 1]["point_before"]["rmse"] += 1e-12
    with pytest.raises(ValueError, match="Exact cached replay differs"):
        recovery.verify_replayed_rows(actual, reference)


@pytest.mark.parametrize("count,reference_count", [(63, 63), (65, 63), (64, 62), (64, 64)])
def test_no_missing_or_extra_attempts(count, reference_count):
    with pytest.raises(ValueError, match="Require64"):
        recovery.verify_replayed_rows(records(count), records(reference_count))


def test_prefix_records_are_byte_semantics_not_runtime_exempt():
    actual = records(64)
    actual[49]["seconds"] += 1.0
    with pytest.raises(ValueError, match="untouched prefix"):
        recovery.verify_replayed_rows(actual, records(63))


def saved_state():
    model = {"decoder.weight": torch.ones(2), "frozen_buffer": torch.tensor(0.0)}
    saved = {"step": 50, "gradient_updates_performed": 47, "best_step": 50, "config_sha256": "config",
             "model": model, "optimizer": {"state": {0: {"step": torch.tensor(47.0)}}}}
    return saved, {"model": deepcopy(model)}


def test_boundary_requires_bitwise_model_and_full_optimizer_count():
    saved, selected = saved_state()
    recovery.check_saved(saved, selected, "config", "selected", {"selected_checkpoint_sha256": "selected"})
    selected["model"]["frozen_buffer"] = torch.tensor(-0.0)
    with pytest.raises(ValueError, match="boundary50"):
        recovery.check_saved(saved, selected, "config", "selected", {"selected_checkpoint_sha256": "selected"})


@pytest.mark.parametrize("field,value", [("step", 49), ("gradient_updates_performed", 50),
                                        ("best_step", 25), ("config_sha256", "wrong")])
def test_rejects_wrong_saved_boundary(field, value):
    saved, selected = saved_state()
    saved[field] = value
    with pytest.raises(ValueError, match="boundary50"):
        recovery.check_saved(saved, selected, "config", "selected", {"selected_checkpoint_sha256": "selected"})


@pytest.mark.parametrize("count", [None, 46, 48])
def test_rejects_missing_or_wrong_optimizer_count(count):
    saved, selected = saved_state()
    saved["optimizer"]["state"] = {} if count is None else {0: {"step": torch.tensor(float(count))}}
    with pytest.raises(ValueError, match="boundary50"):
        recovery.check_saved(saved, selected, "config", "selected", {"selected_checkpoint_sha256": "selected"})


def test_storage_is_additive_and_keeps_original_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    (tmp_path / "config.json").write_text("{}")
    gradients = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    snapshot = {"task_gradients": gradients[:2].clone()}
    record = {"composite_tasks": 2, "protected_tasks": 6}
    captured = {}

    def capture(payload, path):
        captured.update(payload)
        assert path == tmp_path / "raw_final_update.pt"
        return "digest"

    monkeypatch.setattr(recovery.shared_storage, "save", capture)
    assert recovery.save_endpoint(gradients, snapshot, record, [], {}, Path("manifest.json")) == "digest"
    recovery.storage.check_alias(captured)
    assert torch.equal(captured["snapshot"]["task_gradients"], snapshot["task_gradients"])
    assert captured["snapshot"]["task_gradients"].data_ptr() != snapshot["task_gradients"].data_ptr()
    record["raw_endpoint_sha256"] = "digest"
    assert "raw_endpoint_sha256" not in captured["record"]


def test_audit_keeps_scientific_gates_and_uses_original_paths_and_correct_alias():
    assert audit.check_gradient_alias is recovery.storage.check_alias
    assert 'manifest_path = training.original_manifest(step)' in audit.ADAPTED_SOURCE
    assert 'checks.check_endpoint(payload, rows[-1],' in audit.ADAPTED_SOURCE
    assert 'research_selection_tolerance"] != 0' in audit.ADAPTED_SOURCE
    assert 'len(decision["comparisons"]) != 22' in audit.ADAPTED_SOURCE
    assert 'hashes.update(recovery_hashes)' in audit.ADAPTED_SOURCE
    assert '152 * (4 + len(boundaries))' in audit.ADAPTED_SOURCE
    assert audit.ADAPTED_SOURCE_SHA256 == hashlib.sha256(audit.ADAPTED_SOURCE.encode()).hexdigest()
    assert recovery.original_training.OUTPUT != recovery.OUTPUT


def test_run_uses_cached_capture_with_decoder_coordinates_and_original_update():
    source = inspect.getsource(recovery.run)
    assert 'restore_rng(saved["rng"])' in source
    assert 'original_training.decoder.decoder_capture_coordinates(model, view)' in source
    assert 'probe.capture_cached_batch(' in source
    assert 'original_training.paired.protected_update(' in source
    assert 'original_training.validate(' in source
    assert 'sampler.sample(' not in source
    assert 'best_validation_checkpoint' not in source


@pytest.mark.parametrize("script", [recovery, audit])
def test_real_cli_help(script):
    result = subprocess.run([sys.executable, str(Path(script.__file__).resolve()), "--help"],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
