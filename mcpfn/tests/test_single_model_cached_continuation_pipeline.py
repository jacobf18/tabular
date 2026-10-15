import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import recover_single_model_cached_continuation as recovery
import replay_single_model_cached_continuation_teachers as teacher
import run_single_model_cached_continuation_pipeline as pipeline


def test_pipeline_uses_only_reconstruction_and_unchanged_gates():
    source = pipeline.ADAPTED_SOURCE
    assert pipeline.training is recovery
    assert teacher.training is recovery
    assert pipeline.TEACHERS == teacher.OUTPUT
    assert pipeline.AUDIT == recovery.AUDIT
    assert pipeline.PARENT_CACHE == pipeline.baseline.PARENT_CACHE
    assert pipeline.training.confirmation_cases() == pipeline.baseline.training.confirmation_cases()
    assert 'audit["prediction_grids"] != 1368' in source
    assert 'audit["training_episodes"] != 1792' in source
    assert 'research_selection_tolerance' in source and '1136' in source
    assert 'teacher_replay.verify_audit(TEACHERS)' in source
    assert 'training.audit_recovery()' in source
    assert 'original_reservation_hashes()' in source
    assert source.index('run_child(output, "audit-teachers"') < source.index('for stage, cpu in (("replay", False)')
    assert pipeline.ADAPTED_SOURCE_SHA256 == hashlib.sha256(source.encode()).hexdigest()
    assert pipeline.no_confirmation_reason({"step": 0}, {}) is not None
    for name in ("OUTPUT", "TEACHERS", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name) != getattr(pipeline.baseline, name)


def test_teacher_replay_reuses_original_immutable_absolute_paths():
    source = teacher.ADAPTED_SOURCE
    assert 'path = training.original_manifest(batch["step"])' in source
    assert 'if not torch.equal(target, cached["teacher"]):' in source
    assert '1792' in source
    assert teacher.ADAPTED_SOURCE_SHA256 == hashlib.sha256(source.encode()).hexdigest()


def test_storage_guard_is_before_each_child_and_bound_to_registration():
    source = pipeline.ADAPTED_SOURCE
    assert 'require_stage_capacity(label)\n    command =' in source
    assert 'config["stage_write_budget_bytes"] != STAGE_WRITE_BUDGETS' in source
    assert '"stage_write_budget_bytes": STAGE_WRITE_BUDGETS' in source
    assert pipeline.MINIMUM_FREE == 18 * 1024**3


@pytest.mark.parametrize("stage", list(pipeline.STAGE_WRITE_BUDGETS))
def test_insufficient_storage_stops_before_consuming_a_stage(stage, monkeypatch):
    required = pipeline.MINIMUM_FREE + pipeline.STAGE_WRITE_BUDGETS[stage]
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: SimpleNamespace(free=required))
    pipeline.require_stage_capacity(stage)
    monkeypatch.setattr(pipeline.shutil, "disk_usage", lambda path: SimpleNamespace(free=required - 1))
    with pytest.raises(OSError, match="Preserve18 GiB"):
        pipeline.require_stage_capacity(stage)
    with pytest.raises(ValueError, match="declared write budget"):
        pipeline.require_stage_capacity("unknown-stage")


@pytest.mark.parametrize("stage", ["replay", "audit"])
def test_teacher_cli_calls_stage_under_lock(tmp_path, monkeypatch, stage):
    calls = []
    monkeypatch.setattr(teacher, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(teacher.__file__), "--stage", stage])
    monkeypatch.setattr(teacher, stage, lambda output: calls.append((stage, output)))
    teacher.main()
    assert calls == [(stage, tmp_path)]
    assert (tmp_path / "run.lock").exists()


def test_teacher_cli_records_failure_without_retry(tmp_path, monkeypatch):
    failures = []
    monkeypatch.setattr(teacher, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(teacher.__file__), "--stage", "replay"])

    def fail(output):
        raise ValueError("stop")

    monkeypatch.setattr(teacher, "replay", fail)
    monkeypatch.setattr(teacher, "status", lambda output, state, **fields: failures.append((state, fields)))
    with pytest.raises(ValueError, match="stop"):
        teacher.main()
    assert failures == [("failed", {"error": "ValueError('stop')"})]


@pytest.mark.parametrize("script", [teacher, pipeline])
def test_real_cli_help(script):
    result = subprocess.run([sys.executable, str(Path(script.__file__).resolve()), "--help"],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
