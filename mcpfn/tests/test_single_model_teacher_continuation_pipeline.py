import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_single_model_teacher_continuation_pipeline as pipeline
import replay_single_model_continuation_teachers as teacher


def test_unchanged_gates_with_additional_parent_grid_audit():
    source = pipeline.ADAPTED_SOURCE
    assert 'audit["prediction_grids"] != 1368' in source
    assert 'teacher_replay.verify_audit(TEACHERS)' in source
    assert 'training.original.alignment.fixed.PARENT' in source
    assert 'research_selection_tolerance' in source and '1136' in source
    assert source.index('run_child(output, "audit-teachers"') < source.index('for stage, cpu in (("replay", False)')
    assert pipeline.training is teacher.training
    assert pipeline.TEACHERS == teacher.OUTPUT
    assert pipeline.PARENT_CACHE == pipeline.baseline.PARENT_CACHE
    assert pipeline.no_confirmation_reason({"step": 0}, {}) is not None


@pytest.mark.parametrize("script", ["train_single_matrixformer_teacher_continuation.py",
    "audit_single_model_teacher_continuation.py", "replay_single_model_continuation_teachers.py",
    "run_single_model_teacher_continuation_pipeline.py"])
def test_real_cli_help(script):
    root = Path(pipeline.__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"],
                            cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("stage", ["replay", "audit"])
def test_actual_cli_dispatch_under_exclusive_lock(tmp_path, monkeypatch, stage):
    calls = []
    monkeypatch.setattr(teacher, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(teacher.__file__), "--stage", stage])
    monkeypatch.setattr(teacher, "replay", lambda output: calls.append(("replay", output)))
    monkeypatch.setattr(teacher, "audit", lambda output: calls.append(("audit", output)))
    teacher.main()
    assert calls == [(stage, tmp_path)]
    assert (tmp_path / "run.lock").exists()


def test_cli_failure_is_preserved_not_retried(tmp_path, monkeypatch):
    failures = []
    monkeypatch.setattr(teacher, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(teacher.__file__), "--stage", "audit"])

    def fail(output):
        raise ValueError("audit failed")

    monkeypatch.setattr(teacher, "audit", fail)
    monkeypatch.setattr(teacher, "status", lambda output, state, **fields: failures.append((state, fields)))
    with pytest.raises(ValueError, match="audit failed"):
        teacher.main()
    assert failures == [("failed", {"error": "ValueError('audit failed')"})]
