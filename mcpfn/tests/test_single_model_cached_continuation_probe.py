from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

import probe_single_model_cached_continuation as probe


def test_replay_removes_only_generation_teacher_inference_and_original_writes():
    source = probe.CAPTURE_SOURCE
    assert 'sampler.sample(' not in source
    assert 'backend.teacher_target(' not in source
    assert 'digest = save_tensor(' not in source
    assert 'write_json(directory / "manifest.json"' not in source
    assert 'backend.backward_episode(model, criterion, episode, teacher, 1.)' in source
    assert source.count('torch.manual_seed(task["gradient_seed"])') == 2
    assert 'streamed_point_metric_losses(model, criterion, episode, **TRAIN_OPTIONS)' in source
    assert 'objective.backward(retain_graph=position + 1 < len(defined))' in source
    assert 'require_same(' in source
    assert probe.CAPTURE_SOURCE_SHA256 == hashlib.sha256(source.encode()).hexdigest()
    assert 'sampler.sample(' in probe.training.replicated.CAPTURE_SOURCE


def test_scientific_record_excludes_only_runtime_and_memory_measurements():
    reference = {"step": 51, "accepted": True, "seconds": 10, "peak_gib": 12,
                 "projection": {"coefficients": [0.0, 1.0]}, "point_before": {"rmse": 1.0}}
    actual = deepcopy(reference)
    actual.update(seconds=20, peak_gib=13)
    probe.require_same(probe.scientific_record(actual), probe.scientific_record(reference), "same")
    actual["point_before"]["rmse"] += 1e-12
    with pytest.raises(ValueError, match="Exact cached replay differs"):
        probe.require_same(probe.scientific_record(actual), probe.scientific_record(reference), "changed")


@pytest.mark.parametrize("actual,expected", [(True, 1), ([1, 2], [1, 2, 3]), ({"extra": 1}, {}), (-0.0, 0.0)])
def test_exact_comparison_has_no_scalar_or_coverage_tolerance(actual, expected):
    with pytest.raises(ValueError, match="Exact cached replay differs"):
        probe.require_same(actual, expected, "strict")


def test_real_cli_help():
    result = subprocess.run([sys.executable, str(Path(probe.__file__).resolve()), "--help"],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
