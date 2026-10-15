import hashlib
import os
from pathlib import Path
import subprocess
import sys

import probe_single_model_compact_training as probe


def test_compact_probe_keeps_full_exact_comparisons_and_labels_extra_state():
    source = probe.ADAPTED_SOURCE
    assert 'require_equal(generated, cached, "complete_task")' in source
    assert 'for batch in config["batches"]:' in source
    assert 'with compact.runtime(saved_scaling) as replayed:' in source
    assert 'cached["episode"]["scaling"]' in source
    assert 'rating_contexts != 128' in source
    assert '"seed_only_regeneration": False' in source
    assert '"cached_artifacts_deleted": False' in source
    assert '"student_gradients_replayed": False' in source
    assert 'require_seed_only_failure()' in source
    assert probe.ADAPTED_SOURCE_SHA256 == hashlib.sha256(source.encode()).hexdigest()
    assert probe.OUTPUT != probe.baseline.OUTPUT


def test_real_cli_help():
    result = subprocess.run([sys.executable, str(Path(probe.__file__).resolve()), "--help"],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
