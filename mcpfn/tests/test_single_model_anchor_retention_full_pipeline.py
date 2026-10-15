import inspect

import pytest

import test_single_model_full_network_pipeline as predecessor
from single_model_registered_adaptation import replace_once


SOURCE = inspect.getsource(predecessor)
for old, new in {
    "import run_single_model_full_network_pipeline as pipeline": "import run_single_model_anchor_retention_full_pipeline as pipeline",
    "full_network_v34": "anchor_retention_v37",
    "run_single_model_full_network_pipeline.py": "run_single_model_anchor_retention_full_pipeline.py",
    "train_single_matrixformer_full_network.py": "train_single_matrixformer_anchor_retention.py",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


@pytest.mark.parametrize("retention_passes", [False, True])
def test_actual_prerequisite_requires_extra_retention_proof_without_bypassing_original_audits(monkeypatch, retention_passes):
    calls = []
    report = {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
        "training_episodes": 1792, "admission_attempts": {"training_tasks": 1792}, "research_selection_tolerance": 0,
        "raw_endpoint": {}, "decoder_only_training": False, "top_block_training": True, "full_network_training": True,
        "prediction_grids": 1368, "retention_baseline": str(pipeline.training.original.alignment.fixed.PARENT),
        "training_initialization": str(pipeline.training.PARENT), "artifact_sha256": {}}
    monkeypatch.setattr(pipeline, "verify", lambda output: calls.append("registration"))
    monkeypatch.setattr(pipeline, "read", lambda path: report)
    monkeypatch.setattr(pipeline, "verify_hashes", lambda values: calls.append("original_hashes"))
    monkeypatch.setattr(pipeline, "require_compact_audit", lambda current: calls.append("original_audit"))

    def require(training, current):
        assert training is pipeline.training and current is report
        calls.append("retention_audit")
        if not retention_passes:
            raise ValueError("Missing complete retention proof")

    monkeypatch.setattr(pipeline.retention_training, "require_report", require)
    if retention_passes:
        assert pipeline.prerequisite(pipeline.OUTPUT) is report
    else:
        with pytest.raises(ValueError, match="retention proof"):
            pipeline.prerequisite(pipeline.OUTPUT)
    assert calls == ["registration", "original_hashes", "original_audit", "retention_audit"]
