import inspect

import test_single_model_anchor_retention_full_pipeline as previous_tests
from single_model_registered_adaptation import replace_once


SOURCE = inspect.getsource(previous_tests)
for old, new in {
    "run_single_model_anchor_retention_full_pipeline": "run_single_model_fixed_half_shrunk_retention_pipeline",
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_fixed_half_shrunk_retention",
    "anchor_retention_v37": "fixed_half_shrunk_retention_v39",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, '    monkeypatch.setattr(pipeline.retention_training, "require_report", require)',
    '    monkeypatch.setattr(pipeline.retention_training, "require_report", require)\n'
    '    monkeypatch.setattr(pipeline.fixed_trial, "require_report", lambda training, current: calls.append("fixed_half_audit"))')
SOURCE = replace_once(SOURCE,
    '    assert calls == ["registration", "original_hashes", "original_audit", "retention_audit"]',
    '    assert calls == ["registration", "original_hashes", "original_audit", "retention_audit"] + (["fixed_half_audit"] if retention_passes else [])')
exec(compile(SOURCE, __file__, "exec"), globals())


def test_actual_replay_and_gate_require_contraction_and_complete_frozen_benchmarks():
    assert 'fixed_trial.reconstruct_candidate(training, payload, "cuda")' in pipeline.ADAPTED_SOURCE
    assert 'fixed_trial.check_replayed_candidate(payload, candidate["parameters"])' in pipeline.ADAPTED_SOURCE
    assert 'fixed_trial.require_report(training, audit)' in pipeline.ADAPTED_SOURCE
    assert pipeline.training_audit.checks is pipeline.training.paired is pipeline.fixed_trial.protection
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.training_audit.retention_training is pipeline.retention_training
    assert pipeline.training.retention.SPECIFICATION["prior_count"] == 1.
    assert pipeline.training.BOUNDARIES == (0, 10, 25, 50, 64) and pipeline.training.STEPS == 64
    assert 'run_single_matrixformer_cached_benchmarks.py' in pipeline.ADAPTED_SOURCE
    assert 'review_single_matrixformer_campaign.py' in pipeline.ADAPTED_SOURCE
