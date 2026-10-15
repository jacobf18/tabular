import test_single_model_all_imputation_retention_pipeline as original_template


SOURCE = original_template.SOURCE
for old, new in {
    "run_single_model_all_imputation_retention_pipeline": "run_single_model_legacy_blend_teacher_pipeline",
    "train_single_matrixformer_all_imputation_retention": "train_single_matrixformer_legacy_blend_teacher",
    "all_imputation_retention_v40": "legacy_blend_teacher_v41",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


@pytest.mark.parametrize("failure", [None, "original_audit", "retention_audit", "component_audit", "fixed_half_audit"])
def test_actual_prerequisite_requires_all_original_and_added_evidence(monkeypatch, failure):
    calls = []
    report = {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
        "training_episodes": 1792, "admission_attempts": {"training_tasks": 1792}, "research_selection_tolerance": 0,
        "raw_endpoint": {}, "decoder_only_training": False, "top_block_training": True, "full_network_training": True,
        "prediction_grids": 1368, "retention_baseline": str(pipeline.training.original.alignment.fixed.PARENT),
        "training_initialization": str(pipeline.training.PARENT), "artifact_sha256": {}}
    monkeypatch.setattr(pipeline, "verify", lambda output: calls.append("registration"))
    monkeypatch.setattr(pipeline, "read", lambda path: report)
    monkeypatch.setattr(pipeline, "verify_hashes", lambda values: calls.append("original_hashes"))

    def require(name):
        def check(*args):
            assert args[-1] is report
            calls.append(name)
            if failure == name:
                raise ValueError(name)
        return check

    monkeypatch.setattr(pipeline, "require_compact_audit", require("original_audit"))
    monkeypatch.setattr(pipeline.retention_training, "require_report", require("retention_audit"))
    monkeypatch.setattr(pipeline.blend_replay, "require_report", require("component_audit"))
    monkeypatch.setattr(pipeline.fixed_trial, "require_report", require("fixed_half_audit"))
    if failure:
        with pytest.raises(ValueError, match=failure):
            pipeline.prerequisite(pipeline.OUTPUT)
    else:
        assert pipeline.prerequisite(pipeline.OUTPUT) is report
    expected = ["registration", "original_hashes", "original_audit", "retention_audit", "component_audit", "fixed_half_audit"]
    assert calls == (expected[:expected.index(failure) + 1] if failure else expected)


def test_frozen_confirmation_campaign_and_ordinary_model_scope_remain_unchanged():
    assert pipeline.training_audit.training is pipeline.training
    assert pipeline.training_audit.blend_replay is pipeline.blend_replay
    assert pipeline.diagnostics is pipeline.training.diagnostics
    assert pipeline.retention_training is pipeline.training.retention_training
    assert pipeline.training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert len(pipeline.training.confirmation_cases()) == 284
    for marker in ('fixed_trial.reconstruct_candidate(training, payload, "cuda")',
            'fixed_trial.require_report(training, audit)', 'blend_replay.require_report(training, audit)',
            'run_single_matrixformer_cached_benchmarks.py', 'review_single_matrixformer_campaign.py'):
        assert marker in pipeline.ADAPTED_SOURCE
