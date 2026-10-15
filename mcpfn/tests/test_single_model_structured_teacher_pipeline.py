import pytest

import run_single_model_structured_teacher_pipeline as pipeline


def test_full_teacher_replay_precedes_unchanged_endpoint_and_confirmation():
    source = pipeline.ADAPTED_SOURCE
    assert source.index('run_child(output, "teacher-replay"') < source.index('run_child(output, "audit-teachers"')
    assert source.index('run_child(output, "audit-teachers"') < source.index('for stage, cpu in (("replay", False)')
    assert 'teacher_replay.verify_audit(TEACHERS)' in source
    assert 'teacher_replay.register(TEACHERS, output / "config.json")' in source
    assert 'training.original.alignment.fixed.PARENT' in source
    assert 'research_selection_tolerance' in source and '1136' in source
    assert pipeline.training is pipeline.teacher_replay.training
    assert pipeline.TEACHERS == pipeline.teacher_replay.OUTPUT
    assert pipeline.PARENT_CACHE == pipeline.baseline.PARENT_CACHE
    assert 'teacher_replay.verify_audit' not in pipeline.baseline.ADAPTED_SOURCE


def test_actual_pipeline_requires_teacher_audit_even_if_other_training_checks_pass(monkeypatch):
    audit = {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
             "training_episodes": 1792, "research_selection_tolerance": 0, "raw_endpoint": {},
             "decoder_only_training": True, "prediction_grids": 1216,
             "retention_baseline": str(pipeline.training.original.alignment.fixed.PARENT),
             "training_initialization": str(pipeline.training.PARENT), "artifact_sha256": {},
             "teacher_identity_checked_tasks": 1792, "training_teacher_policy": pipeline.training.teacher_policy.POLICY}
    monkeypatch.setattr(pipeline, "verify", lambda output: {})
    monkeypatch.setattr(pipeline, "read", lambda path: audit)
    monkeypatch.setattr(pipeline, "verify_hashes", lambda hashes: None)

    def reject(output):
        raise ValueError("teacher replay incomplete")

    monkeypatch.setattr(pipeline.teacher_replay, "verify_audit", reject)
    with pytest.raises(ValueError, match="teacher replay incomplete"):
        pipeline.prerequisite(pipeline.OUTPUT)
