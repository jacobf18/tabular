import pytest

import run_single_model_point_inclusive_pipeline as pipeline


def test_full_teacher_replay_and_unchanged_qualification_are_required():
    source = pipeline.ADAPTED_SOURCE
    assert source.index('run_child(output, "teacher-replay"') < source.index('run_child(output, "audit-teachers"')
    assert source.index('run_child(output, "audit-teachers"') < source.index('for stage, cpu in (("replay", False)')
    assert 'teacher_replay.verify_audit(TEACHERS)' in source
    assert 'teacher_replay.register(TEACHERS, output / "config.json")' in source
    assert 'training.original.alignment.fixed.PARENT' in source
    assert 'research_selection_tolerance' in source and '1136' in source
    assert pipeline.training is pipeline.teacher_replay.training
    assert pipeline.training is pipeline.training_audit.training
    assert pipeline.TEACHERS == pipeline.teacher_replay.OUTPUT
    assert pipeline.PARENT_CACHE == pipeline.baseline.PARENT_CACHE
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW", "TEACHERS"):
        assert "point_inclusive_v21" in str(getattr(pipeline, name))


def test_no_partial_teacher_replay_can_pass_prerequisite(monkeypatch):
    report = {"complete": True, "complete_training_audit": True, "through_step": 64, "attempted_batches": 64,
              "training_episodes": 1792, "research_selection_tolerance": 0, "raw_endpoint": {},
              "decoder_only_training": True, "prediction_grids": 1216,
              "retention_baseline": str(pipeline.training.original.alignment.fixed.PARENT),
              "training_initialization": str(pipeline.training.PARENT), "artifact_sha256": {},
              "teacher_identity_checked_tasks": 1792, "training_teacher_policy": pipeline.training.teacher_policy.POLICY}
    monkeypatch.setattr(pipeline, "verify", lambda output: {})
    monkeypatch.setattr(pipeline, "read", lambda path: report)
    monkeypatch.setattr(pipeline, "verify_hashes", lambda hashes: None)

    def reject(output):
        raise ValueError("teacher replay incomplete")

    monkeypatch.setattr(pipeline.teacher_replay, "verify_audit", reject)
    with pytest.raises(ValueError, match="teacher replay incomplete"):
        pipeline.prerequisite(pipeline.OUTPUT)
