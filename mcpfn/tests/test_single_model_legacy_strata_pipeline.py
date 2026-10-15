from copy import deepcopy

import pytest

import run_single_model_legacy_strata_pipeline as pipeline


def test_full_qualification_scope_preserves_original_guards_and_real_campaign():
    assert pipeline.training is pipeline.training_audit.training
    assert pipeline.training.STEPS == 64
    assert len(pipeline.training.confirmation_cases()) == 284
    assert pipeline.MINIMUM_FREE == pipeline.template.MINIMUM_FREE
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    assert pipeline.run.__globals__["replay"] is pipeline.replay
    assert pipeline.run.__globals__["audit_replay"] is pipeline.audit_replay
    assert pipeline.materialize_endpoint_tasks.__globals__["actual_endpoint_step"] is pipeline.actual_endpoint_step
    assert pipeline.check_materialized_task.__globals__["actual_endpoint_step"] is pipeline.actual_endpoint_step
    assert "run_single_matrixformer_cached_benchmarks.py" in pipeline.ADAPTED_SOURCE
    assert "review_single_matrixformer_campaign.py" in pipeline.ADAPTED_SOURCE
    assert 'no_confirmation_reason(audit["best"], read(REPLAY / "gate.json"))' in pipeline.ADAPTED_SOURCE
    assert "training_episodes\"] != 1792" not in pipeline.ADAPTED_SOURCE
    assert pipeline.no_confirmation_reason({"step": 0}, {"selected_validation_passed": True})


def test_last_real_endpoint_can_precede_terminal_ordinal_without_fabricating_a_batch(monkeypatch):
    counts = {"scheduled_attempts": 64, "last_optimizer_attempt": 62, "accepted_updates": 50}
    descriptor = {"terminal_step": 64, "optimizer_attempt_step": 62, "accepted_updates": 50,
                  "raw_endpoint_sha256": "a" * 64, "config_sha256": "config"}
    monkeypatch.setattr(pipeline, "sha256", lambda path: "config")
    monkeypatch.setattr(pipeline, "read", lambda path: {"admission_attempts": counts} if path.name == "audit.json" else descriptor)
    assert pipeline.actual_endpoint_step() == 62
    descriptor["optimizer_attempt_step"] = 64
    with pytest.raises(ValueError):
        pipeline.actual_endpoint_step()


def test_empty_endpoint_is_audited_without_fake_neural_replay_or_selection(tmp_path, monkeypatch):
    replay = tmp_path / "replay"
    replay.mkdir()
    monkeypatch.setattr(pipeline, "REPLAY", replay)
    monkeypatch.setattr(pipeline, "sha256", lambda path: "digest")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(pipeline.torch.cuda, "is_initialized", lambda: False)
    report = {"admission_attempts": {"last_optimizer_attempt": None}, "best": {"step": 0},
              "accepted_updates": 0, "raw_endpoint": {"unchanged_initializer_checked": True}}
    monkeypatch.setattr(pipeline, "prerequisite", lambda output: deepcopy(report))

    def forbidden(*args):
        pytest.fail("No optimizer endpoint exists: do not initialize or replay a model")

    monkeypatch.setattr(pipeline, "replay_actual", forbidden)
    monkeypatch.setattr(pipeline, "audit_replay_actual", forbidden)
    monkeypatch.setattr(pipeline, "wait_for_gpu", forbidden)
    pipeline.status(replay, "registered")
    pipeline.replay(tmp_path)
    pipeline.audit_replay(tmp_path)
    result = pipeline.read(replay / "audit.json")
    assert result["complete"] and result["no_optimizer_endpoint"]
    assert result["endpoint_grids"] == result["selected_grids"] == 0
    assert result["student_inference_performed"] is False
    assert not pipeline.read(replay / "gate.json")["selected_validation_passed"]
    with pytest.raises(ValueError):
        pipeline.audit_replay(tmp_path)


def test_same_process_exec_handoff_still_accepts_only_exact_new_auditor(monkeypatch):
    expected = {"pid": 100, "start_ticks": 42,
                "command": "bash -c python scripts/train_single_matrixformer_legacy_strata.py --stage train; " + pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: expected)
    assert pipeline.chain_is_live(expected)
    changed = {**expected, "command": "python scripts/unrelated.py"}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: changed)
    assert not pipeline.chain_is_live(expected)


@pytest.mark.parametrize("mutation", [None, "fixed_task_count", "omitted_admission", "rating_count"])
def test_pipeline_requires_exact_admitted_counts_and_all_scheduled_structured_draws(monkeypatch, mutation):
    counts = {"scheduled_attempts": 64, "training_tasks": 1736, "rating_contexts": 124, "accepted_updates": 60}
    report = {"admission_attempts": counts, "accepted_updates": 60, "teacher_identity_checked_tasks": 1736,
              "training_teacher_policy": pipeline.training.teacher_policy.POLICY,
              "independently_regenerated_training_tasks": 1736, "independently_regenerated_rating_contexts": 124,
              "independently_regenerated_structured_draws": 1536, "exact_complete_task_and_teacher_digest_checks": True,
              "decoder_only_training": False, "top_block_training": True, "top_two_block_training": True,
              "autograd_replayed": False, "full_optimizer_trajectory_replayed": False}
    if mutation == "fixed_task_count":
        report["independently_regenerated_training_tasks"] = 1792
    elif mutation == "omitted_admission":
        report["independently_regenerated_structured_draws"] = 24 * 62
    elif mutation == "rating_count":
        report["independently_regenerated_rating_contexts"] = 128
    monkeypatch.setattr(pipeline, "sha256", lambda path: "audit")
    monkeypatch.setattr(pipeline, "read", lambda path: {"complete": True, "tasks": 1736, "audit_sha256": "audit"}
                        if path.name == "result.json" else {"state": "complete"} if path.name == "status.json"
                        else {"source_sha256": {}})
    monkeypatch.setattr(pipeline.training_audit, "read_prefix", lambda *args: ([], ""))
    monkeypatch.setattr(pipeline.training_audit, "admission_inventory", lambda *args: counts)
    if mutation:
        with pytest.raises(ValueError):
            pipeline.require_compact_audit(report)
    else:
        pipeline.require_compact_audit(report)
