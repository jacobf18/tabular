from copy import deepcopy

import pytest

import recover_single_model_family_balanced_orphaned_chain as recovery


@pytest.mark.parametrize("mutation", [None, "incomplete", "attempts", "updates", "selection", "failure", "not_failed"])
def test_only_exact_finished_training_and_operational_failure_are_eligible(mutation):
    trained = {"state": "training_complete", "step": 64, "accepted_updates": 56, "best": {"step": 25}}
    failed = {"state": "failed", "error": recovery.FAILURE}
    if mutation == "incomplete":
        trained["state"] = "failed"
    elif mutation == "attempts":
        trained["step"] = 63
    elif mutation == "updates":
        trained["accepted_updates"] = 55
    elif mutation == "selection":
        trained["best"]["step"] = 64
    elif mutation == "failure":
        failed["error"] = "Scientific audit failed"
    elif mutation == "not_failed":
        failed["state"] = "waiting_for_training"
    if mutation is None:
        recovery.validate_terminal(trained, failed)
    else:
        with pytest.raises(ValueError):
            recovery.validate_terminal(trained, failed)


@pytest.mark.parametrize("visibility", [None, "0", ""])
def test_gpu_auditor_retains_visibility_and_exact_original_arithmetic(monkeypatch, visibility):
    if visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    if visibility == "":
        with pytest.raises(ValueError, match="hide"):
            recovery.audit_environment()
    else:
        environment = recovery.audit_environment()
        assert environment.get("CUDA_VISIBLE_DEVICES") == visibility
        assert environment["OPENBLAS_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "24"
        assert environment["OMP_NUM_THREADS"] == "4"


@pytest.mark.parametrize("artifact", ["audit", "task_replay", "original_log"])
def test_any_previous_auditor_launch_prevents_a_duplicate(tmp_path, monkeypatch, artifact):
    monkeypatch.setattr(recovery.original, "ROOT", tmp_path)
    monkeypatch.setattr(recovery.original, "AUDIT", tmp_path / "audit")
    path = {"audit": tmp_path / "audit", "task_replay": tmp_path / "audit_task_replay",
            "original_log": tmp_path / "family_balanced_retention_v43_train_audit.log"}[artifact]
    path.touch()
    with pytest.raises(FileExistsError):
        recovery.require_unstarted()


@pytest.mark.parametrize("mutation", [None, "replay_started", "confirmation_started", "extra_evidence", "campaign", "alias"])
def test_only_completely_unused_qualification_reservations_can_proceed(tmp_path, monkeypatch, mutation):
    for name in ("ROOT", "REPLAY", "CONFIRMATION", "AUDIT", "CAMPAIGN", "REVIEW"):
        monkeypatch.setattr(recovery.original, name, tmp_path / name.lower())
    monkeypatch.setattr(recovery.original.training, "OUTPUT", tmp_path / "training")
    for directory in (recovery.original.REPLAY, recovery.original.CONFIRMATION):
        directory.mkdir()
        recovery.write_json(directory / "status.json", {"state": "registered"})
        for name in ("config.json", "registration.json"):
            (directory / name).touch()
    if mutation in ("replay_started", "confirmation_started"):
        directory = recovery.original.REPLAY if mutation == "replay_started" else recovery.original.CONFIRMATION
        recovery.write_json(directory / "status.json", {"state": "failed"})
    elif mutation == "extra_evidence":
        (recovery.original.REPLAY / "prediction.npz").touch()
    elif mutation == "campaign":
        recovery.original.CAMPAIGN.mkdir()
    elif mutation == "alias":
        recovery.original.training.OUTPUT.mkdir()
        (recovery.original.training.OUTPUT / "best_validation_checkpoint.pth").touch()
    if mutation is None:
        recovery.require_unstarted()
    else:
        with pytest.raises(ValueError):
            recovery.require_unstarted()


@pytest.mark.parametrize("passed", [False, True])
def test_complete_original_audit_precedes_unchanged_stages_and_full_real_scope(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    calls = []
    monkeypatch.setattr(recovery, "verify", lambda: calls.append("verify"))
    monkeypatch.setattr(recovery, "audit_environment", lambda: {})
    monkeypatch.setattr(recovery, "run_original_audit", lambda: calls.append("original_complete_audit"))
    monkeypatch.setattr(recovery, "read", lambda path: {"state": "awaiting_independent_audit"}
                        if path.name == "status.json" else {"passed": passed})
    monkeypatch.setattr(recovery, "sha256", lambda path: "fixed-hash")
    monkeypatch.setattr(recovery, "status", lambda *arguments, **keywords: None)
    children = []

    def child(output, label, script, arguments, cpu=False):
        calls.append(label)
        children.append((label, script, deepcopy(arguments), cpu))

    monkeypatch.setattr(recovery.original, "run_child", child)
    recovery.run()
    assert calls.index("original_complete_audit") < calls.index("replay")
    expected = ["replay", "audit-replay", "confirm", "audit-confirmation"]
    assert [row[0] for row in children] == expected + (["benchmarks", "review"] if passed else [])
    assert [row[3] for row in children[:4]] == [False, True, False, True]
    for _, script, arguments, _ in children[:4]:
        assert script == recovery.Path(recovery.original.__file__).resolve()
        assert arguments[-2:] == ["--output", str(recovery.original.OUTPUT)]
    if passed:
        assert children[-2][1].name == "run_single_matrixformer_cached_benchmarks.py"
        assert "--confirmation" in children[-2][2]
        assert children[-1][1].name == "review_single_matrixformer_campaign.py"
    result = recovery.original.read(tmp_path / "result.json")
    assert result["training_repeated"] is False and result["scientific_stages_repeated"] is False
    assert result["goal_success_not_assessed"] is True


def test_failed_original_audit_never_starts_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "verify", lambda: {})
    monkeypatch.setattr(recovery, "audit_environment", lambda: {})

    def failed_audit():
        raise RuntimeError("original audit failed")

    monkeypatch.setattr(recovery, "run_original_audit", failed_audit)
    monkeypatch.setattr(recovery.original, "run_child", lambda *arguments, **keywords: pytest.fail("No bypass"))
    with pytest.raises(RuntimeError, match="original audit failed"):
        recovery.run()
