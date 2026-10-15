import inspect
from pathlib import Path

import pytest
import torch
from threadpoolctl import threadpool_limits

import recover_single_model_legacy_blend_blas as recovery


def test_clones_preserve_all_scientific_code_and_original_globals():
    before = dict(vars(recovery.original))
    namespace = recovery.execution_namespace()
    assert vars(recovery.original).keys() == before.keys()
    assert all(vars(recovery.original)[name] is value for name, value in before.items())
    for name, function in recovery.owned_functions().items():
        if name != "verify":
            assert inspect.unwrap(namespace[name]).__code__ is inspect.unwrap(function).__code__
            assert inspect.unwrap(namespace[name]).__globals__ is namespace
    assert namespace["training"] is recovery.original.training
    assert namespace["REPLAY"] != recovery.original.REPLAY
    assert namespace["CONFIRMATION"] == recovery.original.CONFIRMATION
    assert namespace["POLICY"] == recovery.original.POLICY
    assert namespace["training"].BOUNDARIES == (0, 10, 25, 50, 64)
    assert len(namespace["training"].confirmation_cases()) == 284
    assert namespace["STAGE_WRITE_BUDGETS"] == recovery.original.STAGE_WRITE_BUDGETS


@pytest.mark.parametrize("name", ["confirm", "replay_actual"])
def test_cloned_neural_stages_preserve_no_grad(name):
    namespace = recovery.execution_namespace()

    def stop(output):
        assert not torch.is_grad_enabled()
        raise RuntimeError("prerequisite sentinel")

    namespace["prerequisite"] = stop
    with torch.enable_grad(), pytest.raises(RuntimeError, match="prerequisite sentinel"):
        namespace[name](Path("unused"))
    assert torch.is_grad_enabled()


def test_blas_restoration_is_scoped_and_does_not_initialize_cuda():
    initial_torch_threads = torch.get_num_threads()
    initial_cuda = torch.cuda.is_initialized()
    with threadpool_limits(limits=4, user_api="blas"):
        before = recovery.blas_profile()
        with recovery.original_arithmetic() as profile:
            assert len(profile) == 2
            assert all(pool["num_threads"] == 24 for pool in profile)
            assert torch.get_num_threads() == 4
        assert recovery.blas_profile() == before
    assert torch.get_num_threads() == initial_torch_threads
    assert torch.cuda.is_initialized() == initial_cuda


def test_runtime_restores_thread_pools_after_failure():
    before = recovery.blas_profile()
    torch_threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="sentinel"):
        with recovery.original_arithmetic():
            raise RuntimeError("sentinel")
    assert recovery.blas_profile() == before
    assert torch.get_num_threads() == torch_threads


def test_arithmetic_rejects_a_changed_blas_runtime(monkeypatch):
    monkeypatch.setattr(recovery, "blas_profile", lambda: [])
    with pytest.raises(ValueError, match="original OpenBLAS pools"):
        with recovery.original_arithmetic():
            pytest.fail("An unknown runtime must not execute qualification")


@pytest.fixture
def failed_prefix(tmp_path, monkeypatch):
    for name in ("OUTPUT", "REPLAY", "CONFIRMATION", "AUDIT", "CAMPAIGN", "REVIEW"):
        path = tmp_path / name.lower()
        monkeypatch.setattr(recovery.original, name, path)
        if name not in {"CAMPAIGN", "REVIEW"}:
            path.mkdir()
    training = tmp_path / "training"
    training.mkdir()
    monkeypatch.setattr(recovery.original.training, "OUTPUT", training)
    monkeypatch.setattr(recovery.original, "verify", lambda output: None)
    monkeypatch.setattr(recovery, "verify_hashes", lambda hashes: None)
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: None)
    recovery.write_json(training / "status.json", {"best": {"step": 50}})
    for directory, error in ((recovery.original.OUTPUT,
            "RuntimeError('replay failed; preserve its artifacts without restarting')"),
            (recovery.original.REPLAY, repr(ValueError(recovery.ERROR)))):
        recovery.write_json(directory / "status.json", {"state": "failed", "error": error, "pid": 100})
    for name in ("config.json", "registration.json", "run.lock"):
        (recovery.original.REPLAY / name).touch()
    (recovery.original.OUTPUT / "replay.log").write_text("report = prerequisite(output)\n" + recovery.ERROR)
    for name in ("config.json", "registration.json"):
        (recovery.original.CONFIRMATION / name).touch()
    recovery.write_json(recovery.original.CONFIRMATION / "status.json", {"state": "registered"})
    return tmp_path


def test_exact_empty_failure_and_unused_confirmation_are_required(failed_prefix):
    recovery.require_failed_prefix()
    recovery.require_unused_confirmation()


@pytest.mark.parametrize("mutation", ["live", "error", "artifact", "selection", "entry"])
def test_failure_proof_rejects_changed_or_started_science(failed_prefix, monkeypatch, mutation):
    if mutation == "live":
        monkeypatch.setattr(recovery.original, "process_identity", lambda pid: {"pid": pid})
    elif mutation == "error":
        recovery.write_json(recovery.original.REPLAY / "status.json", {"state": "failed", "error": "other", "pid": 100})
    elif mutation == "artifact":
        (recovery.original.REPLAY / "predictions.npz").touch()
    elif mutation == "selection":
        recovery.write_json(recovery.original.training.OUTPUT / "status.json", {"best": {"step": 64}})
    else:
        (recovery.original.OUTPUT / "replay.log").write_text(recovery.ERROR)
    with pytest.raises(ValueError):
        recovery.require_failed_prefix()


@pytest.mark.parametrize("mutation", ["started", "artifact", "campaign", "review", "published"])
def test_consumed_confirmation_and_publication_cannot_be_recovered(failed_prefix, mutation):
    if mutation == "started":
        recovery.write_json(recovery.original.CONFIRMATION / "status.json", {"state": "complete"})
    elif mutation == "artifact":
        (recovery.original.CONFIRMATION / "prediction.npz").touch()
    elif mutation in {"campaign", "review"}:
        getattr(recovery.original, mutation.upper()).mkdir()
    else:
        (recovery.original.training.OUTPUT / "best_validation_checkpoint.pth").touch()
    with pytest.raises(ValueError):
        recovery.require_unused_confirmation()


@pytest.mark.parametrize("passed", [False, True])
def test_unchanged_original_gate_controls_complete_frozen_campaign(monkeypatch, passed):
    calls = []
    monkeypatch.setattr(recovery, "verify", lambda output: {"confirmation_status_sha256": "reserved"})
    monkeypatch.setattr(recovery, "require_unused_confirmation", lambda: None)
    monkeypatch.setattr(recovery, "sha256", lambda path: "reserved")
    monkeypatch.setattr(recovery.original, "read", lambda path:
        {"state": "awaiting_independent_audit"} if path.name == "status.json" else {"passed": passed})
    monkeypatch.setattr(recovery.original, "run_child", lambda output, label, script, arguments, cpu=False:
        calls.append((label, script, arguments, cpu)))
    monkeypatch.setattr(recovery, "write_json", lambda path, payload: None)
    monkeypatch.setattr(recovery.original, "status", lambda *args, **kwargs: None)
    recovery.run()
    assert [row[0] for row in calls] == ["replay", "audit-replay", "confirm", "audit-confirmation"] + (
        ["benchmarks", "review"] if passed else [])
    if passed:
        assert calls[-2][1].name == "run_single_matrixformer_cached_benchmarks.py"
        assert calls[-1][1].name == "review_single_matrixformer_campaign.py"
        assert calls[-2][2] == ["--run", str(recovery.original.training.OUTPUT), "--confirmation",
            str(recovery.original.CONFIRMATION), "--source", str(recovery.original.PARENT_CACHE),
            "--output", str(recovery.original.CAMPAIGN)]


def test_pinned_evidence_and_scope():
    assert recovery.BLAS_THREADS == 24
    assert recovery.SELECTED_SHA256 == "43d5a07c16812967a7ff0453581e5a0705e2d203ae68ba9403f3482f1f492c81"
    assert recovery.AUDIT_SHA256 == "356672096ce19ee456858b88e42aecd204ba39b2fc9d97d79d62734c934f28e2"
    assert recovery.execution_namespace()["run_child"].__code__ is recovery.original.run_child.__code__
