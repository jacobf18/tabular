from pathlib import Path

import psutil
import pytest

import run_dwip_teacher_mean_horizon as pipeline


def test_controller_environment_is_cpu_only_and_strips_benchmark_overrides(monkeypatch):
    monkeypatch.setenv("BENCHSELECT_DATASETS", "unexpected")
    monkeypatch.setenv("BENCHSELECT_SEED", "999")
    assert pipeline.environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert pipeline.environment(True)["CUDA_VISIBLE_DEVICES"] == "0"
    assert not any(key.startswith("BENCHSELECT_") for key in pipeline.environment())


def test_matching_symlink_is_reused_but_existing_artifacts_are_not_overwritten(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"frozen")
    link = tmp_path / "output"
    pipeline.checked_link(link, source)
    pipeline.checked_link(link, source)
    assert link.is_symlink() and link.read_bytes() == b"frozen"
    other = tmp_path / "other"
    other.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Preserve"):
        pipeline.checked_link(other, source)
    assert other.read_bytes() == b"preserve"


def test_pid_reuse_is_not_a_live_original_handle():
    process = psutil.Process()
    assert pipeline.original_live({"pid": process.pid, "created": process.create_time()})
    assert not pipeline.original_live({"pid": process.pid, "created": process.create_time() - 1})
    assert not pipeline.original_live({"pid": 999999999, "created": 0.})


@pytest.mark.parametrize("tests,errors,failures,skipped", ((499, 0, 0, 0), (600, 1, 0, 0), (600, 0, 1, 0), (600, 0, 0, 1)))
def test_failed_partial_or_skipped_integration_tests_cannot_admit_training(tmp_path, tests, errors, failures, skipped):
    path = tmp_path / "tests.xml"
    path.write_text(f'<testsuites><testsuite tests="{tests}" errors="{errors}" failures="{failures}" skipped="{skipped}"/></testsuites>')
    with pytest.raises(ValueError):
        pipeline.check_test_report(path)


def test_complete_integration_test_report_is_accepted(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="600" errors="0" failures="0" skipped="0"/></testsuites>')
    assert pipeline.check_test_report(path)["tests"] == 600


def test_full_stage_commands_do_not_change_the_training_recipe(tmp_path):
    commands = pipeline.commands(tmp_path)
    assert set(commands) == {"training", "training_audit", "synthetic", "mteb", "transfer", "retention", "runtime", "completion_audit"}
    assert commands["training"][1:] == ["scripts/train_dwip_teacher_mean_horizon.py", "--registration", str(tmp_path / "training_registration.json")]
    assert "--training-only" in commands["training_audit"] and "--training-only" not in commands["completion_audit"]


def test_storage_admission_does_not_create_the_future_work_directory(tmp_path, monkeypatch):
    work_parent = tmp_path / "work"
    work_parent.mkdir()
    root = tmp_path / "queue"
    root.mkdir()
    benchmark = tmp_path / "bench"
    (benchmark / "logs").mkdir(parents=True)
    monkeypatch.setattr(pipeline, "WORK", work_parent)
    monkeypatch.setattr(pipeline, "BENCH", benchmark)
    calls = []
    monkeypatch.setattr(pipeline, "check_storage", lambda budget, free, allocated, launch=False: calls.append(launch) or {"allocated_bytes": allocated})
    pipeline.storage_guard(root, {}, launch=True)
    assert calls == [True]
    assert not list(work_parent.iterdir())


def test_existing_controller_identity_cannot_restart_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "controller_identity.json").write_text('{}')
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="already-started"):
        pipeline.run()


def test_hash_aliases_preserve_symlink_identity_and_reject_conflicts(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"frozen")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    hashes = pipeline.known_hashes({str(alias): "same"}, {str(source): "same"})
    assert hashes == {str(alias): "same", str(source): "same"}
    with pytest.raises(ValueError, match="Conflicting"):
        pipeline.known_hashes({str(alias): "same"}, {str(source): "different"})


def test_launch_preserves_existing_or_partial_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "pipeline_launch.json").write_text("{}")
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not relaunch"))
    with pytest.raises(ValueError, match="no restart"):
        pipeline.launch()


def test_only_new_horizon_commands_and_mean64_reference_are_used(tmp_path):
    commands = pipeline.commands(tmp_path)
    assert commands["synthetic"][1] == "scripts/run_dwip_teacher_mean_horizon.py"
    assert commands["completion_audit"][1] == "scripts/audit_dwip_teacher_mean_horizon.py"
    assert pipeline.CONTROL_ROOT.name == pipeline.PREVIOUS_QUEUE
    assert "teacher_mean_huber1_64k" in pipeline.PREVIOUS_QUEUE
    assert pipeline.HORIZON_STEPS == 128000
