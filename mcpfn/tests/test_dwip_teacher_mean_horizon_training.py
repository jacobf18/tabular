from copy import deepcopy

import pytest

import train_dwip_teacher_mean_horizon as trainer


def fixture(monkeypatch):
    inputs = {str(trainer.REPO / source): "same" for source in (
        "scripts/train_dwip_teacher_mean_horizon.py", "scripts/dwip_teacher_mean_training.py",
        "scripts/dwip_distribution_compact_artifacts.py", "scripts/dwip_teacher_mean_horizon.py",
    )}
    preflight = {"training_ready": True, "input_sha256": inputs, "core_preflight_passed": True,
                 "reference_panels_verified": True, "full_pipeline_and_completion_audit_registered": True,
                 "integration_tests": {"tests": 600, "errors": 0, "failures": 0, "skipped": 0}}
    proposal = {"proposed_queue": trainer.QUEUE}
    monkeypatch.setattr(trainer, "read_json", lambda path: proposal if path == trainer.PROPOSAL else preflight)
    monkeypatch.setattr(trainer, "verify_hashes", lambda values: None)
    monkeypatch.setattr(trainer, "horizon_budget", lambda: {"budget": "reviewed"})
    registration = {"kind": "dwip_teacher_mean_horizon_v1", "queue": trainer.QUEUE, "input_sha256": deepcopy(inputs),
                    "preflight": "/tmp/preflight.json", "preflight_sha256": "digest", "budget": {"budget": "reviewed"}}
    return registration, preflight


def test_partial_preflight_cannot_admit_training(monkeypatch):
    registration, preflight = fixture(monkeypatch)
    preflight["training_ready"] = False
    with pytest.raises(ValueError, match="every declared preflight gate"):
        trainer.validate_registration(registration)


@pytest.mark.parametrize("change", ("queue", "kind", "missing_source", "changed_source", "changed_budget"))
def test_unregistered_changes_cannot_admit_training(monkeypatch, change):
    registration, preflight = fixture(monkeypatch)
    if change in {"queue", "kind"}:
        registration[change] = "other"
    elif change == "missing_source":
        registration["input_sha256"].pop(next(iter(registration["input_sha256"])))
    elif change == "changed_source":
        registration["input_sha256"][next(iter(registration["input_sha256"]))] = "changed"
    else:
        registration["budget"] = {}
    with pytest.raises(ValueError):
        trainer.validate_registration(registration)


def test_successful_registration_requires_matching_full_preflight_inputs(monkeypatch):
    registration, preflight = fixture(monkeypatch)
    assert trainer.validate_registration(registration) == {"proposed_queue": trainer.QUEUE}


@pytest.mark.parametrize("gate", ("core_preflight_passed", "reference_panels_verified", "full_pipeline_and_completion_audit_registered"))
def test_readiness_flag_cannot_bypass_unpassed_integration_gates(monkeypatch, gate):
    registration, preflight = fixture(monkeypatch)
    preflight[gate] = False
    with pytest.raises(ValueError, match="Every integration"):
        trainer.validate_registration(registration)


@pytest.mark.parametrize("field,value", (("tests", 499), ("errors", 1), ("failures", 1), ("skipped", 1)))
def test_readiness_flag_cannot_bypass_complete_tests(monkeypatch, field, value):
    registration, preflight = fixture(monkeypatch)
    preflight["integration_tests"][field] = value
    with pytest.raises(ValueError, match="Every integration"):
        trainer.validate_registration(registration)
