from copy import deepcopy

import pytest

import train_dwip_teacher_mean as trainer


def fixture(monkeypatch):
    inputs = {str(trainer.REPO / source): "same" for source in (
        "scripts/train_dwip_teacher_mean.py", "scripts/dwip_teacher_mean_training.py",
        "scripts/dwip_teacher_compact_artifacts.py", "scripts/preflight_dwip_teacher_mean.py",
    )}
    preflight = {"training_ready": True, "input_sha256": inputs}
    proposal = {"proposed_queue": "trial"}
    monkeypatch.setattr(trainer, "read_json", lambda path: proposal if path.name == "next_single_model_compression_proposal.json" else preflight)
    monkeypatch.setattr(trainer, "verify_hashes", lambda values: None)
    monkeypatch.setattr(trainer, "compact_budget", lambda work: {"budget": "reviewed"})
    registration = {"kind": "dwip_teacher_mean_training_v1", "queue": "trial", "input_sha256": deepcopy(inputs),
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
    assert trainer.validate_registration(registration) == {"proposed_queue": "trial"}
