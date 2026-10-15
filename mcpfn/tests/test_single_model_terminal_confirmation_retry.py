import os

import pytest

import retry_single_model_retained_confirmation as recovery
from recsys_evaluation_common import write_json
from test_single_model_query_registration import fixture_config
from average_single_model_retained_tail import qualification_plan


def test_terminal_state_with_a_live_process_does_not_release_gpu_ordering(tmp_path):
    write_json(tmp_path / "status.json", {"state": "failed", "pid": os.getpid()})
    with pytest.raises(RuntimeError, match="live process"):
        recovery.require_terminal(tmp_path, "failed")


def test_nonterminal_prerequisite_cannot_be_bypassed(tmp_path):
    write_json(tmp_path / "status.json", {"state": "evaluating", "pid": os.getpid()})
    with pytest.raises(ValueError, match="terminal state"):
        recovery.require_terminal(tmp_path, "failed")


@pytest.mark.parametrize("change", [None, "missing", "seed", "average_empty", "regime", "different_failure"])
def test_diagnosis_cannot_change_or_drop_confirmation_cases(change):
    config = fixture_config()
    probe = {"banks": {"selected": config["confirmation_cases"], "average": qualification_plan(config)["cases"]}}
    records = [{"bank": bank, "index": index, "seed": case["seed"],
                "area_prior": {"requested_regime": "legacy", "applied": False},
                "common_nonconstant_columns": 0 if bank == "selected" and index == 6 else 1,
                "query_count_min": 1, "query_count_max": 1}
               for bank, cases in probe["banks"].items() for index, case in enumerate(cases) if case["domain"] == "imputation"]
    if change == "missing":
        records.pop()
    elif change == "seed":
        records[-1]["seed"] += 1
    elif change == "average_empty":
        records[-1]["common_nonconstant_columns"] = 0
    elif change == "regime":
        records[-1]["area_prior"]["applied"] = True
    elif change == "different_failure":
        records[6]["query_count_max"] = 2
    if change:
        with pytest.raises(ValueError):
            recovery.verify_coverage_records(probe, records)
    else:
        recovery.verify_coverage_records(probe, records)


@pytest.mark.parametrize("fail", [False, True])
def test_only_ordering_adapter_changes_and_restores_on_failure(tmp_path, monkeypatch, fail):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    write_json(campaign / "status.json", {"state": "failed"})
    proof = {"original_campaign": str(campaign), "input_sha256": {}}
    original_ready = recovery.original.prerequisite_ready
    calls = []

    def confirm(config, output):
        assert recovery.original.prerequisite_ready(campaign, campaign=True)
        calls.append((config, output))
        if fail:
            raise RuntimeError("original gate raised")

    monkeypatch.setattr(recovery.original, "confirm", confirm)
    config = {"cases": "unchanged", "gate": "unchanged"}
    if fail:
        with pytest.raises(RuntimeError, match="original gate raised"):
            recovery.confirm_after_terminal_failure(config, tmp_path, proof)
    else:
        recovery.confirm_after_terminal_failure(config, tmp_path, proof)
    assert calls == [(config, tmp_path)]
    assert recovery.original.prerequisite_ready is original_ready
