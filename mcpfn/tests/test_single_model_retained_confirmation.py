from copy import deepcopy
import json
import os

import pytest
import torch

from average_single_model_training_tail import WeightAverage
import confirm_single_model_retained_average as confirmation
from recsys_evaluation_common import sha256, write_json
from test_single_model_query_registration import fixture_config


def fixture_average(root):
    producer, campaign, average, output = [root / name for name in ("producer", "campaign", "average", "confirmation")]
    for directory in (producer, campaign, average, output):
        directory.mkdir()
    base = json.loads(json.dumps(fixture_config()))
    initial = {"weight": torch.tensor([.25, -.5, 1.]), "counter": torch.tensor(7)}
    parent = root / "initial.pth"
    torch.save({"model": initial}, parent)
    base["parent_checkpoint"] = str(parent)
    base["input_sha256"] = {str(parent): sha256(parent)}
    write_json(producer / "config.json", base)
    write_json(producer / "registration.json", {"config_sha256": sha256(producer / "config.json")})
    write_json(producer / "status.json", {"state": "training_complete", "step": 400})
    write_json(campaign / "cache_launcher_config.json", {"run": str(producer)})
    write_json(campaign / "status.json", {"state": "not_promoted"})
    source = {"model": base["model"], "producer": str(producer), "producer_config_sha256": sha256(producer / "config.json"),
              "registered_at_producer_step": 239, "registered_unix": 100., "steps": list(range(300, 401, 10)),
              "qualification": json.loads(json.dumps(confirmation.qualification_plan(base))), "source_sha256": {},
              "after_campaign": str(campaign), "after_campaign_config_sha256": sha256(campaign / "cache_launcher_config.json")}
    write_json(average / "config.json", source)
    write_json(average / "registration.json", {"config_sha256": sha256(average / "config.json")})
    (average / "contributing_states").mkdir()
    retained, receipts, collector = [], [], WeightAverage()
    for step in source["steps"]:
        state = {"weight": initial["weight"] + (step - 300) / 1000., "counter": initial["counter"].clone()}
        collector.add(state)
        digest = f"{step:064x}"
        path = average / "contributing_states" / f"step_{step}.pth"
        torch.save({"model": state, "step": step, "producer_config_sha256": source["producer_config_sha256"],
                    "producer_checkpoint_sha256": digest}, path)
        retained.append({"step": step, "path": str(path), "sha256": sha256(path), "producer_checkpoint_sha256": digest,
                         "captured_unix": 101. + step})
        receipts.append({"step": step, "checkpoint_sha256": digest, "captured_unix": 101. + step})
    write_json(average / "retained_states.json", {"expected_steps": source["steps"], "states": retained})
    write_json(average / "receipts.json", {"expected_steps": source["steps"], "snapshots": receipts})
    write_json(average / "retention_result.json", {"complete": True, "states": 11, "source_updates": source["steps"],
                                                 "retained_states_sha256": sha256(average / "retained_states.json")})
    payload = {"model": collector.state(), "step": 400, "source_updates": source["steps"],
               "config_sha256": sha256(average / "config.json"), "receipts_sha256": sha256(average / "receipts.json"),
               "derived_checkpoint": True, "gradient_updates_performed_here": 0}
    checkpoint = average / "averaged_checkpoint.pth"
    torch.save(payload, checkpoint)
    write_json(average / "result.json", {"complete": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": payload["config_sha256"], "receipts_sha256": payload["receipts_sha256"],
        "source_updates": source["steps"], "snapshots": 11, "single_parameter_set": True})
    write_json(average / "status.json", {"state": "complete"})
    config = {"average": str(average), "producer": str(producer), "base": base, "average_config_sha256": sha256(average / "config.json")}
    return config, source, payload, output


@pytest.mark.parametrize("change", [None, "cases", "seed", "gate", "window", "late", "query", "model"])
def test_fixed_qualification_cannot_be_relaxed(tmp_path, change):
    config, source, _, _ = fixture_average(tmp_path)
    if change == "cases":
        source["qualification"]["cases"].pop()
    elif change == "seed":
        source["qualification"]["cases"][0]["seed"] += 1
    elif change == "gate":
        source["qualification"]["retention_metrics"].pop()
    elif change == "window":
        source["steps"].pop(0)
    elif change == "late":
        source["registered_at_producer_step"] = 300
    elif change == "query":
        source["qualification"]["query_budget_applied"] = True
    elif change == "model":
        source["model"] = {"different": True}
    if change:
        with pytest.raises(ValueError):
            confirmation.verify_plan(source, config["base"])
    else:
        confirmation.verify_plan(source, config["base"])


@pytest.mark.parametrize("change", [None, "missing", "hash", "receipt", "configuration", "step", "nonfinite", "inventory", "mean", "counter"])
def test_independent_reconstruction_binds_every_contribution(tmp_path, change):
    config, source, payload, _ = fixture_average(tmp_path)
    average = confirmation.Path(config["average"])
    retained = confirmation.read(average / "retained_states.json")
    if change == "missing":
        retained["states"].pop()
    elif change == "hash":
        retained["states"][0]["sha256"] = "0" * 64
    elif change == "receipt":
        retained["states"][0]["producer_checkpoint_sha256"] = "0" * 64
    elif change in {"configuration", "step", "nonfinite", "inventory"}:
        path = confirmation.Path(retained["states"][0]["path"])
        state = torch.load(path, weights_only=True)
        if change == "configuration":
            state["producer_config_sha256"] = "changed"
        elif change == "step":
            state["step"] = 310
        elif change == "nonfinite":
            state["model"]["weight"][0] = torch.nan
        else:
            state["model"]["extra"] = torch.ones(1)
        torch.save(state, path)
        retained["states"][0]["sha256"] = sha256(path)
    elif change == "mean":
        payload["model"]["weight"][0] += .001
    elif change == "counter":
        payload["model"]["counter"] += 1
    write_json(average / "retained_states.json", retained)
    result = confirmation.read(average / "retention_result.json")
    result["retained_states_sha256"] = sha256(average / "retained_states.json")
    write_json(average / "retention_result.json", result)
    arguments = average, source, payload["model"], confirmation.read(average / "receipts.json")
    if change:
        with pytest.raises(ValueError):
            confirmation.reconstruct(*arguments)
    else:
        audit = confirmation.reconstruct(*arguments)
        assert audit["arithmetic_independently_reconstructed"] and audit["states"] == 11
        assert audit["non_bitwise_elements"] == 0
        assert not torch.cuda.is_initialized()


def test_candidate_audit_checks_mean_and_original_inventory(tmp_path):
    config, _, _, output = fixture_average(tmp_path)
    _, audit = confirmation.audit_candidate(config, output)
    assert audit["tensors"] == 2 and audit["elements"] == 4 and audit["changed_tensors"] == 1
    assert audit["arithmetic_independently_reconstructed"] and audit["selected_step"] == 400
    assert not (confirmation.Path(config["average"]) / "best_validation_checkpoint.pth").exists()
    write_json(confirmation.Path(config["producer"]) / "status.json", {"state": "training", "step": 399})
    with pytest.raises(ValueError, match="400-update"):
        confirmation.audit_candidate(config, output)


@pytest.mark.parametrize("state,campaign,expected", [("complete", False, True), ("not_promoted", True, True),
    ("failed", True, None), ("interrupted", True, None), ("not_promoted", False, None),
    ("evaluating", True, False), ("waiting_for_step", False, False)])
def test_ordering_never_bypasses_a_live_campaign_or_failure(tmp_path, state, campaign, expected):
    write_json(tmp_path / "status.json", {"state": state, "pid": os.getpid()})
    if expected is None:
        with pytest.raises(RuntimeError):
            confirmation.prerequisite_ready(tmp_path, campaign=campaign)
    else:
        assert confirmation.prerequisite_ready(tmp_path, campaign=campaign) is expected


def test_registration_preserves_prospective_plan_and_detects_changes(tmp_path):
    config, source, _, output = fixture_average(tmp_path)
    average = confirmation.Path(config["average"])
    registered = confirmation.register(average, output)
    confirmation.verify_registration(registered, output)
    assert registered["qualification"] == source["qualification"]
    assert len(registered["cases"]) == 92
    assert registered["after"] == source["after_campaign"]
    assert not registered["selection_performed"] and not registered["real_benchmark_inputs_used"]
    write_json(average / "config.json", {**source, "changed": True})
    with pytest.raises(ValueError):
        confirmation.verify_registration(registered, output)


@pytest.mark.parametrize("regression", [None, *confirmation.column.RETENTION_METRICS])
def test_confirmation_uses_all_six_metrics_and_publishes_only_on_pass(tmp_path, monkeypatch, regression):
    fixture, _, _, output = fixture_average(tmp_path)
    average = confirmation.Path(fixture["average"])
    config = confirmation.register(average, output)
    baseline = {"imputation": {"missbench_nrmse": 1., "uci_nrmse": 1., "common_nrmse": 1., "rmse": 1.},
                "recsys": {"rmse": 1., "mae": 1.}}
    student = {domain: {name: .9 for name in values} for domain, values in baseline.items()}
    if regression:
        student[regression[0]][regression[1]] = 1.01
    evaluations = iter([baseline, student])
    model = torch.nn.Module()
    monkeypatch.setattr(model, "load_state_dict", lambda state, strict: None)
    monkeypatch.setattr(confirmation, "evaluation_setup", lambda base: (model, None, None, None, None))

    def prepare(sampler, teachers, criterion, specifications, directory):
        assert specifications == config["cases"] and len(specifications) == 92
        bank = [{"metadata": {**case, "area_prior": {"requested_regime": "legacy", "applied": False}}} for case in specifications]
        torch.save(bank, directory / "synthetic_bank.pt")
        return bank, {"bank_sha256": sha256(directory / "synthetic_bank.pt"),
                      "general": deepcopy(baseline), "teacher": deepcopy(baseline), "references": deepcopy(baseline)}

    monkeypatch.setattr(confirmation.column.continuation, "prepare_bank", prepare)
    monkeypatch.setattr(confirmation.column.continuation, "evaluate_bank", lambda *arguments: (next(evaluations), []))
    confirmation.confirm(config, output)
    gate = confirmation.read(output / "gate.json")
    assert gate["passed"] is (regression is None)
    assert len(gate["retention_metrics"]) == 6 and len(gate["primary_metrics"]) == 4
    assert gate["cases"] == 92 and not gate["real_benchmark_success"]
    assert gate["arithmetic_independently_reconstructed"]
    assert (average / "best_validation_checkpoint.pth").exists() is (regression is None)
