from copy import deepcopy

import pytest

import diagnose_single_model_confirmation_nrmse as diagnostic
from recsys_evaluation_common import sha256, write_json
from train_single_matrixformer_continuation import bank_specs


def confirmation_fixture(root):
    run, confirmation, campaign = [root / name for name in ("run", "confirmation", "campaign")]
    run.mkdir()
    confirmation.mkdir()
    checkpoints = []
    for name in ("general.pth", "specialist.pth", "parent.pth", "best_validation_checkpoint.pth"):
        path = run / name
        path.write_bytes(name.encode())
        checkpoints.append(str(path))
    bank = confirmation / "synthetic_bank.pt"
    bank.write_bytes(b"immutable synthetic bank fixture")
    specifications = bank_specs(34100501, confirmation=True)
    config = {"source_sha256": {}, "input_sha256": {}, "teacher_config_sha256": {},
              "teachers": checkpoints[:2], "parent_checkpoint": checkpoints[2],
              "confirmation_cases": specifications, "seed": 31100501, "memory_fraction": .8}
    write_json(run / "config.json", config)
    write_json(run / "registration.json", {"config_sha256": sha256(run / "config.json")})
    write_json(run / "status.json", {"state": "training_complete", "best_step": 150})
    cases = []
    for index, specification in enumerate(specifications):
        shape = specification["shape"][::-1] if specification.get("transpose") else specification["shape"]
        cases.append({"index": index, "seed": specification["seed"], "domain": specification["domain"],
                      "gradient_data": "synthetic", "rows": shape[0], "columns": shape[1]})
    for name in ("reference", "initialization", "student"):
        write_json(confirmation / f"{name}.json", {"cases": deepcopy(cases), "bank_sha256": sha256(bank)})
    write_json(confirmation / "gate.json", {"cases": 46, "selected_step": 150, "passed": False,
                                            "checkpoint_sha256": sha256(run / "best_validation_checkpoint.pth"),
                                            "bank_sha256": sha256(bank)})
    write_json(confirmation / "manifest.json", {"purpose": "synthetic fixture"})
    write_json(confirmation / "status.json", {"state": "complete"})
    write_json(campaign / "cache_launcher_config.json", {"run": str(run)})
    write_json(campaign / "status.json", {"state": "not_promoted"})
    return run, confirmation, campaign


def test_registration_keeps_all_full_contexts_exact_models_and_original_rejection(tmp_path):
    run, confirmation, campaign = confirmation_fixture(tmp_path)
    gate_before = sha256(confirmation / "gate.json")
    config = diagnostic.make_config(run, confirmation, campaign)
    assert config["case_indices"] == list(range(30))
    assert config["case_shapes"]["24"] == [2000, 1000]
    assert config["case_shapes"]["25"] == [1000, 2000]
    assert config["original_gate_passed"] is False and config["original_selected_step"] == 150
    assert config["pooled_rmse_parity"] == {"rtol": 1e-6, "atol": 1e-6}
    assert set(config["checkpoints"]) == {"general", "specialist", "initialization", "candidate"}
    assert all(config[field] is False for field in ("training_performed", "checkpoint_selection_performed",
                                                   "real_benchmark_data_used", "gate_changed"))
    assert "cpu_diagnostic" not in config
    output = tmp_path / "diagnostic"
    diagnostic.full.register(config, output)
    diagnostic.full.verify_config(config, output)
    assert sha256(confirmation / "gate.json") == gate_before
    assert not (output / "predictions").exists()


@pytest.mark.parametrize("failure", ["missing_case", "duplicate_seed", "wrong_seed", "real", "wrong_domain", "wrong_shape"])
def test_confirmation_pairing_and_complete_scope_fail_closed(tmp_path, failure):
    run, confirmation, campaign = confirmation_fixture(tmp_path)
    path = confirmation / "student.json"
    report = diagnostic.metric.read(path)
    if failure == "missing_case":
        report["cases"].pop()
    elif failure == "duplicate_seed":
        path = confirmation / "reference.json"
        report = diagnostic.metric.read(path)
        report["cases"][1]["seed"] = report["cases"][0]["seed"]
    elif failure == "wrong_seed":
        report["cases"][0]["seed"] += 1
    elif failure == "real":
        report["cases"][0]["gradient_data"] = "real"
    elif failure == "wrong_domain":
        report["cases"][30]["domain"] = "imputation"
    else:
        report["cases"][24]["rows"] = 100
    write_json(path, report)
    with pytest.raises(ValueError):
        diagnostic.confirmation_inputs(run, confirmation, campaign)


@pytest.mark.parametrize("failure", ["step_zero", "gate_scope", "checkpoint", "bank", "unfinished", "wrong_campaign"])
def test_requires_completed_bound_confirmation_not_a_step_zero_rejection(tmp_path, failure):
    run, confirmation, campaign = confirmation_fixture(tmp_path)
    if failure in ("step_zero", "gate_scope"):
        gate = diagnostic.metric.read(confirmation / "gate.json")
        gate["selected_step" if failure == "step_zero" else "cases"] = 0
        write_json(confirmation / "gate.json", gate)
    elif failure == "checkpoint":
        (run / "best_validation_checkpoint.pth").write_bytes(b"changed")
    elif failure == "bank":
        (confirmation / "synthetic_bank.pt").write_bytes(b"changed")
    elif failure == "unfinished":
        write_json(confirmation / "status.json", {"state": "building_synthetic_bank"})
    else:
        write_json(campaign / "cache_launcher_config.json", {"run": "another run"})
    with pytest.raises(ValueError, match="completed, unchanged"):
        diagnostic.confirmation_inputs(run, confirmation, campaign)
