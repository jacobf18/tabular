from copy import deepcopy

import pytest
import torch

import confirm_single_matrixformer_query_budget as confirmation
from recsys_evaluation_common import sha256, write_json
from test_single_model_area_registration import fixture_config as area_config
import train_single_matrixformer_query_budget as training


def fixture_config():
    config = area_config(version=7)
    config.update(version=8, **training.SEEDS, query_budget_objective=deepcopy(training.SPECIFICATION),
                  validation_cases=training.previous.previous.repeated_bank(training.SEEDS["validation_seed"]),
                  confirmation_cases=training.previous.previous.repeated_bank(training.SEEDS["confirmation_seed"], confirmation=True))
    return config


def evidence_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    config = {"policy": deepcopy(training.SPECIFICATION), "cases": 256, "query_seed_start": 50100501,
              "input_sha256": {}, "source_sha256": {}, "training_performed": False, "model_inference_performed": False,
              "checkpoint_selection_performed": False, "gate_changed": False, "real_benchmark_inputs_used": False}
    write_json(root / "config.json", config)
    write_json(root / "registration.json", {"config_sha256": sha256(root / "config.json")})
    write_json(root / "status.json", {"state": "complete"})
    records = [{"index": index, "seed": config["query_seed_start"] + index,
                "budget": training.budget_for_seed(config["query_seed_start"] + index),
                "original_queries": 100, "subsampled_queries": 16,
                "original_uci_only_columns": int(index == 0),
                "subsampled_uci_only_columns": 2 if index == 0 else int(index < 7)} for index in range(256)]
    write_json(root / "cases.json", records)
    torch.save([], root / "query_masks.pt")
    for name in ("original_columns.csv", "subsampled_columns.csv"):
        (root / name).write_text("fixture\n")
    write_json(root / "summary.json", {"complete": True, "config_sha256": sha256(root / "config.json"),
        "mask_sha256": sha256(root / "query_masks.pt"), "source_unchanged": True, "column_coverage_unchanged": True,
        "training_performed": False, "model_inference_performed": False, "gate_changed": False,
        "cuda_initialized": False, "real_benchmark_success": False,
        "arms": {"original": {"cases": 256, "queried_columns": 22464, "uci_only_columns": 1, "cases_with_uci_only_columns": 1},
                 "subsampled": {"cases": 256, "queried_columns": 22464, "uci_only_columns": 8, "cases_with_uci_only_columns": 7}}})


@pytest.mark.parametrize("failure", [None, "hash", "missing", "seed", "mask", "totals", "columns", "real_data"])
def test_query_evidence_requires_complete_unchanged_paired_synthetic_coverage(tmp_path, failure):
    evidence_fixture(tmp_path)
    if failure in ("missing", "seed", "totals"):
        records = training.balanced.read(tmp_path / "cases.json")
        if failure == "missing":
            records.pop()
        elif failure == "seed":
            records[0]["seed"] += 1
        else:
            records[0]["subsampled_uci_only_columns"] += 1
        write_json(tmp_path / "cases.json", records)
    elif failure in ("hash", "columns"):
        report = training.balanced.read(tmp_path / "summary.json")
        if failure == "hash":
            report["config_sha256"] = "changed"
        else:
            report["arms"]["subsampled"]["queried_columns"] -= 1
        write_json(tmp_path / "summary.json", report)
    elif failure == "mask":
        torch.save([True], tmp_path / "query_masks.pt")
    elif failure == "real_data":
        config = training.balanced.read(tmp_path / "config.json")
        config["real_benchmark_inputs_used"] = True
        write_json(tmp_path / "config.json", config)
    if failure:
        with pytest.raises(ValueError):
            training.query_evidence(tmp_path)
    else:
        assert training.query_evidence(tmp_path)["subsampled_uci_only_columns"] == 8


def test_registration_preserves_priors_controls_architecture_and_original_gate(tmp_path):
    recipe, evidence, campaign = [tmp_path / name for name in ("v7", "evidence", "campaign")]
    old = area_config(version=7)
    write_json(recipe / "config.json", old)
    write_json(recipe / "registration.json", {"config_sha256": sha256(recipe / "config.json")})
    write_json(recipe / "status.json", {"state": "training_complete"})
    write_json(campaign / "status.json", {"state": "not_promoted"})
    write_json(campaign / "cache_launcher_config.json", {"run": str(recipe)})
    evidence_fixture(evidence)
    config = training.make_config(recipe, evidence, campaign)
    training.validate_recipe(config)
    for name in ("parent_checkpoint", "model", "training_teacher_policy", "probability_loss_scales", "column_objective",
                 "spectral_amplitude_mixture", "rating_training_shapes", "confirmation", "primary_metrics", "retention_metrics"):
        assert config[name] == old[name]
    assert {**config["phases"][0], "name": old["phases"][0]["name"]} == old["phases"][0]
    assert len(config["validation_cases"]) == 56 and len(config["confirmation_cases"]) == 92
    old_seeds = {case["seed"] for case in old["validation_cases"] + old["confirmation_cases"]}
    new_seeds = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    assert not old_seeds & new_seeds and len(new_seeds) == 148


@pytest.mark.parametrize("failure", ["seed", "budget", "weight", "prior", "gate", "bank", "steps"])
def test_recipe_mutations_are_rejected(failure):
    config = fixture_config()
    if failure == "seed":
        config["query_seed_offset"] += 1
    elif failure == "budget":
        config["query_budget_objective"]["query_budgets"].append(32)
    elif failure == "weight":
        config["query_budget_objective"]["full_query_weight"] = .1
    elif failure == "prior":
        config["spectral_amplitude_mixture"]["scaled_probability"] = 1.
    elif failure == "gate":
        config["retention_metrics"].pop()
    elif failure == "bank":
        config["validation_cases"][0]["seed"] += 1
    else:
        config["steps"] = 500
    with pytest.raises(ValueError):
        training.validate_recipe(config)


@pytest.mark.parametrize("failure", [None, "training_budget", "scaled", "missing"])
def test_confirmation_rejects_any_training_query_adapter_or_scaled_prior(failure):
    bank = [{"metadata": {"domain": "imputation", "area_prior": {"requested_regime": "legacy", "applied": False}}}
            for _ in range(92)]
    if failure == "training_budget":
        bank[0]["metadata"]["query_budget_objective"] = {}
    elif failure == "scaled":
        bank[0]["metadata"]["area_prior"]["applied"] = True
    elif failure == "missing":
        bank.pop()
    if failure:
        with pytest.raises(ValueError):
            confirmation.verify_legacy_cases(bank)
    else:
        confirmation.verify_legacy_cases(bank)


def test_step_zero_cannot_enter_benchmarks(tmp_path):
    run, output = tmp_path / "run", tmp_path / "confirmation"
    config = fixture_config()
    write_json(run / "config.json", config)
    write_json(run / "registration.json", {"config_sha256": sha256(run / "config.json")})
    write_json(run / "status.json", {"state": "training_complete"})
    torch.save({"step": 0, "config_sha256": sha256(run / "config.json")}, run / "best_validation_checkpoint.pth")
    confirmation.confirm(run, output)
    assert not training.balanced.read(output / "gate.json")["passed"]
    assert not training.balanced.read(output / "manifest.json")["query_budget_applied"]


@pytest.mark.parametrize("failure", [None, "ratio", "rating", "zero_gradient", "seed"])
def test_full_preflights_require_real_rescaled_query_objectives_and_unchanged_ratings(tmp_path, failure):
    config = fixture_config()
    write_json(tmp_path / "config.json", config)
    cases = []
    for index, shape in enumerate(([2000, 1000], [1000, 2000], [2000, 1000])):
        case = {"seed": config["preflight_seed"] + index, "rows": shape[0], "columns": shape[1],
                "grad_norm": .2, "loss": .8, "teacher": {"target": "initialization" if index < 2 else "specialist"}}
        if index < 2:
            seed = case["seed"] + config["query_seed_offset"]
            full = {"weight": 1., "thresholds": [1e-3, 1e-4], "query_cells": 500000}
            subset = {"weight": 1., "thresholds": [1e-3, 1e-4], "query_cells": 2000}
            case.update(area_prior={"applied": True, "reason": "dimension_scaled",
                                    "signal_gain": training.area.signal_gain(2000, 1000)},
                        column_objective=full,
                        query_budget_objective={"seed": seed, "budget": training.budget_for_seed(seed),
                                                "full": full, "subsampled": subset, "full_query_weight": .5,
                                                "subsampled_query_weight": .5, "maximum_cell_coefficient": 500.,
                                                "full_to_subsampled_query_ratio": 250.})
        cases.append(case)
    if failure == "ratio":
        cases[0]["query_budget_objective"]["full_to_subsampled_query_ratio"] = 1.
    elif failure == "rating":
        cases[2]["query_budget_objective"] = {}
    elif failure == "zero_gradient":
        cases[0]["grad_norm"] = 0.
    elif failure == "seed":
        cases[1]["query_budget_objective"]["seed"] += 1
    write_json(tmp_path / "preflight.json", {"passed": True, "config_sha256": sha256(tmp_path / "config.json"), "cases": cases})
    if failure:
        with pytest.raises(ValueError):
            training.validate_preflight(config, tmp_path)
    else:
        training.validate_preflight(config, tmp_path)
