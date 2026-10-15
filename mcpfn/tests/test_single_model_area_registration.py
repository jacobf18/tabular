from copy import deepcopy

import pytest
import torch

import confirm_single_matrixformer_area_mixture as confirmation
from recsys_evaluation_common import sha256, write_csv, write_json
import train_single_matrixformer_area_mixture as training


def evidence_fixture(root):
    shapes = [[240, 80], [512, 256], [1000, 1000], [2000, 1000], [1000, 2000]]
    config = {"reference_area": 19200, "shapes": shapes, "source_sha256": {}, "input_sha256": {},
              "training_performed": False, "checkpoint_selection_performed": False,
              "real_benchmark_data_used": False, "gate_changed": False}
    write_json(root / "config.json", config)
    write_json(root / "registration.json", {"config_sha256": sha256(root / "config.json")})
    write_json(root / "status.json", {"state": "complete"})
    write_json(root / "summary.json", {"complete": True, "config_sha256": sha256(root / "config.json"),
        "paired_matrices": 240, "regime_rows": 480, "draws": 48, "orthonormal_energy_checks_passed": True,
        "dimension_scaling_checks_passed": True, "training_performed": False, "gate_changed": False,
        "cuda_initialized": False, "real_benchmark_success": False})
    write_json(root / "draws.json", [{}] * 48)
    rows = [{"component": component, "replica": replica, "rows": rows, "columns": columns, "regime": regime,
             "expected_snr_relative_to_reference": 1. if regime == "area_scaled" else 19200 / (rows * columns)}
            for component in range(6) for replica in range(8) for rows, columns in shapes for regime in ("legacy", "area_scaled")]
    write_csv(root / "per_shape.csv", rows)
    write_csv(root / "by_family_shape.csv", [{"fixture": True}])


def fixture_config(version=6):
    result = {"version": version, "source_sha256": {}, "input_sha256": {}, "teacher_config_sha256": {},
              "training_teacher_policy": dict(training.parent.TEACHER_POLICY),
              "probability_loss_scales": dict(training.balanced.PROBABILITY_SCALES),
              "column_objective": deepcopy(training.column.COLUMN_OBJECTIVE), "bank_replicates": 2,
              "primary_metrics": [list(pair) for pair in training.column.PRIMARY_METRICS],
              "retention_metrics": [list(pair) for pair in training.column.RETENTION_METRICS],
              "parent_checkpoint": "unchanged confirmed V2", "model": {"unchanged": True},
              "steps": 400, "validate_every": 50, "peak_lr": 1e-5, "warmup_steps": 20,
              "weight_decay": .02, "grad_clip": .5, "rating_training_shapes": [[1000, 1000], [2000, 1000]],
              "phases": [{"name": "column_aligned", "end_step": 400, "rows_low": 256, "rows_high": 2000,
                          "cols_low": 128, "cols_high": 1000, "batch_size": 1}],
              "confirmation": "unchanged six-metric gate",
              "validation_cases": training.previous.repeated_bank(38100501),
              "confirmation_cases": training.previous.repeated_bank(39100501, confirmation=True)}
    if version == 7:
        result.update(**training.SEEDS, spectral_amplitude_mixture=deepcopy(training.area.SPECIFICATION),
                      validation_cases=training.previous.repeated_bank(training.SEEDS["validation_seed"]),
                      confirmation_cases=training.previous.repeated_bank(training.SEEDS["confirmation_seed"], confirmation=True))
    return result


@pytest.mark.parametrize("failure", [None, "missing", "duplicate", "ratio", "real_data", "summary_hash"])
def test_prior_evidence_requires_complete_unmodified_synthetic_mechanism_probe(tmp_path, failure):
    evidence_fixture(tmp_path)
    if failure in ("missing", "duplicate", "ratio"):
        import csv
        with (tmp_path / "per_shape.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if failure == "missing":
            rows.pop()
        elif failure == "duplicate":
            rows[-1] = rows[0]
        else:
            rows[-1]["expected_snr_relative_to_reference"] = 2.
        write_csv(tmp_path / "per_shape.csv", rows)
    elif failure == "real_data":
        config = training.balanced.read(tmp_path / "config.json")
        config["real_benchmark_data_used"] = True
        write_json(tmp_path / "config.json", config)
    elif failure == "summary_hash":
        summary = training.balanced.read(tmp_path / "summary.json")
        summary["config_sha256"] = "wrong"
        write_json(tmp_path / "summary.json", summary)
    if failure:
        with pytest.raises(ValueError):
            training.prior_evidence(tmp_path)
    else:
        assert training.prior_evidence(tmp_path)["directory"] == str(tmp_path)


def test_registration_changes_only_training_prior_and_seeds_not_legacy_task_or_initialization(tmp_path):
    recipe, evidence, campaign = [tmp_path / name for name in ("v6", "evidence", "campaign")]
    old = fixture_config()
    write_json(recipe / "config.json", old)
    write_json(recipe / "registration.json", {"config_sha256": sha256(recipe / "config.json")})
    write_json(recipe / "status.json", {"state": "training_complete"})
    write_json(campaign / "cache_launcher_config.json", {"run": str(recipe)})
    write_json(campaign / "status.json", {"state": "not_promoted"})
    evidence_fixture(evidence)
    config = training.make_config(recipe, evidence, campaign)
    training.validate_recipe(config)
    for field in ("parent_checkpoint", "model", "steps", "validate_every", "peak_lr", "warmup_steps", "weight_decay", "grad_clip",
                  "rating_training_shapes", "training_teacher_policy", "probability_loss_scales", "column_objective",
                  "primary_metrics", "retention_metrics", "confirmation"):
        assert config[field] == old[field]
    for field in ("validation_cases", "confirmation_cases"):
        assert [{key: value for key, value in case.items() if key != "seed"} for case in config[field]] == [
            {key: value for key, value in case.items() if key != "seed"} for case in old[field]]
    assert {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}.isdisjoint(
        {case["seed"] for case in old["validation_cases"] + old["confirmation_cases"]})
    output = tmp_path / "registered"
    training.balanced.register(config, output)
    training.continuation.verify_run(config, output)
    assert not (output / "best_validation_checkpoint.pth").exists()


@pytest.mark.parametrize("failure", ["probability", "validation_regime", "retention", "seed", "shape", "steps", "loss"])
def test_recipe_rejects_changes_to_registered_mixture_legacy_banks_or_retention(failure):
    config = fixture_config(7)
    training.validate_recipe(config)
    if failure == "probability":
        config["spectral_amplitude_mixture"]["scaled_probability"] = 1.
    elif failure == "validation_regime":
        config["spectral_amplitude_mixture"]["validation_regime"] = "area_scaled"
    elif failure == "retention":
        config["retention_metrics"].pop()
    elif failure == "seed":
        config["data_seed"] = 123
    elif failure == "shape":
        config["confirmation_cases"][0]["shape"] = [128, 64]
    elif failure == "steps":
        config["steps"] = 800
    else:
        config["column_objective"]["weight"] = 0.
    with pytest.raises(ValueError):
        training.validate_recipe(config)


@pytest.mark.parametrize("failure", [None, "unscaled", "seed", "rating_scaled", "loss"])
def test_preflight_requires_both_scaled_full_orientations_and_unchanged_rating(failure, tmp_path):
    config = fixture_config(7)
    write_json(tmp_path / "config.json", config)
    cases = [{"seed": config["preflight_seed"] + index, "grad_norm": 1., "rows": shape[0], "columns": shape[1],
              "teacher": {"target": "initialization" if index < 2 else "specialist"}}
             for index, shape in enumerate(((2000, 1000), (1000, 2000), (2000, 1000)))]
    for case in cases[:2]:
        case.update(area_prior={"applied": True, "reason": "dimension_scaled", "signal_gain": training.area.signal_gain(2000, 1000)},
                    column_objective={"weight": 1.})
    if failure == "unscaled":
        cases[0]["area_prior"]["applied"] = False
    elif failure == "seed":
        cases[1]["seed"] = 3
    elif failure == "rating_scaled":
        cases[2]["area_prior"] = {}
    elif failure == "loss":
        cases[0]["column_objective"]["weight"] = 0.
    write_json(tmp_path / "preflight.json", {"passed": True, "config_sha256": sha256(tmp_path / "config.json"), "cases": cases})
    if failure:
        with pytest.raises(ValueError):
            training.validate_preflight(config, tmp_path)
    else:
        training.validate_preflight(config, tmp_path)


def test_step_zero_confirmation_cannot_build_a_bank_or_promote(tmp_path, monkeypatch):
    run, output = tmp_path / "run", tmp_path / "confirmation"
    config = fixture_config(7)
    write_json(run / "config.json", config)
    write_json(run / "registration.json", {"config_sha256": sha256(run / "config.json")})
    write_json(run / "status.json", {"state": "training_complete"})
    torch.save({"step": 0, "config_sha256": sha256(run / "config.json")}, run / "best_validation_checkpoint.pth")
    monkeypatch.setattr(confirmation, "evaluation_setup", lambda *args: pytest.fail("Unchanged initialization must not be evaluated"))
    confirmation.confirm(run, output)
    assert not training.balanced.read(output / "gate.json")["passed"]
    assert training.balanced.read(output / "manifest.json")["prior_regime"] == "legacy"
    assert not (output / "synthetic_bank.pt").exists()


def test_confirmation_rejects_a_scaled_prior_even_if_it_would_help_scores(tmp_path, monkeypatch):
    run, output = tmp_path / "run", tmp_path / "confirmation"
    config = fixture_config(7)
    write_json(run / "config.json", config)
    write_json(run / "registration.json", {"config_sha256": sha256(run / "config.json")})
    write_json(run / "status.json", {"state": "training_complete"})
    torch.save({"step": 50, "config_sha256": sha256(run / "config.json")}, run / "best_validation_checkpoint.pth")
    monkeypatch.setattr(confirmation, "evaluation_setup", lambda *args: (None, None, None, None, None))
    values = {"imputation": {name: 1. for name in ("rmse", "missbench_nrmse", "uci_nrmse", "common_nrmse")},
              "recsys": {"rmse": 1., "mae": 1.}}
    reference = {"general": values, "teacher": values, "references": deepcopy(values)}
    bank = [{"metadata": {"domain": "imputation", "area_prior": {"requested_regime": "area_scaled", "applied": True}}}]
    monkeypatch.setattr(training.continuation, "prepare_bank", lambda *args: (bank, reference))
    with pytest.raises(ValueError, match="modified training prior"):
        confirmation.confirm(run, output)
    assert not (output / "gate.json").exists()
