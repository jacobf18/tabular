from copy import deepcopy

import pytest

from recsys_evaluation_common import sha256, write_json
import single_model_column_objective as column
import train_single_matrixformer_column_aligned as training


def evidence_fixture(root, parent_recipe):
    config = {"run": str(parent_recipe), "case_indices": list(range(30)),
              "pooled_rmse_parity": {"rtol": 1e-6, "atol": 1e-6},
              "input_sha256": {}, "source_sha256": {}, "checkpoint_sha256": {},
              "training_performed": False, "checkpoint_selection_performed": False,
              "real_benchmark_data_used": False, "gate_changed": False}
    write_json(root / "config.json", config)
    write_json(root / "registration.json", {"config_sha256": sha256(root / "config.json")})
    cases = []
    for index in range(30):
        scores = {role: {"pooled_rmse": .8, "missbench_nrmse": .8, "uci_nrmse": .8}
                  for role in training.diagnostic.ROLES}
        scores["initialization"] = {"pooled_rmse": 1., "missbench_nrmse": 1., "uci_nrmse": 1.}
        scores["candidate"] = {"pooled_rmse": 1.01, "missbench_nrmse": .99, "uci_nrmse": .98}
        case = {"index": index, "seed": index, "domain": "imputation", "gradient_data": "synthetic",
                "prior": "low_rank_mcar", "scores": scores}
        cases.append(case)
        write_json(root / "cases" / f"{index:03d}.json", case)
    rows = [row for row in training.diagnostic.full.cpu.summarize(cases) if row["family"] == "ALL"]
    summary = {"complete": True, "cases": 30, "pooled_parity_checks": 120, "pooled_parity_passed": True,
               "model_state_unchanged": True, "training_performed": False, "checkpoint_selection_performed": False,
               "gate_changed": False, "config_sha256": sha256(root / "config.json"), "summary": rows}
    write_json(root / "summary.json", summary)
    write_json(root / "status.json", {"state": "complete"})
    write_json(root / "saved_prediction_audit.json", {"complete": True, "prediction_cases_audited": 30,
               "all_saved_scores_reproduce": True, "cuda_initialized": False, "original_gate_changed": False,
               "config_sha256": sha256(root / "config.json"), "summary_sha256": sha256(root / "summary.json")})


@pytest.mark.parametrize("failure", [None, "case_index", "score", "gate_changed", "audit_hash", "wrong_parent"])
def test_evidence_binds_the_completed_full_context_order_reversal(tmp_path, failure):
    parent_recipe = tmp_path / "v5"
    root = tmp_path / "evidence"
    evidence_fixture(root, parent_recipe)
    if failure in ("case_index", "score"):
        case = training.balanced.read(root / "cases/000.json")
        if failure == "case_index":
            case["index"] = 7
        else:
            case["scores"]["candidate"]["pooled_rmse"] = .5
        write_json(root / "cases/000.json", case)
    elif failure == "gate_changed":
        summary = training.balanced.read(root / "summary.json")
        summary["gate_changed"] = True
        write_json(root / "summary.json", summary)
    elif failure == "audit_hash":
        audit = training.balanced.read(root / "saved_prediction_audit.json")
        audit["summary_sha256"] = "wrong"
        write_json(root / "saved_prediction_audit.json", audit)
    elif failure == "wrong_parent":
        parent_recipe = tmp_path / "other"
    if failure:
        with pytest.raises(ValueError):
            training.evidence(root, parent_recipe)
    else:
        assert training.evidence(root, parent_recipe)["directory"] == str(root)


def test_registration_preserves_teacher_priors_and_old_retention_with_fresh_larger_banks(tmp_path):
    recipe, diagnostic, campaign = [tmp_path / name for name in ("v5", "diagnostic", "campaign")]
    config = {"version": 5, "source_sha256": {}, "input_sha256": {}, "teacher_config_sha256": {},
              "training_teacher_policy": dict(training.parent.TEACHER_POLICY),
              "probability_loss_scales": dict(training.balanced.PROBABILITY_SCALES),
              "parent_checkpoint": "unchanged confirmed V2", "model": {"unchanged": True},
              "steps": 400, "validate_every": 50, "peak_lr": 1e-5, "warmup_steps": 20,
              "weight_decay": .02, "grad_clip": .5, "rating_training_shapes": [[1000, 1000], [2000, 1000]],
              "phases": [{"name": "parent_teacher", "end_step": 400, "rows_low": 256, "rows_high": 2000,
                          "cols_low": 128, "cols_high": 1000, "batch_size": 1}],
              "validation_cases": training.continuation.bank_specs(33100501),
              "confirmation_cases": training.continuation.bank_specs(34100501, confirmation=True)}
    write_json(recipe / "config.json", config)
    write_json(recipe / "registration.json", {"config_sha256": sha256(recipe / "config.json")})
    write_json(recipe / "status.json", {"state": "training_complete"})
    write_json(campaign / "cache_launcher_config.json", {"run": str(recipe)})
    write_json(campaign / "status.json", {"state": "not_promoted"})
    evidence_fixture(diagnostic, recipe)
    new = training.make_config(recipe, diagnostic, campaign)
    training.validate_recipe(new)
    for field in ("parent_checkpoint", "model", "steps", "validate_every", "peak_lr", "warmup_steps",
                  "weight_decay", "grad_clip", "rating_training_shapes", "training_teacher_policy", "probability_loss_scales"):
        assert new[field] == config[field]
    assert {key: value for key, value in new["phases"][0].items() if key != "name"} == {
        key: value for key, value in config["phases"][0].items() if key != "name"}
    old_seeds = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    new_seeds = [case["seed"] for case in new["validation_cases"] + new["confirmation_cases"]]
    assert len(new["validation_cases"]) == 56 and len(new["confirmation_cases"]) == 92
    assert len(set(new_seeds)) == 148 and old_seeds.isdisjoint(new_seeds)
    assert new["data_seed"] + 16 * new["steps"] + 1 < min(new_seeds)
    assert {("imputation", "rmse"), ("recsys", "rmse"), ("recsys", "mae")} <= set(map(tuple, new["retention_metrics"]))
    output = tmp_path / "registered"
    training.balanced.register(new, output)
    training.continuation.verify_run(new, output)
    assert not (output / "best_validation_checkpoint.pth").exists()
    changed = deepcopy(new)
    changed["retention_metrics"].pop()
    with pytest.raises(ValueError, match="registered"):
        training.validate_recipe(changed)


def test_new_bank_replicas_retain_all_contexts_and_families():
    for confirmation, expected in ((False, 56), (True, 92)):
        bank = training.repeated_bank(39100501, confirmation=confirmation)
        half = expected // 2
        assert len(bank) == expected
        for first, second in zip(bank[:half], bank[half:]):
            assert {key: value for key, value in first.items() if key != "seed"} == {
                key: value for key, value in second.items() if key != "seed"}
            assert second["seed"] == first["seed"] + 1000
