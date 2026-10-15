from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import train_single_matrixformer_domain_teachers as domain
from recsys_evaluation_common import sha256, write_json
from test_single_model_probability_balance import parent_fixture


@pytest.mark.parametrize("kind,expected", [("imputation", 0), ("recsys", 1)])
def test_only_the_declared_domain_expert_is_called(kind, expected, monkeypatch):
    models = [torch.nn.Linear(2, 2).eval().requires_grad_(False) for _ in range(2)]
    episode = SimpleNamespace(source=torch.zeros(1, 2, 2),
                              metadata={"domain": kind, "gradient_data": "synthetic"})
    called = []

    def predict(model, source, criterion, **options):
        called.append(model)
        return torch.ones_like(source)

    monkeypatch.setattr(domain, "streamed_point_predictions", predict)
    monkeypatch.setattr(domain, "compute_context", lambda device: nullcontext())
    target, routing = domain.domain_teacher_target(models, None, episode)
    assert called == [models[expected]] and not target.requires_grad
    assert routing["specialist_weight"] == float(expected)
    assert routing["target"] == domain.TEACHER_POLICY[kind]


@pytest.mark.parametrize("failure", ["real", "unknown_domain", "training_teacher", "gradient_teacher"])
def test_teacher_rejects_wrong_data_or_unfrozen_experts(failure):
    models = [torch.nn.Linear(2, 2).eval().requires_grad_(False) for _ in range(2)]
    episode = SimpleNamespace(source=torch.zeros(1, 2, 2),
                              metadata={"domain": "imputation", "gradient_data": "synthetic"})
    if failure == "real":
        episode.metadata["gradient_data"] = "real"
    elif failure == "unknown_domain":
        episode.metadata["domain"] = "other"
    elif failure == "training_teacher":
        models[1].train()
    else:
        models[0].requires_grad_(True)
    with pytest.raises(ValueError):
        domain.domain_teacher_target(models, None, episode)


@pytest.mark.parametrize("failure", [False, True])
def test_adapter_changes_training_not_reference_bank_and_always_restores(monkeypatch, failure):
    original_target = domain.continuation.teacher_target
    original_preflight = domain.balanced.teacher_target

    def bank():
        assert domain.continuation.teacher_target is original_target
        if failure:
            raise RuntimeError("bank interrupted")
        return "original blend reference"

    monkeypatch.setattr(domain.continuation, "prepare_bank", bank)
    try:
        with domain.training_teachers():
            assert domain.continuation.teacher_target is domain.domain_teacher_target
            assert domain.balanced.teacher_target is domain.domain_teacher_target
            assert domain.continuation.prepare_bank() == "original blend reference"
            assert domain.continuation.teacher_target is domain.domain_teacher_target
    except RuntimeError:
        assert failure
    assert domain.continuation.teacher_target is original_target
    assert domain.continuation.prepare_bank is bank
    assert domain.balanced.teacher_target is original_preflight


def evidence_fixture(root):
    root.mkdir()
    config = {"training_performed": False, "selection_performed": False,
              "real_benchmark_data_used": False, "case_indices": list(range(24)),
              "input_sha256": {}, "source_sha256": {}, "checkpoint_sha256": {}}
    write_json(root / "config.json", config)
    rows = [{"role": role, "pooled_rmse": value, "missbench_nrmse": value, "uci_nrmse": value}
            for role, value in (("general", 1.), ("observed_blend", 1.01))]
    write_json(root / "summary.json", {"complete": True, "cases": 24, "model_state_unchanged": True,
                                       "cuda_initialized": False, "training_performed": False,
                                       "checkpoint_selection_performed": False,
                                       "config_sha256": sha256(root / "config.json"), "summary": rows})
    write_json(root / "status.json", {"state": "complete"})
    for index in range(24):
        write_json(root / "cases" / f"{index:03d}.json",
                   {"index": index, "seed": index, "gradient_data": "synthetic", "domain": "imputation",
                    "scores": {row["role"]: row for row in rows}})


def test_evidence_requires_reproducible_completed_synthetic_comparisons(tmp_path):
    root = tmp_path / "diagnostic"
    evidence_fixture(root)
    assert domain.teacher_evidence(root)["cases"] == 24
    case = domain.balanced.read(root / "cases/000.json")
    case["scores"]["general"]["pooled_rmse"] = .5
    write_json(root / "cases/000.json", case)
    with pytest.raises(ValueError, match="does not reproduce"):
        domain.teacher_evidence(root)


@pytest.mark.parametrize("field", ["cuda_initialized", "training_performed", "checkpoint_selection_performed"])
def test_evidence_rejects_unregistered_execution_or_selection(tmp_path, field):
    root = tmp_path / "diagnostic"
    evidence_fixture(root)
    summary = domain.balanced.read(root / "summary.json")
    summary[field] = True
    write_json(root / "summary.json", summary)
    with pytest.raises(ValueError, match="immutable synthetic"):
        domain.teacher_evidence(root)


def test_registration_is_a_fixed_parent_sibling_with_new_seeds_and_unchanged_gates(tmp_path):
    parent, confirmation, gradients, previous_campaign = parent_fixture(tmp_path)
    probability = tmp_path / "probability"
    base = domain.balanced.make_config(parent, confirmation, gradients, previous_campaign)
    domain.balanced.register(base, probability)
    diagnostic = tmp_path / "teacher_diagnostic"
    evidence_fixture(diagnostic)
    campaign = tmp_path / "probability_benchmarks"
    write_json(campaign / "cache_launcher_config.json", {"run": str(probability)})
    config = domain.make_config(probability, diagnostic, campaign)
    for field in ("model", "parent_checkpoint", "selection", "confirmation", "steps", "peak_lr",
                  "warmup_steps", "probability_loss_scales", "rating_training_shapes", "objective"):
        assert config[field] == base[field]
    assert config["training_teacher_policy"] == domain.TEACHER_POLICY
    assert "checkpoint.pth" not in " ".join(key for key in config["input_sha256"] if str(probability) in key)
    old = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    new = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    assert len(new) == 74 and old.isdisjoint(new)
    output = tmp_path / "candidate"
    domain.balanced.register(config, output)
    domain.continuation.verify_run(config, output)
    assert not (output / "best_validation_checkpoint.pth").exists()


def test_train_rejects_an_unregistered_teacher_policy():
    with pytest.raises(ValueError, match="registered"):
        domain.train({"training_teacher_policy": {"imputation": "blend", "recsys": "specialist"}}, None)
