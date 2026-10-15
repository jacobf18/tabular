from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recsys_evaluation_common import sha256, write_json
from single_model_distillation import Episode
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_single_model_domain_teachers import evidence_fixture as small_evidence_fixture
from test_single_model_probability_balance import parent_fixture
from test_streamed_training import example
import train_single_matrixformer_domain_teachers as domain
import train_single_matrixformer_parent_teacher as parent


def models_and_episode(kind="imputation"):
    models = [torch.nn.Linear(2, 2).eval().requires_grad_(False) for _ in range(3)]
    episode = SimpleNamespace(source=torch.ones(1, 2, 2),
                              metadata={"domain": kind, "gradient_data": "synthetic"})
    return models, episode


@pytest.mark.parametrize("kind,expected", [("imputation", 0), ("recsys", 2)])
def test_target_uses_frozen_initialization_or_specialist_not_old_general(kind, expected, monkeypatch):
    models, episode = models_and_episode(kind)
    called = []

    def predict(model, source, criterion, **options):
        called.append(model)
        return model(source)

    monkeypatch.setattr(parent, "streamed_point_predictions", predict)
    monkeypatch.setattr(parent, "compute_context", lambda device: nullcontext())
    target, routing = parent.parent_teacher_target(models[0], models[1:], None, episode)
    assert called == [models[expected]] and not target.requires_grad
    assert routing["target"] == parent.TEACHER_POLICY[kind]
    assert routing["specialist_weight"] == float(kind == "recsys")


@pytest.mark.parametrize("failure", ["real", "unknown", "parent_training", "parent_gradient", "expert_gradient", "absent"])
def test_rejects_invalid_episode_or_unfrozen_teachers(failure):
    models, episode = models_and_episode()
    if failure == "real":
        episode.metadata["gradient_data"] = "real"
    elif failure == "unknown":
        episode.metadata["domain"] = "unknown"
    elif failure == "parent_training":
        models[0].train()
    elif failure == "parent_gradient":
        models[0].requires_grad_(True)
    elif failure == "expert_gradient":
        models[1].requires_grad_(True)
    else:
        models[0] = None
    with pytest.raises(ValueError):
        parent.parent_teacher_target(models[0], models[1:], None, episode)


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_parent_is_independent_and_reference_bank_and_callbacks_are_preserved(monkeypatch, preflight, failure):
    models, episode = models_and_episode()
    student = torch.nn.Linear(2, 2).train()
    initial = deepcopy(student.state_dict())
    deployment_keys = set(initial)
    captured = []
    original_target = parent.continuation.teacher_target
    original_preflight_target = parent.balanced.teacher_target

    def setup(config):
        return student, None, None, models[1:], None

    def predict(model, source, criterion, **options):
        captured.append(model)
        return model(source)

    def bank():
        assert parent.continuation.teacher_target is original_target
        if failure:
            raise RuntimeError("reference interrupted")
        return "original observed blend"

    monkeypatch.setattr(parent.continuation, "setup", setup)
    monkeypatch.setattr(parent.balanced, "setup", setup)
    monkeypatch.setattr(parent.continuation, "prepare_bank", bank)
    monkeypatch.setattr(parent, "streamed_point_predictions", predict)
    monkeypatch.setattr(parent, "compute_context", lambda device: nullcontext())
    try:
        with parent.training_teachers():
            active = parent.balanced if preflight else parent.continuation
            with pytest.raises(RuntimeError, match="Initialize"):
                active.teacher_target(models[1:], None, episode)
            result = active.setup({})
            assert result[0] is student and result[3] == models[1:]
            before, _ = active.teacher_target(result[3], None, episode)
            retained = captured[-1]
            assert retained is not student and not retained.training
            assert not any(parameter.requires_grad for parameter in retained.parameters())
            for name, value in retained.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
                assert value.data_ptr() != student.state_dict()[name].data_ptr()
            with torch.no_grad():
                student.weight.add_(1.)
            after, _ = active.teacher_target(result[3], None, episode)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            assert student.training and set(student.state_dict()) == deployment_keys
            assert parent.continuation.prepare_bank() == "original observed blend"
    except RuntimeError as error:
        assert failure and "reference interrupted" in str(error)
    assert parent.continuation.setup is setup and parent.balanced.setup is setup
    assert parent.continuation.teacher_target is original_target
    assert parent.balanced.teacher_target is original_preflight_target
    assert parent.continuation.prepare_bank is bank


def full_evidence_fixture(root, checkpoint):
    config = {"input_sha256": {}, "source_sha256": {}, "checkpoint_sha256": {str(checkpoint): sha256(checkpoint)},
              "case_indices": list(range(30)), "checkpoints": {"initialization": str(checkpoint)},
              "training_performed": False, "checkpoint_selection_performed": False,
              "real_benchmark_data_used": False, "gate_changed": False}
    write_json(root / "config.json", config)
    rows = [{"role": role, "pooled_rmse": value, "missbench_nrmse": value, "uci_nrmse": value}
            for role, value in (("general", 1.), ("observed_blend", 1.01), ("initialization", .99), ("candidate", 1.005))]
    write_json(root / "summary.json", {"complete": True, "cases": 30, "small_cases": 24, "large_cases": 6,
                                       "model_state_unchanged": True, "pooled_parity_passed": True, "pooled_parity_checks": 120,
                                       "training_performed": False, "checkpoint_selection_performed": False, "gate_changed": False,
                                       "config_sha256": sha256(root / "config.json"), "summary": rows})
    write_json(root / "status.json", {"state": "complete"})
    for index in range(30):
        write_json(root / "cases" / f"{index:03d}.json",
                   {"index": index, "seed": index, "gradient_data": "synthetic", "domain": "imputation",
                    "scores": {row["role"]: row for row in rows}})


@pytest.mark.parametrize("field,value", [("cases", 24), ("pooled_parity_checks", 96), ("pooled_parity_passed", False),
                                         ("model_state_unchanged", False), ("training_performed", True),
                                         ("checkpoint_selection_performed", True), ("gate_changed", True)])
def test_evidence_rejects_incomplete_or_changed_diagnostics(tmp_path, field, value):
    checkpoint = tmp_path / "parent.pth"
    torch.save({}, checkpoint)
    root = tmp_path / "diagnostic"
    full_evidence_fixture(root, checkpoint)
    summary = parent.balanced.read(root / "summary.json")
    summary[field] = value
    write_json(root / "summary.json", summary)
    with pytest.raises(ValueError, match="immutable full-context"):
        parent.teacher_evidence(root, checkpoint)


def test_evidence_requires_matching_parent_and_reproducible_scores(tmp_path):
    checkpoint = tmp_path / "parent.pth"
    torch.save({}, checkpoint)
    root = tmp_path / "diagnostic"
    full_evidence_fixture(root, checkpoint)
    assert parent.teacher_evidence(root, checkpoint)["cases"] == 30
    with pytest.raises(ValueError, match="immutable full-context"):
        parent.teacher_evidence(root, tmp_path / "other.pth")
    case = parent.balanced.read(root / "cases/000.json")
    case["scores"]["initialization"]["pooled_rmse"] = 0.
    write_json(root / "cases/000.json", case)
    with pytest.raises(ValueError, match="does not reproduce"):
        parent.teacher_evidence(root, checkpoint)


def test_registration_keeps_v2_initialization_losses_contexts_and_all_original_gates(tmp_path):
    initial, confirmation, gradients, previous_campaign = parent_fixture(tmp_path)
    probability = tmp_path / "probability"
    base = parent.balanced.make_config(initial, confirmation, gradients, previous_campaign)
    parent.balanced.register(base, probability)
    small = tmp_path / "small"
    small_evidence_fixture(small)
    probability_campaign = tmp_path / "probability_campaign"
    write_json(probability_campaign / "cache_launcher_config.json", {"run": str(probability)})
    domain_run = tmp_path / "domain"
    domain_config = domain.make_config(probability, small, probability_campaign)
    parent.balanced.register(domain_config, domain_run)
    write_json(domain_run / "status.json", {"state": "training_complete"})
    campaign = tmp_path / "domain_campaign"
    write_json(campaign / "cache_launcher_config.json", {"run": str(domain_run)})
    write_json(campaign / "status.json", {"state": "not_promoted"})
    full = tmp_path / "full"
    full_evidence_fixture(full, Path(domain_config["parent_checkpoint"]))
    config = parent.make_config(domain_run, full, campaign)
    for field in ("model", "parent_checkpoint", "selection", "confirmation", "steps", "peak_lr",
                  "warmup_steps", "probability_loss_scales", "rating_training_shapes", "objective",
                  "reference_teacher_policy"):
        assert config[field] == domain_config[field]
    assert config["training_teacher_policy"] == parent.TEACHER_POLICY
    old = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]
           + domain_config["validation_cases"] + domain_config["confirmation_cases"]}
    new = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    assert len(new) == 74 and old.isdisjoint(new)
    assert config["data_seed"] + 16 * config["steps"] + 1 < min(new)
    output = tmp_path / "candidate"
    parent.balanced.register(config, output)
    parent.continuation.verify_run(config, output)
    assert not (output / "best_validation_checkpoint.pth").exists()


def test_train_rejects_other_teacher_policies():
    with pytest.raises(ValueError, match="registered"):
        parent.train({"training_teacher_policy": domain.TEACHER_POLICY}, None)


def test_actual_cpu_training_keeps_teachers_frozen_and_saves_only_one_network(tmp_path, monkeypatch):
    runner = parent.continuation
    student, source, truth, mask = example(shape=(1, 9, 7))
    initial = deepcopy(student.state_dict())
    teachers = [deepcopy(student).eval().requires_grad_(False) for _ in range(2)]
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    optimizer = torch.optim.SGD(student.parameters(), lr=.01)
    retained = []
    predict = parent.streamed_point_predictions

    def record_predict(model, *arguments, **keywords):
        if model not in teachers and not retained:
            retained.append(model)
        return predict(model, *arguments, **keywords)

    def sample(domain, step, **specification):
        return Episode(source, truth, mask, {"domain": domain, "gradient_data": "synthetic", "seed": step})

    bank = [runner.cache_episode(sample(kind, 0)) for kind in ("imputation", "recsys")]
    torch.save(bank, tmp_path / "synthetic_bank.pt")
    reference = {"bank_sha256": sha256(tmp_path / "synthetic_bank.pt"),
                 "references": {kind: {"rmse": 1., "mae": 1.} for kind in ("imputation", "recsys")}}
    write_json(tmp_path / "reference.json", reference)
    config = {"steps": 2, "peak_lr": .01, "warmup_steps": 0, "validate_every": 1, "grad_clip": 1.,
              "validation_cases": [], "rating_training_shapes": [[1000, 1000], [2000, 1000]],
              "probability_loss_scales": dict(parent.balanced.PROBABILITY_SCALES),
              "training_teacher_policy": dict(parent.TEACHER_POLICY)}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "preflight.json", {"config_sha256": sha256(tmp_path / "config.json")})
    monkeypatch.setattr(parent, "streamed_point_predictions", record_predict)
    monkeypatch.setattr(runner, "verify_run", lambda *args: None)
    monkeypatch.setattr(runner, "storage_guard", lambda *args: None)
    monkeypatch.setattr(runner, "setup", lambda config: (student, optimizer, criterion, teachers, SimpleNamespace(sample=sample)))
    monkeypatch.setattr(runner, "prepare_bank", lambda *args: (bank, reference))
    monkeypatch.setattr(runner, "rng_state", lambda: {"cpu": torch.get_rng_state(), "cuda": []})
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    evaluate_bank = runner.evaluate_bank
    monkeypatch.setattr(runner, "evaluate_bank", lambda *args: evaluate_bank(*args, device="cpu"))
    parent.train(config, tmp_path)
    assert runner.read(tmp_path / "status.json")["state"] == "training_complete"
    latest = torch.load(tmp_path / "latest_checkpoint.pth", weights_only=True)
    assert latest["step"] == 2 and set(latest["model"]) == set(initial)
    assert any(not torch.equal(value, initial[name]) for name, value in latest["model"].items())
    selected = torch.load(tmp_path / "best_validation_checkpoint.pth", weights_only=True)
    assert set(selected["model"]) == set(initial) and "teachers" not in selected
    assert len(retained) == 1
    for model in teachers + retained:
        assert not model.training and not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    assert sha256(tmp_path / "synthetic_bank.pt") == reference["bank_sha256"]
