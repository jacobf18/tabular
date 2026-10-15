from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from recsys_evaluation_common import sha256, write_json
from single_model_distillation import Episode, distillation_loss
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_streamed_training import assert_gradients_match, example
import train_single_matrixformer_probability_balance as balanced


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
def test_balanced_loss_changes_only_probability_gradient(domain):
    torch.manual_seed(121)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    logits = torch.randn(11, 20, requires_grad=True)
    truth = torch.randn(11)
    teacher = torch.randn(11, requires_grad=True)
    point = 2. * functional.huber_loss(criterion.mean(logits), teacher.detach(), delta=1., reduction="none")
    original = distillation_loss(criterion, logits, truth, teacher, nll_weight=.1 if domain == "recsys" else 1.)
    scale = balanced.PROBABILITY_SCALES[domain]
    actual = balanced.balanced_loss(criterion, logits, truth, teacher, domain=domain,
                                    probability_scales=balanced.PROBABILITY_SCALES)
    torch.testing.assert_close(actual, point + scale * (original - point))
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    original_gradient = torch.autograd.grad(original.sum(), logits, retain_graph=True)[0]
    point_gradient = torch.autograd.grad(point.sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, point_gradient + scale * (original_gradient - point_gradient))
    actual.mean().backward()
    assert teacher.grad is None
    if domain == "recsys":
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0., -1., float("nan"), 1.01])
def test_invalid_probability_scales_rejected_before_forward(scale):
    with pytest.raises(ValueError, match="Probability scale"):
        balanced.balanced_loss(None, None, None, None, domain="imputation",
                               probability_scales={"imputation": scale, "recsys": 1.})


def test_rejects_unknown_domain_and_non_synthetic_episode():
    with pytest.raises(ValueError, match="explicit"):
        balanced.balanced_loss(None, None, None, None, domain="unknown", probability_scales=balanced.PROBABILITY_SCALES)
    episode = Episode(None, None, None, {"gradient_data": "real"})
    with pytest.raises(ValueError, match="synthetic"):
        balanced.backward_episode(None, None, episode, None, .5, probability_scales=balanced.PROBABILITY_SCALES)


@pytest.mark.parametrize("domain", ["imputation", "recsys"])
def test_actual_backward_matches_dense_gradients(domain):
    reference, source, truth, mask = example(shape=(1, 9, 7))
    model = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth, requires_grad=True)
    episode = Episode(source, truth, mask, {"domain": domain, "gradient_data": "synthetic"})
    expected = balanced.balanced_loss(criterion, reference(source, decode_mask=mask), truth[mask], teacher[mask],
                                     domain=domain, probability_scales=balanced.PROBABILITY_SCALES).mean()
    actual = balanced.backward_episode(model, criterion, episode, teacher, .5,
                                        probability_scales=balanced.PROBABILITY_SCALES)
    (.5 * expected).backward()
    assert actual == pytest.approx(float(expected.detach()), abs=1e-6)
    assert_gradients_match(reference, model)
    assert teacher.grad is None


def test_adapter_restores_original_backward_on_failure(tmp_path, monkeypatch):
    original = balanced.continuation.backward_episode

    def interrupted(config, output, resume):
        assert balanced.continuation.backward_episode.func is balanced.backward_episode
        assert balanced.continuation.backward_episode.keywords["probability_scales"] == balanced.PROBABILITY_SCALES
        raise RuntimeError("interrupted")

    monkeypatch.setattr(balanced.continuation, "train", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        balanced.train({"probability_loss_scales": dict(balanced.PROBABILITY_SCALES)}, tmp_path)
    assert balanced.continuation.backward_episode is original


def test_cpu_training_uses_balanced_backward_and_saves_one_network(tmp_path, monkeypatch):
    runner = balanced.continuation
    model, source, truth, mask = example(shape=(1, 9, 7))
    original_state = deepcopy(model.state_dict())
    original_backward = runner.backward_episode
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

    def sample(domain, step, **specification):
        return Episode(source, truth, mask, {"domain": domain, "gradient_data": "synthetic", "seed": step})

    bank = [runner.cache_episode(sample(domain, 0)) for domain in ("imputation", "recsys")]
    torch.save(bank, tmp_path / "synthetic_bank.pt")
    reference = {"bank_sha256": sha256(tmp_path / "synthetic_bank.pt"),
                 "references": {domain: {"rmse": 1., "mae": 1.} for domain in ("imputation", "recsys")}}
    write_json(tmp_path / "reference.json", reference)
    config = {"steps": 2, "peak_lr": .01, "warmup_steps": 0, "validate_every": 1, "grad_clip": 1.,
              "validation_cases": [], "rating_training_shapes": [[1000, 1000], [2000, 1000]],
              "probability_loss_scales": dict(balanced.PROBABILITY_SCALES)}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "preflight.json", {"config_sha256": sha256(tmp_path / "config.json")})
    monkeypatch.setattr(runner, "verify_run", lambda *args: None)
    monkeypatch.setattr(runner, "storage_guard", lambda *args: None)
    monkeypatch.setattr(runner, "setup", lambda config: (model, optimizer, criterion, [], SimpleNamespace(sample=sample)))
    monkeypatch.setattr(runner, "prepare_bank", lambda *args: (bank, reference))
    monkeypatch.setattr(runner, "teacher_target", lambda *args: (torch.zeros_like(truth), {}))
    monkeypatch.setattr(runner, "rng_state", lambda: {"cpu": torch.get_rng_state(), "cuda": []})
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    evaluate_bank = runner.evaluate_bank
    monkeypatch.setattr(runner, "evaluate_bank", lambda *args: evaluate_bank(*args, device="cpu"))
    balanced.train(config, tmp_path)
    assert runner.read(tmp_path / "status.json")["state"] == "training_complete"
    latest = torch.load(tmp_path / "latest_checkpoint.pth", weights_only=True)
    assert latest["step"] == 2 and set(latest["model"]) == set(original_state)
    assert any(not torch.equal(value, original_state[name]) for name, value in latest["model"].items())
    assert all(torch.isfinite(value).all() for value in latest["model"].values())
    selected = torch.load(tmp_path / "best_validation_checkpoint.pth", weights_only=True)
    assert "teachers" not in selected and "optimizer" not in selected
    assert runner.backward_episode is original_backward
    assert sha256(tmp_path / "synthetic_bank.pt") == reference["bank_sha256"]


def diagnostic_fixture(directory, ratio=3.):
    config = {"input_sha256": {}, "source_sha256": {}, "training_performed": False,
              "real_benchmark_inputs_used": False, "case_indices": list(range(8)) + list(range(16, 28)),
              "precision": "BF16 streamed full contexts"}
    write_json(directory / "config.json", config)
    summary = {"complete": True, "cases": 20, "model_state_unchanged": True,
               "config_sha256": sha256(directory / "config.json"),
               "groups": {"imputation": {"mean_gradient": {"probability_to_point_norm": ratio}}}}
    (directory / "summary.json").write_text(json.dumps(summary))
    write_json(directory / "status.json", {"state": "complete"})


@pytest.mark.parametrize("ratio", [None, 0., 2., float("nan")])
def test_registration_requires_full_context_support(tmp_path, ratio):
    diagnostic_fixture(tmp_path, ratio)
    with pytest.raises(ValueError, match="does not support"):
        balanced.diagnostic_evidence(tmp_path)


def test_registration_rejects_unfinished_or_mutated_diagnostic(tmp_path):
    diagnostic_fixture(tmp_path)
    assert balanced.diagnostic_evidence(tmp_path)["imputation_probability_to_point_norm"] == 3.
    write_json(tmp_path / "status.json", {"state": "diagnosing"})
    with pytest.raises(ValueError, match="completed"):
        balanced.diagnostic_evidence(tmp_path)
    write_json(tmp_path / "status.json", {"state": "complete"})
    summary = balanced.read(tmp_path / "summary.json")
    summary["model_state_unchanged"] = False
    write_json(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match="immutable"):
        balanced.diagnostic_evidence(tmp_path)


def test_campaign_waiter_fails_closed_and_accepts_only_terminal_success(tmp_path, monkeypatch):
    for state in ("complete", "not_promoted"):
        write_json(tmp_path / "status.json", {"state": state})
        assert balanced.campaign_ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="stopped"):
        balanced.campaign_ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "evaluating", "pid": 123})
    monkeypatch.setattr(balanced.os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(RuntimeError, match="missing"):
        balanced.campaign_ready(tmp_path)


def parent_fixture(root):
    parent, confirmation, diagnostic, campaign = [root / name for name in ("parent", "confirmation", "diagnostic", "campaign")]
    config = {"input_sha256": {}, "source_sha256": {}, "teacher_config_sha256": {},
              "phases": [{"name": "previous", "end_step": 400}],
              "validation_cases": [{"seed": 15100501}], "confirmation_cases": [{"seed": 16100501}],
              "initializer_screen": "old screen", "initializer_selection": {"selected": "old parent"},
              "rating_training_shapes": [[1000, 1000], [2000, 1000]], "model": {"unchanged": True},
              "selection": "unchanged minimax retention policy", "confirmation": "unchanged strict gate"}
    write_json(parent / "config.json", config)
    write_json(parent / "registration.json", {"config_sha256": sha256(parent / "config.json")})
    write_json(parent / "status.json", {"state": "training_complete", "best_step": 350})
    torch.save({"model": {"weight": torch.ones(2)}, "step": 350}, parent / "best_validation_checkpoint.pth")
    write_json(confirmation / "status.json", {"state": "complete"})
    write_json(confirmation / "gate.json", {"passed": True, "initialization_retained": True,
                                            "general_retained": True, "selected_step": 350,
                                            "checkpoint_sha256": sha256(parent / "best_validation_checkpoint.pth")})
    diagnostic_fixture(diagnostic)
    write_json(campaign / "config.json", {"policy": "no score input"})
    return parent, confirmation, diagnostic, campaign


def test_registration_keeps_frozen_model_and_gate_and_uses_fresh_seeds(tmp_path):
    parent, confirmation, diagnostic, campaign = parent_fixture(tmp_path)
    config = balanced.make_config(parent, confirmation, diagnostic, campaign)
    original = balanced.read(parent / "config.json")
    for field in ("model", "selection", "confirmation", "rating_training_shapes"):
        assert config[field] == original[field]
    assert config["probability_loss_scales"] == {"imputation": .1, "recsys": 1.}
    assert config["steps"] == 400 and config["peak_lr"] == 1e-5
    assert config["parent_checkpoint"] == str(parent / "best_validation_checkpoint.pth")
    assert "initializer_selection" not in config and "initializer_screen" not in config
    validation = {case["seed"] for case in config["validation_cases"]}
    confirmation_seeds = {case["seed"] for case in config["confirmation_cases"]}
    assert len(validation) == 28 and len(confirmation_seeds) == 46
    assert validation.isdisjoint(confirmation_seeds)
    assert config["data_seed"] + 16 * config["steps"] + 1 < min(validation)
    output = tmp_path / "candidate"
    balanced.register(config, output)
    balanced.continuation.verify_run(config, output)
    assert balanced.read(output / "status.json")["state"] == "registered"
    assert not (output / "best_validation_checkpoint.pth").exists()
    with pytest.raises(FileExistsError):
        balanced.register(config, output)


def test_registration_rejects_parent_checkpoint_not_bound_to_gate(tmp_path):
    parent, confirmation, diagnostic, campaign = parent_fixture(tmp_path)
    torch.save({"model": {"weight": torch.zeros(2)}, "step": 350}, parent / "best_validation_checkpoint.pth")
    with pytest.raises(ValueError, match="synthetic-confirmed"):
        balanced.make_config(parent, confirmation, diagnostic, campaign)
