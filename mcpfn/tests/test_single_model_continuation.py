import json
from types import SimpleNamespace

import pytest
import torch

import confirm_single_matrixformer_continuation as confirmation
import train_single_matrixformer_continuation as continuation
from recsys_evaluation_common import sha256, write_json
from single_model_distillation import Episode


def metrics(imputation=1., rmse=1., mae=1.):
    return {"imputation": {"rmse": imputation, "mae": imputation}, "recsys": {"rmse": rmse, "mae": mae}}


def test_selector_rejects_hidden_tradeoffs_and_allows_step_zero():
    baseline = metrics()
    score, ratios, eligible = continuation.selection_score(baseline, baseline, baseline)
    assert score == 1. and eligible and len(ratios) == 3
    score, _, eligible = continuation.selection_score(metrics(.9, .95, 1.001), baseline, baseline)
    assert score == 1.001 and not eligible
    score, _, eligible = continuation.selection_score(metrics(.9, .95, .96), baseline, baseline)
    assert score == .96 and eligible
    with pytest.raises(ValueError):
        continuation.selection_score(metrics(rmse=float("nan")), baseline, baseline)


def test_banks_have_disjoint_seeds_and_large_rating_cases():
    selection = continuation.bank_specs(15100501)
    confirmation = continuation.bank_specs(16100501, confirmation=True)
    assert len(selection) == 28 and len(confirmation) == 46
    assert {case["seed"] for case in selection}.isdisjoint(case["seed"] for case in confirmation)
    for cases in (selection, confirmation):
        assert {case["shape"] for case in cases if case["domain"] == "recsys"} == {(1000, 1000), (2000, 1000)}
        assert len({case["seed"] for case in cases}) == len(cases)


def test_rating_training_alternates_context_sizes_without_changing_imputation():
    calls = []
    sampler = SimpleNamespace(sample=lambda domain, step, **keywords: calls.append((domain, step, keywords)))
    config = {"rating_training_shapes": [[1000, 1000], [2000, 1000]]}
    for step in range(1, 5):
        continuation.training_episode(sampler, config, "recsys", step)
    continuation.training_episode(sampler, config, "imputation", 3)
    assert [call[2]["shape"] for call in calls[:4]] == [(1000, 1000), (2000, 1000)] * 2
    assert calls[-1] == ("imputation", 3, {})


def test_confirmation_requires_improvement_and_both_control_retentions():
    references = metrics()
    assert confirmation.confirmation_gate(metrics(.9, .9, .9), metrics(), metrics(), references)["passed"]
    assert not confirmation.confirmation_gate(metrics(), metrics(), metrics(), references)["passed"]
    assert not confirmation.confirmation_gate(metrics(.8, .8, 1.01), metrics(), metrics(), references)["passed"]
    gate = confirmation.confirmation_gate(metrics(.9, .9, .9), metrics(), metrics(.85, 1., 1.), references)
    assert gate["initialization_retained"] and not gate["general_retained"] and not gate["passed"]


def test_confirmation_waiter_rejects_failed_or_stale_training(tmp_path, monkeypatch):
    write_json(tmp_path / "status.json", {"state": "training_complete"})
    assert confirmation.training_ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="not completed"):
        confirmation.training_ready(tmp_path)
    write_json(tmp_path / "status.json", {"state": "training", "pid": 123})
    monkeypatch.setattr(confirmation.os, "kill", lambda *args: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(RuntimeError, match="process is missing"):
        confirmation.training_ready(tmp_path)


def synthetic_screen(directory):
    config = {"input_sha256": {}, "source_sha256": {}, "base_training_config": "base.json", "base_config_sha256": "base",
              "protocol": "protocol.json", "protocol_sha256": "protocol",
              "cases": [{"domain": "imputation", "seed": 1}, {"domain": "recsys", "seed": 2}],
              "candidates": [{"name": "first"}, {"name": "second"}]}
    write_json(directory / "status.json", {"state": "complete"})
    write_json(directory / "config.json", config)
    write_json(directory / "reference.json", {"bank_sha256": "bank", "selection_references": metrics()})
    for candidate in config["candidates"]:
        cases = [{"index": index, **case, "student": {"rmse": 1., "mae": 1.}} for index, case in enumerate(config["cases"])]
        write_json(directory / "candidates" / f"{candidate['name']}.json",
                   {"candidate": candidate, "config_sha256": "config", "bank_sha256": "bank", "cases": cases,
                    "score": 1., "metrics": metrics(), "ratios": {"imputation_rmse": 1., "recsys_rmse": 1., "recsys_mae": 1.}})
    selected = {"selected": {"name": "first"}, "training_performed": False, "real_benchmark_inputs_used": False,
                "config_sha256": "config", "bank_sha256": "bank", "checkpoint": "checkpoint.pth", "checkpoint_sha256": "checkpoint",
                "candidates_completed": 2, "cases_per_candidate": 2, "score": 1., "metrics": metrics(),
                "ratios": {"imputation_rmse": 1., "recsys_rmse": 1., "recsys_mae": 1.}}
    write_json(directory / "selection.json", selected)
    return selected


def test_selection_audit_enforces_complete_bank_and_exact_tie_break(tmp_path, monkeypatch):
    selected = synthetic_screen(tmp_path)
    monkeypatch.setattr(continuation, "verify_hashes", lambda hashes: None)
    assert continuation.verified_selection(tmp_path)[1] == selected
    selected["selected"] = {"name": "second"}
    write_json(tmp_path / "selection.json", selected)
    with pytest.raises(ValueError, match="registered synthetic winner"):
        continuation.verified_selection(tmp_path)
    write_json(tmp_path / "status.json", {"state": "screening"})
    with pytest.raises(ValueError, match="complete"):
        continuation.verified_selection(tmp_path)


@pytest.mark.parametrize("field,value", [("seed", 99), ("domain", "recsys"), ("index", 1)])
def test_selection_audit_rejects_changed_case_identity(tmp_path, monkeypatch, field, value):
    synthetic_screen(tmp_path)
    monkeypatch.setattr(continuation, "verify_hashes", lambda hashes: None)
    path = tmp_path / "candidates/first.json"
    report = continuation.read(path)
    report["cases"][0][field] = value
    write_json(path, report)
    with pytest.raises(ValueError, match="registered synthetic bank"):
        continuation.verified_selection(tmp_path)


def test_selection_audit_rejects_aggregate_not_matching_cases(tmp_path, monkeypatch):
    synthetic_screen(tmp_path)
    monkeypatch.setattr(continuation, "verify_hashes", lambda hashes: None)
    path = tmp_path / "candidates/first.json"
    report = continuation.read(path)
    report["metrics"]["recsys"]["mae"] = .1
    write_json(path, report)
    with pytest.raises(ValueError, match="does not reproduce"):
        continuation.verified_selection(tmp_path)


def test_cpu_training_saves_single_student_and_synthetic_bank(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)

    def sample(domain, step, **specification):
        truth = torch.zeros(1, 2, 2)
        mask = torch.ones_like(truth, dtype=torch.bool)
        return Episode(truth.masked_fill(mask, torch.nan), truth, mask,
                       {"domain": domain, "gradient_data": "synthetic", "seed": specification.get("seed", step)})

    sampler = SimpleNamespace(sample=sample)
    bank = [continuation.cache_episode(sample(domain, 0)) for domain in ("imputation", "recsys")]
    torch.save(bank, tmp_path / "synthetic_bank.pt")
    reference = {"bank_sha256": sha256(tmp_path / "synthetic_bank.pt"), "references": metrics()}
    write_json(tmp_path / "reference.json", reference)
    config = {"steps": 2, "peak_lr": .1, "warmup_steps": 0, "validate_every": 1, "grad_clip": 1., "validation_cases": [],
              "rating_training_shapes": [[1000, 1000], [2000, 1000]]}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "preflight.json", {"config_sha256": sha256(tmp_path / "config.json")})
    monkeypatch.setattr(continuation, "verify_run", lambda *args: None)
    monkeypatch.setattr(continuation, "storage_guard", lambda *args: None)
    monkeypatch.setattr(continuation, "setup", lambda config: (model, optimizer, None, [], sampler))
    monkeypatch.setattr(continuation, "prepare_bank", lambda *args: (bank, reference))
    monkeypatch.setattr(continuation, "teacher_target", lambda *args: (None, {}))
    monkeypatch.setattr(continuation, "learning_rate", lambda *args: .1)
    monkeypatch.setattr(continuation, "rng_state", lambda: {"cpu": torch.get_rng_state(), "cuda": []})
    monkeypatch.setattr(continuation.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(continuation, "streamed_point_predictions",
                        lambda model, source, criterion, **kwargs: torch.ones_like(source) * model.weight[0, 0])
    evaluate_bank = continuation.evaluate_bank
    monkeypatch.setattr(continuation, "evaluate_bank", lambda model, criterion, bank: evaluate_bank(model, criterion, bank, device="cpu"))

    def backward(model, criterion, episode, target, weight):
        loss = model.weight.square().mean()
        (weight * loss).backward()
        return float(loss.detach())

    monkeypatch.setattr(continuation, "backward_episode", backward)
    continuation.train(config, tmp_path)
    status = continuation.read(tmp_path / "status.json")
    assert status["state"] == "training_complete" and status["best_step"] == 2
    state = torch.load(tmp_path / "best_validation_checkpoint.pth", weights_only=True)
    assert list(state["model"]) == ["weight"] and state["step"] == 2
    assert "teachers" not in state and "initialization_only" not in state
    rows = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all([case["domain"] for case in row["microsteps"]] == ["imputation", "recsys"] for row in rows)
    assert model.training
    assert sha256(tmp_path / "synthetic_bank.pt") == reference["bank_sha256"]
