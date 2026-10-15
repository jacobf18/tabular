from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import single_model_fixed_contraction as contraction
from single_model_distillation import Episode


class TinyStudent(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))


def specifications(seed, confirmation=False):
    imputation, ratings, per_family = (60, 32, 16) if confirmation else (40, 16, 8)
    result = [{"domain": "imputation" if index < imputation else "recsys", "seed": seed + index}
              for index in range(imputation + ratings)]
    for family in contraction.mixture.FAMILIES:
        for repetition in range(per_family):
            result.append({"domain": "imputation", "structured": {"family": family, "seed": seed + len(result)}})
    return result


class TinySampler:
    def sample(self, domain, step, seed=None, structured=None):
        metadata = {"domain": domain, "gradient_data": "synthetic", "seed": seed}
        if structured is not None:
            metadata.update(structured=True, **structured)
        target = torch.linspace(-1., 1., 12).reshape(1, 4, 3)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, -1] = False
        source = target.masked_fill(mask, torch.nan)
        ratings = None
        if domain == "recsys":
            ratings = SimpleNamespace(means=torch.full((3,), 3.), stds=torch.full((3,), .5),
                                      row_offsets=torch.zeros(4), row_scales=torch.linspace(.8, 1.2, 4))
        return Episode(source, target, mask, metadata, ratings)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    producer, run, output = [tmp_path / name for name in ("producer", "candidate", "qualification")]
    for directory in (producer, run, output):
        directory.mkdir()
    parent_path, tail_path = tmp_path / "parent.pth", tmp_path / "tail.pth"
    monkeypatch.setattr(contraction, "PRODUCER", producer)
    monkeypatch.setattr(contraction, "PARENT", parent_path)
    monkeypatch.setattr(contraction, "TAIL", tail_path)
    base = {"query_seed_offset": 0, "retention_metrics": [list(pair) for pair in contraction.mixture.column.RETENTION_METRICS],
            "validation_cases": specifications(100)}
    config = {"base": base, "confirmation_cases": specifications(10000, confirmation=True)}
    contraction.write_json(run / "config.json", config)
    monkeypatch.setattr(contraction, "verify_registration", lambda directory: deepcopy(config))
    state = {"phase": None, "mode": "pass", "audit_failure": None, "events": []}
    sampler = TinySampler()

    def setup(current):
        state["events"].append(("setup", state["phase"]))
        return TinyStudent(1.), object(), object(), [TinyStudent(1.2), TinyStudent(1.1)], sampler

    def predict(model, source, criterion, **options):
        value = float(model.weight.detach()[0])
        if state["phase"] == "confirmation":
            if state["mode"] == "confirmation_regression" and abs(value - .75) < 1e-6:
                value = 1.25
            if state["mode"] == "general_regression" and abs(value - 1.2) < 1e-6:
                value = .5
        return torch.linspace(-1., 1., 12).reshape_as(source) + value

    original_runtime = contraction.mixture.runtime

    @contextmanager
    def runtime(current, directory):
        previous_phase = state["phase"]
        state["phase"] = directory.name
        try:
            with original_runtime(current, directory):
                evaluate = contraction.mixture.backend.evaluate_bank

                def cpu_evaluate(model, criterion, bank):
                    state["events"].append(("evaluate", state["phase"], float(model.weight.detach()[0]), len(bank)))
                    return evaluate(model, criterion, bank, device="cpu")

                contraction.mixture.backend.evaluate_bank = cpu_evaluate
                yield
        finally:
            state["phase"] = previous_phase

    monkeypatch.setattr(contraction.mixture, "original_setup", setup)
    monkeypatch.setattr(contraction.mixture, "StructuredSampler", lambda current: sampler)
    monkeypatch.setattr(contraction.mixture, "streamed_point_predictions", predict)
    monkeypatch.setattr(contraction.mixture, "reference_teacher_target", lambda teachers, criterion, episode:
                        (predict(teachers[1], episode.source, criterion), {"target": "unit-test fixture"}))
    monkeypatch.setattr(contraction.mixture, "runtime", runtime)

    with runtime(base, producer):
        student, optimizer, criterion, teachers, active_sampler = contraction.mixture.backend.setup(base)
        bank, reference = contraction.mixture.backend.prepare_bank(active_sampler, teachers, criterion, base["validation_cases"], producer)
        initial, initial_cases = contraction.mixture.backend.evaluate_bank(student, criterion, bank)
    contraction.write_json(producer / "validation_0.json", {"student": initial, "cases": initial_cases})
    state["events"].clear()

    def prepare(mode="pass", audit_failure=None):
        state.update(mode=mode, audit_failure=audit_failure)
        parent = TinyStudent(1.).state_dict()
        tail = TinyStudent(2. if mode == "selection_regression" else 0.).state_dict()
        torch.save({"model": parent}, parent_path)
        torch.save({"model": tail}, tail_path)
        checkpoint = run / "contracted_checkpoint.pth"
        torch.save({"model": contraction.interpolate(parent, tail), "step": 400, "epoch": 400,
            "config_sha256": contraction.sha256(run / "config.json"), "derived_checkpoint": True,
            "gradient_updates_performed_here": 0, "alpha": .25, "step_semantics": contraction.SEMANTICS}, checkpoint)
        contraction.write_json(run / "build.json", {"checkpoint_sha256": contraction.sha256(checkpoint)})

    def audit(current_run, current_output, phase):
        state["events"].append(("audit", phase))
        if state["audit_failure"] == phase:
            record = contraction.read(current_output / phase / "student.json")
            record["student"]["imputation"]["rmse"] += 1.
            contraction.write_json(current_output / phase / "student.json", record)
        contraction.audit(current_run, current_output, phase)

    monkeypatch.setattr(contraction, "audit_subprocess", audit)
    return SimpleNamespace(run=run, output=output, state=state, prepare=prepare)


def test_development_failure_cannot_consume_confirmation_or_publish_alias(lifecycle):
    lifecycle.prepare("selection_regression")
    contraction.qualify(lifecycle.run, lifecycle.output)
    gate = contraction.read(lifecycle.output / "gate.json")
    assert not gate["passed"] and not gate["confirmation_evaluated"]
    assert not (lifecycle.output / "confirmation").exists()
    assert not (lifecycle.run / "best_validation_checkpoint.pth").exists()
    audit = contraction.read(lifecycle.output / "selection/audit.json")
    assert audit["cases"] == 152 and audit["prediction_grids_rescored"] == 608
    assert [event for event in lifecycle.state["events"] if event[0] == "audit"] == [("audit", "selection")]


@pytest.mark.parametrize("mode", ["confirmation_regression", "general_regression"])
def test_confirmation_rejection_cannot_publish_alias(lifecycle, mode):
    lifecycle.prepare(mode)
    contraction.qualify(lifecycle.run, lifecycle.output)
    gate = contraction.read(lifecycle.output / "gate.json")
    assert gate["confirmation_evaluated"] and not gate["passed"]
    assert not (lifecycle.run / "best_validation_checkpoint.pth").exists()
    assert contraction.read(lifecycle.output / "confirmation/audit.json")["prediction_grids_rescored"] == 1136
    if mode == "general_regression":
        assert gate["initialization_retained"]
        assert not gate["legacy_general_retained"] and not gate["structured_general_retained"]


def test_success_uses_fresh_initialization_and_requires_both_raw_audits(lifecycle):
    lifecycle.prepare()
    contraction.qualify(lifecycle.run, lifecycle.output)
    gate = contraction.read(lifecycle.output / "gate.json")
    assert gate["passed"] and gate["confirmation_evaluated"]
    assert gate["derived_checkpoint"] and gate["gradient_updates_performed_here"] == 0
    assert (lifecycle.run / "best_validation_checkpoint.pth").resolve() == lifecycle.run / "contracted_checkpoint.pth"
    relevant = [event for event in lifecycle.state["events"] if event[0] in ("evaluate", "audit")]
    assert relevant == [("evaluate", "selection", .75, 152), ("audit", "selection"),
                        ("evaluate", "confirmation", 1., 284), ("evaluate", "confirmation", .75, 284),
                        ("audit", "confirmation")]
    assert "Real-benchmark success is not assessed" in (lifecycle.output / "RESULTS.md").read_text()


@pytest.mark.parametrize("phase", ["selection", "confirmation"])
def test_failed_raw_audit_blocks_promotion_and_later_phases(lifecycle, phase):
    lifecycle.prepare(audit_failure=phase)
    with pytest.raises(ValueError, match="Original aggregation differs"):
        contraction.qualify(lifecycle.run, lifecycle.output)
    assert not (lifecycle.output / "gate.json").exists()
    assert not (lifecycle.run / "best_validation_checkpoint.pth").exists()
    if phase == "selection":
        assert not (lifecycle.output / "confirmation").exists()
