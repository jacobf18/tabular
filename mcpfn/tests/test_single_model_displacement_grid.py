from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import single_model_displacement_grid as grid
from test_single_model_contraction_lifecycle import TinySampler, TinyStudent, specifications


def test_grid_is_fixed_order_complete_and_has_known_control():
    assert len(grid.GRID) == 12
    assert [(spec["alpha"], spec["beta"]) for spec in grid.GRID] == [
        (alpha, beta) for alpha in (.1, .25, .4) for beta in (0., .025, .05, .1)]
    assert grid.GRID[4] == {"id": "candidate_04", "alpha": .25, "beta": 0.}
    cases = grid.confirmation_cases()
    assert len(cases) == len({case.get("structured", case)["seed"] for case in cases}) == 284


@pytest.mark.parametrize("specification", grid.GRID)
def test_parameter_arithmetic_keeps_one_network(specification):
    parent = {"weight": torch.tensor([1., -2., 0.]), "integer": torch.tensor(2)}
    first = {"weight": torch.tensor([.5, -1., .25]), "integer": torch.tensor(2)}
    second = {"weight": torch.tensor([.75, -3., -.25]), "integer": torch.tensor(2)}
    candidate = grid.combine(parent, first, second, specification)
    details = grid.check_arithmetic(parent, first, second, candidate, specification)
    assert details["arithmetic_independently_reconstructed"] and details["tensors"] == 2
    assert details["changed_tensors"] == 1
    assert torch.equal(parent["weight"], torch.tensor([1., -2., 0.]))
    candidate["weight"][0] += .01
    with pytest.raises(ValueError, match="reconstruction differs"):
        grid.check_arithmetic(parent, first, second, candidate, specification)


@pytest.mark.parametrize("failure", ["integer", "shape", "dtype", "nonfinite", "inventory", "coefficient"])
def test_invalid_candidate_sources_or_coefficients_fail(failure):
    parent = {"weight": torch.ones(2), "integer": torch.tensor(1)}
    first, second, specification = deepcopy(parent), deepcopy(parent), deepcopy(grid.GRID[0])
    if failure == "integer":
        second["integer"] += 1
    elif failure == "shape":
        second["weight"] = torch.ones(3)
    elif failure == "dtype":
        second["weight"] = torch.ones(2, dtype=torch.float64)
    elif failure == "nonfinite":
        parent["weight"][0] = float("nan")
    elif failure == "inventory":
        second["extra"] = torch.zeros(1)
    else:
        specification["alpha"] = .15
    with pytest.raises(ValueError):
        grid.combine(parent, first, second, specification)


def test_selection_requires_all_candidates_and_uses_registered_ties():
    decisions = {spec["id"]: {"passed": False, "score": .1} for spec in grid.GRID}
    assert not grid.choose(decisions)["passed"]
    for specification in grid.GRID[2:5]:
        decisions[specification["id"]] = {"passed": True, "score": 1.}
    chosen = grid.choose(decisions)
    assert chosen["selected"] == grid.GRID[2]
    decisions[grid.GRID[4]["id"]]["score"] = .9
    assert grid.choose(decisions)["selected"] == grid.GRID[4]
    with pytest.raises(ValueError, match="complete grid"):
        grid.choose(dict(reversed(list(decisions.items()))))
    decisions.pop(grid.GRID[-1]["id"])
    with pytest.raises(ValueError, match="complete grid"):
        grid.choose(decisions)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    producer, run, output = [tmp_path / name for name in ("producer", "grid", "qualification")]
    for directory in (producer, run, output):
        directory.mkdir()
    parent_path, first_path, second_path = [tmp_path / name for name in ("parent.pth", "first.pth", "second.pth")]
    monkeypatch.setattr(grid, "PRODUCER", producer)
    monkeypatch.setattr(grid, "PARENT", parent_path)
    monkeypatch.setattr(grid, "TAILS", (first_path, second_path))
    base = {"query_seed_offset": 0, "retention_metrics": [list(pair) for pair in grid.mixture.column.RETENTION_METRICS],
            "validation_cases": specifications(100)}
    config = {"base": base, "confirmation_cases": specifications(10000, confirmation=True)}
    grid.write_json(run / "config.json", config)
    monkeypatch.setattr(grid, "verify_registration", lambda directory: deepcopy(config))
    state = {"phase": None, "mode": "pass", "audit_failure": None, "events": []}
    sampler = TinySampler()

    def setup(current):
        return TinyStudent(1.), object(), object(), [TinyStudent(1.2), TinyStudent(1.1)], sampler

    def predict(model, source, criterion, **options):
        value = float(model.weight.detach()[0])
        if state["phase"] == "confirmation":
            if state["mode"] == "confirmation_regression" and value < 1.:
                value = 1.5
            if state["mode"] == "general_regression" and abs(value - 1.2) < 1e-6:
                value = .1
        return torch.linspace(-1., 1., 12).reshape_as(source) + value

    original_runtime = grid.mixture.runtime

    @contextmanager
    def runtime(current, directory):
        previous = state["phase"]
        state["phase"] = directory.name
        try:
            with original_runtime(current, directory):
                evaluate = grid.mixture.backend.evaluate_bank

                def cpu_evaluate(model, criterion, bank):
                    state["events"].append(("evaluate", state["phase"], float(model.weight.detach()[0]), len(bank)))
                    return evaluate(model, criterion, bank, device="cpu")

                grid.mixture.backend.evaluate_bank = cpu_evaluate
                yield
        finally:
            state["phase"] = previous

    monkeypatch.setattr(grid.mixture, "original_setup", setup)
    monkeypatch.setattr(grid.mixture, "StructuredSampler", lambda current: sampler)
    monkeypatch.setattr(grid.mixture, "streamed_point_predictions", predict)
    monkeypatch.setattr(grid.mixture, "reference_teacher_target", lambda teachers, criterion, episode:
                        (predict(teachers[1], episode.source, criterion), {"target": "unit-test fixture"}))
    monkeypatch.setattr(grid.mixture, "runtime", runtime)
    with runtime(base, producer):
        student, optimizer, criterion, teachers, active_sampler = grid.mixture.backend.setup(base)
        bank, reference = grid.mixture.backend.prepare_bank(active_sampler, teachers, criterion, base["validation_cases"], producer)
        initial, initial_cases = grid.mixture.backend.evaluate_bank(student, criterion, bank)
    grid.write_json(producer / "validation_0.json", {"student": initial, "cases": initial_cases})
    state["events"].clear()

    def prepare(mode="pass", audit_failure=None):
        state.update(mode=mode, audit_failure=audit_failure)
        parent = TinyStudent(1.).state_dict()
        tail = TinyStudent(2. if mode == "selection_regression" else 0.).state_dict()
        for path, model in ((parent_path, parent), (first_path, tail), (second_path, tail)):
            torch.save({"model": model}, path)
        receipts = {}
        for specification in grid.GRID:
            directory = run / specification["id"]
            directory.mkdir()
            path = directory / "checkpoint.pth"
            torch.save({"model": grid.combine(parent, tail, tail, specification),
                        **grid.metadata(specification, grid.sha256(run / "config.json"))}, path)
            receipts[specification["id"]] = {"checkpoint_sha256": grid.sha256(path)}
        grid.write_json(run / "build.json", {"complete": True, "candidates": receipts})

    def audit(current_run, current_output, phase):
        state["events"].append(("audit", phase))
        if state["audit_failure"] == phase:
            directory = current_output / phase
            if phase == "selection":
                directory /= grid.GRID[0]["id"]
            record = grid.read(directory / "student.json")
            record["student"]["imputation"]["rmse"] += 1.
            grid.write_json(directory / "student.json", record)
        grid.audit(current_run, current_output, phase)

    monkeypatch.setattr(grid, "audit_subprocess", audit)
    return SimpleNamespace(run=run, output=output, state=state, prepare=prepare)


def test_full_grid_selection_confirmation_and_raw_audits(lifecycle):
    lifecycle.prepare()
    grid.qualify(lifecycle.run, lifecycle.output)
    gate = grid.read(lifecycle.output / "gate.json")
    assert gate["passed"] and gate["confirmation_evaluated"]
    assert gate["selected_candidate"] == grid.GRID[-1]
    assert gate["selected_step"] == 400 and gate["gradient_updates_performed_here"] == 0
    assert (lifecycle.run / "best_validation_checkpoint.pth").resolve() == lifecycle.run / "candidate_11/checkpoint.pth"
    selection_audit = grid.read(lifecycle.output / "selection/audit.json")
    assert selection_audit["prediction_grids_rescored"] == 2280 and selection_audit["candidates_audited"] == 12
    assert grid.read(lifecycle.output / "confirmation/audit.json")["prediction_grids_rescored"] == 1136
    events = lifecycle.state["events"]
    assert len(events) == 16
    assert events[12:] == [("audit", "selection"), ("evaluate", "confirmation", 1., 284),
                           ("evaluate", "confirmation", .5, 284), ("audit", "confirmation")]
    assert "Real-benchmark success is not assessed" in (lifecycle.output / "RESULTS.md").read_text()


@pytest.mark.parametrize("mode", ["selection_regression", "confirmation_regression", "general_regression"])
def test_rejection_has_no_promotion_or_fallback(lifecycle, mode):
    lifecycle.prepare(mode)
    grid.qualify(lifecycle.run, lifecycle.output)
    gate = grid.read(lifecycle.output / "gate.json")
    assert not gate["passed"] and not (lifecycle.run / "best_validation_checkpoint.pth").exists()
    assert gate["confirmation_evaluated"] == (mode != "selection_regression")
    assert len([event for event in lifecycle.state["events"] if event[:2] == ("evaluate", "selection")]) == 12
    confirmations = [event for event in lifecycle.state["events"] if event[:2] == ("evaluate", "confirmation")]
    assert len(confirmations) == (0 if mode == "selection_regression" else 2)
    if mode == "general_regression":
        assert gate["initialization_retained"] and not gate["structured_general_retained"]


@pytest.mark.parametrize("phase", ["selection", "confirmation"])
def test_raw_score_tampering_blocks_promotion(lifecycle, phase):
    lifecycle.prepare(audit_failure=phase)
    with pytest.raises(ValueError, match="Original aggregation differs"):
        grid.qualify(lifecycle.run, lifecycle.output)
    assert not (lifecycle.output / "gate.json").exists()
    assert not (lifecycle.run / "best_validation_checkpoint.pth").exists()
    if phase == "selection":
        assert not (lifecycle.output / "confirmation").exists()


def test_report_cardinality_is_checked_before_scoring():
    with pytest.raises(ValueError, match="exactly the registered cases"):
        grid.audited_metrics([{}], {"cases": []}, ("student",))


def test_qualification_receipt_blocks_reusing_confirmation_under_new_output(tmp_path):
    run, first, second = [tmp_path / name for name in ("run", "first", "second")]
    grid.write_json(run / "config.json", {})
    grid.write_json(run / "status.json", {"state": "built_not_qualified"})
    grid.start_qualification(run, first)
    assert grid.read(run / "status.json")["state"] == "qualifying"
    with pytest.raises(ValueError, match="exactly once"):
        grid.start_qualification(run, second)
    grid.write_json(run / "status.json", {"state": "built_not_qualified"})
    with pytest.raises(FileExistsError):
        grid.start_qualification(run, second)
    assert not second.exists()


@pytest.mark.parametrize("field", ["grid", "selection", "confirmation_cases", "confirmation_seeds", "step_semantics", "source"])
def test_registration_binds_grid_gates_fresh_cases_and_sources(tmp_path, monkeypatch, field):
    run, producer = tmp_path / "run", tmp_path / "producer"
    monkeypatch.setattr(grid, "PRODUCER", producer)
    monkeypatch.setattr(grid.fixed, "validate_recipe", lambda config: None)
    base = {"model": {"test": True}}
    grid.write_json(producer / "config.json", base)
    source = tmp_path / "source.py"
    source.write_text("original\n")
    config = {"grid": deepcopy(grid.GRID), "parent": str(grid.PARENT), "tails": list(map(str, grid.TAILS)),
              "producer": str(producer), "selection": grid.POLICY, "base": base, "model": base["model"],
              "confirmation_cases": grid.confirmation_cases(), "confirmation_seeds": deepcopy(grid.SEEDS),
              "step_semantics": grid.SEMANTICS, "input_sha256": {}, "source_sha256": {str(source): grid.sha256(source)}}
    grid.write_json(run / "config.json", config)
    grid.write_json(run / "registration.json", {"config_sha256": grid.sha256(run / "config.json")})
    grid.verify_registration(run)
    if field == "source":
        source.write_text("changed\n")
    else:
        if field == "grid":
            config[field][0]["alpha"] = .15
        elif field == "confirmation_cases":
            config[field].pop()
        elif field == "confirmation_seeds":
            config[field]["legacy"] += 1
        else:
            config[field] = "changed"
        grid.write_json(run / "config.json", config)
        with pytest.raises(ValueError, match="registration changed"):
            grid.verify_registration(run)
        grid.write_json(run / "registration.json", {"config_sha256": grid.sha256(run / "config.json")})
    with pytest.raises((ValueError, RuntimeError)):
        grid.verify_registration(run)
