from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import single_model_four_draw_trial_blend_replay as blend
import single_model_four_draw_trial_retention as retention
import single_model_four_draw_trial_storage as storage
import train_single_matrixformer_balanced_native_quarter as training
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.fixture(scope="module")
def schedule():
    return storage.budget.batches(training)


def test_complete_teacher_storage_matches_independent_area_ledger(schedule):
    assert blend.storage_accounting(schedule) == storage.teacher_storage(schedule)
    assert blend.storage_accounting(schedule)["raw_component_grids"] == 13312


def test_all_7168_ordered_teacher_tasks_and_admitted_subsets_are_retained(schedule, tmp_path):
    runtime = SimpleNamespace(OUTPUT=tmp_path)
    for steps in ([], [1], [2, 4, 128], list(range(1, 129))):
        counts = {"admitted_steps": steps, "training_tasks": 56 * len(steps)}
        tasks = blend.tasks_for(runtime, {"batches": schedule}, counts)
        assert len(tasks) == 56 * len(steps)
        assert [(step, task["index"]) for step, task in tasks.values()] == [(step, index) for step in steps for index in range(56)]
        for path, (step, task) in tasks.items():
            assert Path(path) == tmp_path / "training_batches" / f"step_{step:03d}" / f'task_{task["index"]:02d}.pt'


@pytest.mark.parametrize("steps,count", [([1], 28), ([0], 56), ([129], 56), ([1, 1], 112), ([3, 2], 112)])
def test_incomplete_or_invalid_admitted_teacher_coverage_is_rejected(schedule, tmp_path, steps, count):
    with pytest.raises(ValueError):
        blend.tasks_for(SimpleNamespace(OUTPUT=tmp_path), {"batches": schedule}, {"admitted_steps": steps, "training_tasks": count})


def test_teacher_storage_rejects_missing_native_structured_or_rating_tasks(schedule):
    for index in (0, 1, 13):
        shortened = deepcopy(schedule)
        shortened[0]["tasks"].pop(index)
        with pytest.raises(ValueError):
            blend.storage_accounting(shortened)


def test_retention_directory_and_component_budget_cover_every_registered_attempt(tmp_path):
    for step in (1, 64, 65, 128):
        assert retention.directory(tmp_path, step).name == f"step_{step:03d}"
    for step in (0, 129, True, 1.):
        with pytest.raises(ValueError):
            retention.directory(tmp_path, step)
    assert retention.expected_component_counts(0) == {}
    assert retention.expected_component_counts(1) == {"v8": 48, "v29": 52}
    assert retention.expected_component_counts(128) == {"v8": 6144, "v29": 6656}
    for count in (-1, 129, True, 1.):
        with pytest.raises(ValueError):
            retention.expected_component_counts(count)


def component_rows():
    return ([{"eligible": True, "reference_role": "v29", "component_roles": ["v29"], "components_bitwise_equal": True} for index in range(4)]
        + [{"eligible": True, "reference_role": "structured_mean", "component_roles": ["v8", "v29"], "components_bitwise_equal": True} for index in range(48)]
        + [{"eligible": False} for index in range(4)])


def test_full_component_report_checks_actual_rows_not_just_summary_counts():
    rows = component_rows()
    report = {"component_role_counts": {"v8": 48, "v29": 52}, "component_grids": 100}
    retention.require_report_components(report, rows, {"admitted_steps": [1]})
    for changed in (rows[:-5] + rows[-4:], [{**rows[0], "components_bitwise_equal": False}] + rows[1:],
            rows[:-1] + [{"eligible": False, "component_roles": ["v29"]}]):
        with pytest.raises(ValueError):
            retention.require_report_components(report, changed, {"admitted_steps": [1]})
    with pytest.raises(ValueError):
        retention.require_report_components({"component_role_counts": {"v8": 24, "v29": 26}, "component_grids": 50}, rows,
            {"admitted_steps": [1]})


def test_live_callback_namespaces_bind_four_draw_objective_and_components():
    assert retention.retention is storage.budget.retention
    assert retention.cohort_retention is storage.budget.retention
    assert retention.require_report_components.__globals__ is vars(retention)
    for function in (retention.require_report_components, retention.expected_component_counts, retention.directory,
            blend.tasks_for, blend.storage_accounting, blend.verify_records):
        assert not unresolved_globals(function), function.__name__
    for source, fragments in ((retention.RETENTION_TRAINING_SOURCE, ('len(manifest["records"]) != 56', 'len(batch["tasks"]) != 56')),
            (blend.SOURCE, ('blend_tasks != 52 * len(counts["admitted_steps"])', 'counts["training_tasks"] != 56 * len(steps)'))):
        assert all(fragment in source for fragment in fragments)


def test_no_inherited_numeric_two_draw_or_long_budget_assumptions_remain():
    for source in (retention.RETENTION_TRAINING_SOURCE, blend.SOURCE):
        constants = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and type(node.value) is int}
        assert constants.isdisjoint({26, 28, 50, 64, 65, 256, 257})


def test_final_qualification_reopens_all56_reference_decisions(schedule, monkeypatch, tmp_path):
    batch = schedule[0]
    output = tmp_path / "replay"
    output.mkdir()
    config = {"retention_storage": {"replay": str(output)}, "batches": [batch]}
    rows = []
    for task in batch["tasks"]:
        specification = task["specification"]
        metadata = {**specification.get("structured", specification), "domain": specification["domain"],
            "structured": task["cohort"] not in {"legacy", "recsys"}, "family": task["cohort"], "gradient_data": "synthetic"}
        role = retention.retention.reference_role(metadata)
        row = {"task": str(tmp_path / "training_batches/step_001" / f'task_{task["index"]:02d}.pt'),
            "metadata": metadata, "reference_role": role, "eligible": role is not None,
            "full_episode_and_identity_checked": True, "weighting": retention.retention.retention_weighting(metadata),
            "gradient_scale": 1., "task_weight": task["weight"]}
        if role is not None:
            row.update(component_roles=list(retention.retention.reference_weights(role)), components_bitwise_equal=True)
        rows.append(row)
    report = {"complete": True, "policy": retention.POLICY, "specification": retention.retention.SPECIFICATION,
        "training_config_sha256": "digest", "reference": {}, "eligibility_decisions": 56, "admitted_batches": 1,
        "reference_grids": 52, "reference_role_counts": {"v29": 4, "structured_mean": 48},
        "training_performed": False, "historical_full_autograd_replayed": False, "no_inference_ensemble": True,
        "artifact_sha256": {}, "component_role_counts": {"v8": 48, "v29": 52}, "component_grids": 100}
    rows_path = output / "tasks.jsonl"
    rows_path.write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(retention, "identity", lambda registered: {})
    monkeypatch.setattr(retention, "sha256", lambda path: "digest")
    monkeypatch.setattr(retention, "verify_hashes", lambda hashes: None)
    runtime = SimpleNamespace(OUTPUT=tmp_path, read=lambda path: config if path.name == "config.json" else report)
    audit = {"admission_attempts": {"admitted_steps": [1], "training_tasks": 56},
        "artifact_sha256": {str(output / name): "digest" for name in ("summary.json", "tasks.jsonl")}}
    assert retention.require_report(runtime, audit) == report
    rows_path.write_text("\n".join(json.dumps(row) for row in rows[:28]))
    with pytest.raises(ValueError):
        retention.require_report(runtime, audit)
