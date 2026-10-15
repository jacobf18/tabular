from copy import deepcopy
import inspect

import pytest

import summarize_single_model_general_native_outcome as report
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def closed_documents():
    gate = {"passed": False, "selected_step": 0, "confirmation_evaluated": False, "real_benchmark_success": False}
    return {"training_audit": {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 58, "training_episodes": 1792, "prediction_grids": 1368,
        "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
        "independently_regenerated_structured_draws": 1536, "best": {"step": 0}, "confirmation_evaluated": False,
        "research_selection_tolerance": 0, "full_network_training": True, "fixed_half_training": True,
        "displacement_fraction": .5, "autograd_replayed": False, "full_optimizer_trajectory_replayed": False},
        "retention": {"complete": True, "eligibility_decisions": 1792, "reference_grids": 1664,
                      "reference_role_counts": {"general": 128, "v8": 1536}},
        "components": {"complete": True, "blend_tasks": 1664, "raw_component_grids": 3328,
                       "observed_weights_refitted_on_cpu": True},
        "endpoint_audit": {"complete": True, "endpoint_grids": 56, "selected_grids": 0,
                           "cuda_initialized": False, "autograd_replayed": False},
        "gate": gate, "result": {"complete": True, "gate": deepcopy(gate), "campaign": None, "review": None,
                                 "no_inference_ensemble": True, "goal_success_not_assessed": True}}


@pytest.mark.parametrize("section,field,value", [
    ("training_audit", "through_step", 50), ("training_audit", "accepted_updates", 57),
    ("training_audit", "training_episodes", 1791), ("training_audit", "prediction_grids", 1367),
    ("training_audit", "best", {"step": 10}), ("training_audit", "confirmation_evaluated", True),
    ("training_audit", "research_selection_tolerance", 1e-6), ("training_audit", "full_optimizer_trajectory_replayed", True),
    ("retention", "reference_role_counts", {"v8": 1664}), ("retention", "reference_grids", 1663),
    ("components", "raw_component_grids", 3327), ("components", "observed_weights_refitted_on_cpu", False),
    ("endpoint_audit", "selected_grids", 152), ("endpoint_audit", "endpoint_grids", 55),
    ("gate", "passed", True), ("gate", "confirmation_evaluated", True),
    ("result", "campaign", "/benchmarks"), ("result", "review", "/review"),
    ("result", "no_inference_ensemble", False), ("result", "complete", False),
])
def test_closure_refuses_incomplete_or_relabelled_science(section, field, value):
    documents = closed_documents()
    report.validate_closed(documents)
    documents[section][field] = value
    with pytest.raises(ValueError):
        report.validate_closed(documents)


def boundaries():
    records = []
    for step in report.BOUNDARIES:
        comparisons = [{"group": f"cohort_{index}", "metric": "common_nrmse", "initialization": 1.,
                        "candidate": .9 if step == 0 or index else 1.1,
                        "retained": bool(step == 0 or index)} for index in range(22)]
        records.append({"step": step, "accepted_updates": step, "best_step": 0, "selected_checkpoint_sha256": None,
                        "decision": {"comparisons": comparisons, "research_selection_tolerance": 0,
                                     "passed": step == 0, "score": 1. if step == 0 else 1.1}})
    return records


@pytest.mark.parametrize("mutation", ["omit_boundary", "reorder_boundary", "omit_guard", "duplicate_guard",
    "changed_limit", "tolerance", "selected", "alias", "passed", "retained", "nonfinite"])
def test_all_boundaries_and_every_original_comparison_are_mandatory(mutation):
    records = boundaries()
    if mutation == "omit_boundary":
        records.pop()
    elif mutation == "reorder_boundary":
        records.reverse()
    elif mutation == "omit_guard":
        records[0]["decision"]["comparisons"].pop()
    elif mutation == "duplicate_guard":
        records[0]["decision"]["comparisons"][1] = deepcopy(records[0]["decision"]["comparisons"][0])
    elif mutation == "changed_limit":
        records[-1]["decision"]["comparisons"][0]["initialization"] = 2.
    elif mutation == "tolerance":
        records[-1]["decision"]["research_selection_tolerance"] = 1e-6
    elif mutation == "selected":
        records[-1]["best_step"] = 10
    elif mutation == "alias":
        records[-1]["selected_checkpoint_sha256"] = "digest"
    elif mutation == "passed":
        records[-1]["decision"]["passed"] = True
    elif mutation == "retained":
        records[-1]["decision"]["comparisons"][0]["retained"] = True
    else:
        records[-1]["decision"]["comparisons"][0]["candidate"] = float("nan")
    with pytest.raises(ValueError):
        report.comparison_rows(records)


def test_report_preserves_actual_initializer_and_every_adverse_delta():
    records = boundaries()
    rows = report.comparison_rows(records)
    assert len(rows) == 110
    assert all(row["actual_V29_initializer"] == .9 for row in rows)
    assert all(row["V8_limit"] == 1. for row in rows)
    assert sum(not row["retained_vs_V8"] for row in rows) == 4
    assert sum(row["delta_vs_actual_V29"] > 0 for row in rows) == 4
    text = report.render({"artifact_sha256": {}}, records, rows)
    assert "No trained boundary qualifies" in text and "not real MissBench/UCI/recommender scores" in text
    assert "not a seed-controlled causal comparison" in text
    for index in range(22):
        assert f"| cohort_{index} | common_nrmse |" in text


def test_actual_report_has_no_unresolved_globals_and_does_not_start_workers():
    for value in vars(report).values():
        if inspect.isfunction(value) and value.__globals__ is vars(report):
            assert not unresolved_globals(value)
    source = inspect.getsource(report)
    assert "subprocess" not in source and "load_state_dict" not in source
    assert report.pipeline.training.OUTPUT.name == "general_native_retention_v47"
