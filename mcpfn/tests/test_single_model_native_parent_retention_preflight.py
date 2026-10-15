import ast
from pathlib import Path


TEST_SOURCE = Path(__file__).with_name("test_single_model_dual_parent_retention_preflight.py").read_text()
removed = {"complete_diagnosis", "test_new_probe_requires_exact_stopped_negative_attribution"}
nodes = [node for node in ast.parse(TEST_SOURCE).body if isinstance(node, ast.FunctionDef) and node.name in removed]
lines = TEST_SOURCE.splitlines(keepends=True)
for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
    del lines[start:node.end_lineno]
TEST_SOURCE = "".join(lines)
for old, new in {
    "preflight_single_model_dual_parent_retention as probe": "preflight_single_model_native_parent_retention as probe",
    "preflight_single_model_parent_structured_retention as predecessor": "preflight_single_model_dual_parent_retention as predecessor",
    "1291100901, 1292100901": "1321100901, 1322100901",
    "dual_parent_retention_preflight_v1": "native_parent_retention_preflight_v1",
    '("dual_parent_retention", "dual_parent_retention_preflight")': '("native_parent_retention", "native_parent_retention_preflight")',
    "require_completed_v49": "require_completed_v50",
    '{"general": 2, "v8": 24, "v29": 24}': '{"v8": 24, "v29": 26}',
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent preflight test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def closed_reports():
    gate = {"passed": False, "confirmation_evaluated": False, "selected_step": 0, "real_benchmark_success": False}
    return {
        "training": {"complete": True, "complete_training_audit": True, "through_step": 64,
            "accepted_updates": 56, "best": {"step": 0}, "independently_regenerated_training_tasks": 1764,
            "independently_regenerated_rating_contexts": 126, "independently_regenerated_structured_draws": 1536,
            "admission_attempts": {"data_rejected_steps": [24]}, "autograd_replayed": False,
            "full_optimizer_trajectory_replayed": False, "confirmation_evaluated": False},
        "endpoint": {"complete": True, "endpoint_grids": 56, "selected_grids": 0, "cuda_initialized": False},
        "pipeline": {"complete": True, "gate": gate, "campaign": None, "review": None},
        "gate": gate,
        "external": {"complete": True, "negative_no_selection_closure": True,
            "failed_guards_by_trained_boundary": {"10": 3, "25": 4, "50": 3, "64": 4},
            "goal_achieved": False, "fresh_confirmation_evaluated": False, "real_benchmarks_evaluated": False,
            "checkpoint_qualified": False, "new_training_or_inference_performed": False},
        "retention": {"complete": True, "admitted_batches": 63, "reference_grids": 1638,
            "component_grids": 3150, "component_role_counts": {"general": 126, "v8": 1512, "v29": 1512}},
        "blend": {"complete": True, "training_tasks_checked": 1764, "blend_tasks": 1638,
            "raw_component_grids": 3276, "query_truth_used_for_routing": False},
    }


@pytest.mark.parametrize("change", [None, ("training", "accepted_updates", 57),
    ("training", "independently_regenerated_training_tasks", 1792), ("training", "autograd_replayed", True),
    ("training", "confirmation_evaluated", True), ("endpoint", "endpoint_grids", 55),
    ("endpoint", "selected_grids", 152), ("endpoint", "cuda_initialized", True),
    ("pipeline", "complete", False), ("pipeline", "campaign", "/real"), ("pipeline", "review", {}),
    ("gate", "passed", True), ("gate", "confirmation_evaluated", True), ("gate", "selected_step", 64),
    ("external", "goal_achieved", True), ("external", "checkpoint_qualified", True),
    ("external", "failed_guards_by_trained_boundary", {"10": 3, "25": 4, "50": 3, "64": 0}),
    ("retention", "component_grids", 1638), ("retention", "reference_grids", 1664),
    ("retention", "component_role_counts", {"v8": 1512, "v29": 1638}),
    ("blend", "raw_component_grids", 3275), ("blend", "query_truth_used_for_routing", True)])
def test_predecessor_requires_complete_original_negative_closure(change):
    reports = closed_reports()
    if change is None:
        probe.validate_closed_v50(reports)
    else:
        report, key, value = change
        reports[report][key] = value
        with pytest.raises(ValueError):
            probe.validate_closed_v50(reports)


def test_new_native_component_identity_is_checked_without_general_substitution(tmp_path):
    episode = originals.episode()
    payload = payload_for(episode)
    assert set(payload["components"]) == {"v29"}
    probe.validate_saved_components(episode, payload)
    path = tmp_path / "native.npz"
    np.savez(path, prediction=payload["prediction"].numpy(), component_v29=payload["components"]["v29"].numpy())
    with np.load(path, allow_pickle=False) as archive:
        probe.validate_replay_archive(archive, "v29", payload)
    payload["components"] = {"general": payload["components"]["v29"]}
    with pytest.raises(ValueError):
        probe.validate_saved_components(episode, payload)
