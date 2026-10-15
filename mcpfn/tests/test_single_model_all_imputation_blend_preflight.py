from copy import deepcopy
import inspect
from types import FunctionType

import pytest

import preflight_single_model_all_imputation_blend_teacher as probe
import preflight_single_model_uniform_v8_retention as previous
import test_single_model_legacy_blend_replay as replay_test_blueprint
import test_single_model_uniform_v8_retention_preflight as retention_test_blueprint
from single_model_registered_adaptation import replace_once


def test_probe_changes_only_teacher_policy_and_fresh_seeds_not_supervision_or_retention():
    current, original = probe.batch(), previous.batch()
    assert len(current["tasks"]) == len(original["tasks"]) == 28
    for actual, expected in zip(current["tasks"], original["tasks"]):
        restored = deepcopy(actual)
        assert restored["gradient_seed"] != expected["gradient_seed"]
        restored["gradient_seed"] = expected["gradient_seed"]
        specification = restored["specification"]
        specification.get("structured", specification)["seed"] = expected["specification"].get(
            "structured", expected["specification"])["seed"]
        assert restored == expected
    assert probe.reference_identities() == previous.reference_identities()
    assert probe.PARENT == previous.PARENT and probe.PARENT_SHA256 == previous.PARENT_SHA256
    assert probe.ARITHMETIC == {"blas_threads": 24, "torch_threads": 4}
    assert probe.teacher_policy.POLICY == {"legacy": "observed_blend", "structured": "observed_blend", "recsys": "specialist"}
    assert previous.teacher_policy.POLICY["structured"] == "general"
    assert probe.DATA_SEED == 1141100901 and probe.GRADIENT_SEED == 1142100901
    for name in ("run", "verify", "audit", "register_trial", "replay_cohort_targets"):
        function = inspect.unwrap(getattr(probe, name))
        assert function.__globals__ is vars(probe) and function.__closure__ is None
    assert "require_completed_v44" in probe.require_predecessor.__code__.co_names
    assert probe.BLEND_COMPONENT_ALLOWANCE == 16329216
    before, after = deepcopy(previous.STORAGE_ACCOUNTING), deepcopy(probe.STORAGE_ACCOUNTING)
    assert after.pop("checkpoint_artifact_allowance_bytes") == 529649792 < 512 * 1024**2
    before.pop("checkpoint_artifact_allowance_bytes")
    assert before == after


reference_namespace = {**vars(retention_test_blueprint), "probe": probe}
for name in ("retention_fixture", "test_all_reference_and_weighting_decisions_are_bound",
        "test_substituting_v29_on_any_original_imputation_task_is_rejected",
        "test_independent_cpu_audit_preserves_all_26_v8_references_and_both_rating_exclusions"):
    blueprint = getattr(retention_test_blueprint, name)
    cloned = FunctionType(blueprint.__code__, reference_namespace, name, blueprint.__defaults__, blueprint.__closure__)
    cloned.__dict__.update(blueprint.__dict__)
    reference_namespace[name] = cloned
    if name.startswith("test_"):
        globals()[name] = cloned


REPLAY_TEST_SOURCE = inspect.getsource(replay_test_blueprint)
REPLAY_TEST_SOURCE = replace_once(REPLAY_TEST_SOURCE,
    "import preflight_single_model_legacy_blend_teacher as probe",
    "import preflight_single_model_all_imputation_blend_teacher as probe")
REPLAY_TEST_SOURCE = replace_once(REPLAY_TEST_SOURCE,
    "import single_model_legacy_blend_replay as replay",
    "import single_model_all_imputation_blend_replay as replay")
REPLAY_TEST_SOURCE = replace_once(REPLAY_TEST_SOURCE, 'result["blend_tasks"] == 2 and result["component_grids"] == 4',
    'result["blend_tasks"] == 26 and result["component_grids"] == 52')
REPLAY_TEST_SOURCE = replace_once(REPLAY_TEST_SOURCE, 'calls.count("specialist") == 4', 'calls.count("specialist") == 28')
REPLAY_TEST_SOURCE = REPLAY_TEST_SOURCE[:REPLAY_TEST_SOURCE.index('@pytest.mark.parametrize("mutation", ["missing_task", "large_shape"')]
exec(compile(REPLAY_TEST_SOURCE, __file__, "exec"), globals())


def completed_reports():
    training, replay, gate, audit, closure, diagnostic = retention_test_blueprint.completed_reports()
    training["accepted_updates"] = 57
    training["best"]["step"] = 50
    training["admission_attempts"] = {"data_rejected_steps": [], "optimizer_attempts": 64}
    gate.update(passed=True, selected_step=50, checkpoint_sha256=probe.SELECTED_SHA256)
    closure.update(gate=deepcopy(gate), campaign=str(probe.ROOT / "uniform_v8_retention_v44_benchmarks"),
        review=str(probe.ROOT / "uniform_v8_retention_v44_review"))
    campaign = {"all_evaluations_complete": True, "audits": {name: {} for name in ("tabular", "recsys", "llm")},
        "candidate": {"checkpoint_sha256": probe.SELECTED_SHA256}}
    review = {"complete": True, "strict_point_estimate_parity": False, "all_primary_point_estimates_retained": False,
        "required_comparisons": 392, "primary_comparisons": 18, "primary_regressions": 14,
        "required_regressions": 241, "comparison_tolerance": 0, "practical_equivalence_margin": None,
        "cuda_initialized": False, "all_raw_predictions_reaudited": True,
        "original_campaign_hashes_unchanged": True, "goal_success_not_assessed": True}
    return training, replay, gate, audit, closure, campaign, review


@pytest.mark.parametrize("entry,key,value", [
    (0, "accepted_updates", 56), (0, "prediction_grids", 1216), (0, "autograd_replayed", True),
    (0, "admission_attempts", {"data_rejected_steps": [40], "optimizer_attempts": 63}),
    (1, "selected_grids", 0), (2, "passed", False), (2, "checkpoint_sha256", "wrong"), (2, "cases", 283),
    (3, "prediction_grids", 1135), (3, "cuda_initialized", True), (4, "campaign", None), (4, "review", None),
    (5, "all_evaluations_complete", False), (5, "audits", {"tabular": {}, "recsys": {}}),
    (6, "complete", False), (6, "strict_point_estimate_parity", True), (6, "required_comparisons", 391),
    (6, "primary_comparisons", 17), (6, "primary_regressions", 13), (6, "required_regressions", 240),
    (6, "comparison_tolerance", .01),
    (6, "practical_equivalence_margin", .01), (6, "all_raw_predictions_reaudited", False),
    (6, "original_campaign_hashes_unchanged", False), (6, "goal_success_not_assessed", False)])
def test_incomplete_or_reinterpreted_v44_evidence_is_rejected(entry, key, value):
    reports = completed_reports()
    probe.validate_completed(*reports)
    reports[entry][key] = value
    with pytest.raises(ValueError):
        probe.validate_completed(*reports)


def test_missing_closed_report_binding_cannot_authorize_a_probe(monkeypatch):
    monkeypatch.setattr(probe, "CLOSED_REPORTS", None)
    with pytest.raises(ValueError, match="eight complete immutable"):
        probe.require_completed_v44()


def test_complete_binding_reads_review_sources_from_the_hashed_config(monkeypatch):
    reports = completed_reports()
    for report in (reports[0], reports[1], reports[3]):
        report["artifact_sha256"] = {}
    review = reports[-1]
    review.update(comparisons_sha256="comparisons", config_sha256="config",
        candidate=deepcopy(reports[-2]["candidate"]),
        model_inventory={"tensors": 254, "elements": 77998688, "changed_tensors": 238})
    reports[-2]["audits"] = {name: {"path": f"/test/{name}/audit.json", "sha256": name}
        for name in ("tabular", "recsys", "llm")}
    records = dict(zip(probe.CLOSED_REPORTS, [{"source_sha256": {}}, *reports]))
    records["uniform_v8_retention_v44_review/registration.json"] = {"config_sha256": "config", "registered_unix": 1}
    records["uniform_v8_retention_v44_review/config.json"] = {
        "candidate": deepcopy(review["candidate"]), "source_sha256": {"review-source": "source"},
        "input_sha256": {"review-input": "input"}, "training_performed": False,
        "checkpoint_selection_performed": False, "inference_performed": False, "original_artifacts_modified": False}
    for name in ("uniform_v8_retention_v44", "uniform_v8_retention_v44_endpoint_replay",
            "uniform_v8_retention_v44_confirmation", "uniform_v8_retention_v44_pipeline",
            "uniform_v8_retention_v44_benchmarks", "uniform_v8_retention_v44_review"):
        records[f"{name}/status.json"] = {"state": "complete", "pid": -101}
    checked = []
    monkeypatch.setattr(probe, "read", lambda path: records[str(path.relative_to(probe.ROOT))])
    monkeypatch.setattr(probe, "sha256", lambda path: "status")
    monkeypatch.setattr(probe, "verify_hashes", lambda hashes: checked.append(dict(hashes)))
    monkeypatch.setattr(probe.diagnosis.original, "process_identity", lambda pid: None)
    result = probe.require_completed_v44()
    assert {"review-source": "source"} in checked and {"review-input": "input"} in checked
    assert result[str(probe.ROOT / "uniform_v8_retention_v44_review/comparisons.csv")] == "comparisons"
    assert len(result) == 19
