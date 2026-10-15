"""One fresh native-parent retention probe after V50's verified negative closure."""

import ast
from collections import Counter
import fcntl
from pathlib import Path
import types

import preflight_single_model_dual_parent_retention as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_native_parent_retention as cohort_retention
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
diagnosis = probe_blueprint.diagnosis
CLOSURE_HASHES = {
    "dual_parent_v50_audit/audit.json": "1935ed5ac8b6dfb3c6bfb176366589456f361ec7ff69b1ec2bb16ca1dbce4ec7",
    "dual_parent_v50_endpoint_replay/audit.json": "1570eeef9e8c947e4958fa1463abd3d365ef1c2ce15238d10143d2ae604ddbd1",
    "dual_parent_v50_pipeline/result.json": "b77e89952dfe4fc30e8a8a89a684442d893caa92831c9e3ba0b47f74d06df31f",
    "dual_parent_v50_confirmation/gate.json": "3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94",
    "dual_parent_v50_final_review_v1/external_recheck.json": "4a93a53a0439a48b3cc4073e1e3597f4edd685c5aabcc14decb304c911e8318e",
    "dual_parent_v50_final_review_v1/OUTCOME.md": "bb19388227686e8e4a96ac0399b59c43ffdbf283b3781fbc7d8054f7d553e675",
}
CLOSURE_DOCUMENTS = {
    "training": ROOT / "dual_parent_v50_audit/audit.json",
    "endpoint": ROOT / "dual_parent_v50_endpoint_replay/audit.json",
    "pipeline": ROOT / "dual_parent_v50_pipeline/result.json",
    "gate": ROOT / "dual_parent_v50_confirmation/gate.json",
    "external": ROOT / "dual_parent_v50_final_review_v1/external_recheck.json",
    "retention": Path("/mnt/matrixformer_scratch/dual_parent_v50/anchor_retention_replay/summary.json"),
    "blend": Path("/mnt/matrixformer_scratch/dual_parent_v50/all_imputation_blend_teacher_components/summary.json"),
}
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility update after V50's complete independent "
    "training/endpoint audits and verified no-selection closure. No V50 confirmation or real "
    "benchmark case was evaluated. Initialize only from unchanged actual V29, never V50 or a "
    "probe endpoint. " + cohort_retention.POLICY
    + " Authorize one feasibility update and independent audits only. Preserve77,998,496 "
    "parameters/238 trainable tensors,16 frozen buffers,84 raw gradients,30 grouped protections, "
    "fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm cap and exact rollback. Replay52 "
    "original imputation-teacher components, all50 retention components (24 V8 and26 V29) "
    "and26 complete targets independently. General is still a verified reference identity "
    "but performs zero retention-target inference. Preserve24 BLAS/4 Torch threads,.9 allocator "
    "cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB probe budgets. No retry, mask repair, "
    "full-trial authorization, probe warm start, confirmation, consumed-bank selection, changed "
    "evaluation or benchmark-success claim. Keep all original22 development guards,32 fresh "
    "confirmation comparisons and every frozen real suite/full review for any later trial."
)


def validate_closed_v50(reports):
    training_report, endpoint, pipeline, gate = (reports[name] for name in ("training", "endpoint", "pipeline", "gate"))
    external, retention_report, blend = (reports[name] for name in ("external", "retention", "blend"))
    if (training_report["complete"] is not True or training_report["complete_training_audit"] is not True
            or training_report["through_step"] != 64 or training_report["accepted_updates"] != 56
            or training_report["best"]["step"] != 0 or training_report["independently_regenerated_training_tasks"] != 1764
            or training_report["independently_regenerated_rating_contexts"] != 126
            or training_report["independently_regenerated_structured_draws"] != 1536
            or training_report["admission_attempts"]["data_rejected_steps"] != [24]
            or training_report["autograd_replayed"] is not False
            or training_report["full_optimizer_trajectory_replayed"] is not False
            or training_report["confirmation_evaluated"] is not False
            or endpoint["complete"] is not True or endpoint["endpoint_grids"] != 56
            or endpoint["selected_grids"] != 0 or endpoint["cuda_initialized"] is not False
            or retention_report["complete"] is not True or retention_report["admitted_batches"] != 63
            or retention_report["reference_grids"] != 1638 or retention_report["component_grids"] != 3150
            or retention_report["component_role_counts"] != {"general": 126, "v8": 1512, "v29": 1512}
            or blend["complete"] is not True or blend["training_tasks_checked"] != 1764
            or blend["blend_tasks"] != 1638 or blend["raw_component_grids"] != 3276
            or blend["query_truth_used_for_routing"] is not False):
        raise ValueError("Preserve every original V50 admitted/rejected task, component and audit limitation")
    if (pipeline["complete"] is not True or pipeline["gate"] != gate
            or pipeline["campaign"] is not None or pipeline["review"] is not None
            or gate["passed"] is not False or gate["confirmation_evaluated"] is not False
            or gate["selected_step"] != 0 or gate["real_benchmark_success"] is not False
            or external["complete"] is not True or external["negative_no_selection_closure"] is not True
            or external["failed_guards_by_trained_boundary"] != {"10": 3, "25": 4, "50": 3, "64": 4}
            or any(external[name] is not False for name in ("goal_achieved", "fresh_confirmation_evaluated",
                "real_benchmarks_evaluated", "checkpoint_qualified", "new_training_or_inference_performed"))):
        raise ValueError("Require V50's exact negative no-selection closure without confirmation or real data")


def require_completed_v50():
    evidence = {str(ROOT / name): digest for name, digest in CLOSURE_HASHES.items()}
    verify_hashes(evidence)
    reports = {name: read(path) for name, path in CLOSURE_DOCUMENTS.items()}
    validate_closed_v50(reports)
    for name in ("dual_parent_v50", "dual_parent_v50_audit_task_replay", "dual_parent_v50_endpoint_replay",
            "dual_parent_v50_confirmation", "dual_parent_v50_pipeline"):
        path = ROOT / name / "status.json"
        state = read(path)
        if (state["state"] not in {"complete", "training_complete"}
                or diagnosis.original.process_identity(state["pid"]) is not None):
            raise ValueError("Stop every original V50 worker after its registered terminal state")
        evidence[str(path)] = sha256(path)
    if any((ROOT / "dual_parent_v50" / name).exists() or (ROOT / "dual_parent_v50" / name).is_symlink()
            for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth")):
        raise ValueError("The negative V50 closure must not publish a selected or qualified-best alias")
    for report in reports.values():
        for path, digest in report.get("artifact_sha256", {}).items():
            if path in evidence and evidence[path] != digest:
                raise ValueError("Every predecessor artifact path must have one immutable identity")
            evidence[path] = digest
    evidence.update({str(path): sha256(path) for path in CLOSURE_DOCUMENTS.values()})
    verify_hashes(evidence)
    return evidence


for name in ("reference_identities", "validate_saved_components", "validate_replay_archive"):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Probe helpers must bind the new native-parent reference policy")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

blueprint_source = Path(probe_blueprint.__file__).read_text()
replay_node = next(node for node in ast.parse(blueprint_source).body
    if isinstance(node, ast.FunctionDef) and node.name == "replay_cohort_targets")
RETENTION_REPLAY_SOURCE = ast.get_source_segment(blueprint_source, replay_node)
RETENTION_REPLAY_SOURCE = replace_once(RETENTION_REPLAY_SOURCE,
    '{"general": 2, "v8": 24, "v29": 24}', '{"v8": 24, "v29": 26}')
exec(compile(RETENTION_REPLAY_SOURCE, __file__, "exec"), globals())

NATIVE_PARENT_PROBE_SOURCE = probe_blueprint.DUAL_PROBE_SOURCE
for old, new in {
    "single_model_dual_parent_retention as retention": "single_model_native_parent_retention as retention",
    "dual_parent_retention_preflight_v1": "native_parent_retention_preflight_v1",
    "1291100901, 1292100901": "1321100901, 1322100901",
    "require_completed_v49": "require_completed_v50",
    '{"general": 2, "structured_mean": 24}': '{"v29": 2, "structured_mean": 24}',
    '{"general": 2, "v8": 24, "v29": 24}': '{"v8": 24, "v29": 26}',
}.items():
    if old not in NATIVE_PARENT_PROBE_SOURCE:
        raise ValueError(f"Missing explicit native-parent probe adaptation: {old}")
    NATIVE_PARENT_PROBE_SOURCE = NATIVE_PARENT_PROBE_SOURCE.replace(old, new)
NATIVE_PARENT_PROBE_SOURCE = replace_assignment(NATIVE_PARENT_PROBE_SOURCE, "POLICY_TEXT", POLICY_TEXT)
registration_source = "REGISTER_SOURCE = replace_once(REGISTER_SOURCE, " + repr(
    '    sources.update(Path(module.__file__).resolve()') + ", " + repr(
    '    sources.update({REPO / "tests/test_single_model_native_parent_retention.py", '
    'REPO / "tests/test_single_model_native_parent_retention_preflight.py"})\n'
    '    sources.update(Path(module.__file__).resolve()') + ")"
NATIVE_PARENT_PROBE_SOURCE = replace_once(NATIVE_PARENT_PROBE_SOURCE,
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_source + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
exec(compile(NATIVE_PARENT_PROBE_SOURCE, __file__, "exec"), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    if blend_replay.storage_accounting(result)["reserved_checkpoint_bytes"] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError("Preserve original teacher-component replay and prospective storage capacity")
    return result


main = types.FunctionType(probe_blueprint.main.__code__, globals(), "main", probe_blueprint.main.__defaults__)


if __name__ == "__main__":
    main()
