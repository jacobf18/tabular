"""One fresh all-imputation column-mean probe after V51's negative closure."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import preflight_single_model_native_parent_retention as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_all_imputation_column_mean as cohort_retention
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
diagnosis = probe_blueprint.diagnosis
CLOSURE_HASHES = {
    "native_parent_v51_audit/audit.json": "177506f3fd472753a6da4a4c4c6707d227f55302d4d59a6b176147a4bd0c81ec",
    "native_parent_v51_endpoint_replay/audit.json": "b063ceb68c8ecf5990b322ed5fdf190507a2933be1aa5b7ea45e283ef79803f8",
    "native_parent_v51_pipeline/result.json": "f9093fcb55c6b6039ca76c220c7730dcbbc4c7425bb287f7682d41489654f304",
    "native_parent_v51_confirmation/gate.json": "3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94",
    "native_parent_v51_terminal_review_v1/closure.json": "7fa21ba50e1e2dd1a7e03fc36da9e64b6a4b10be4dff8fa0ec13aae44ded63b4",
    "native_parent_v51_terminal_review_v1/OUTCOME.md": "02d5166c28b0ab8616293d8592ab6cf1ddec9ad9f72c0bcba706e8246e65477d",
    "all_imputation_column_mean_endpoint_analysis_v1/receipt.json": "eb2430fb7cd93a041882de49654b25ab718d19756589d911f2ad46c6f9549120",
    "all_imputation_column_mean_endpoint_analysis_v1/OUTCOME.md": "39c276926bca7814c31959415a06431cc396b7b960f14e9152771e848fe3b0e9",
}
CLOSURE_DOCUMENTS = {
    "training": ROOT / "native_parent_v51_audit/audit.json",
    "endpoint": ROOT / "native_parent_v51_endpoint_replay/audit.json",
    "pipeline": ROOT / "native_parent_v51_pipeline/result.json",
    "gate": ROOT / "native_parent_v51_confirmation/gate.json",
    "external": ROOT / "native_parent_v51_terminal_review_v1/closure.json",
    "retention": Path("/mnt/matrixformer_scratch/native_parent_v51/anchor_retention_replay/summary.json"),
    "blend": Path("/mnt/matrixformer_scratch/native_parent_v51/all_imputation_blend_teacher_components/summary.json"),
}
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility update after V51's complete audited "
    "negative no-selection closure. Initialize only from unchanged actual V29, never V51 or "
    "a probe endpoint. " + cohort_retention.POLICY
    + " This registration authorizes one feasibility update and its independent audits only. "
    "Preserve77,998,496 parameters/238 trainable tensors,16 frozen buffers,84 raw gradients, "
    "30 grouped protections,fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm cap and "
    "exact rollback. Replay52 original imputation-teacher components, all50 retention components "
    "(24 V8 and26 V29) and26 complete targets independently. Independently reconstruct the "
    "teacher-mean terms with column weighting on all26 imputation tasks, not only native tasks. "
    "General remains verified but supplies no retention target. Preserve24 BLAS/4 Torch threads, "
    ".9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB probe budgets. No retry, "
    "mask repair, full-trial authorization, probe warm start, confirmation, consumed-bank selection, "
    "changed evaluation or benchmark-success claim. Keep all original22 development guards, "
    "32 fresh confirmation comparisons and every frozen real suite/full review for a later trial."
)


def validate_closed_v51(reports):
    training_report, endpoint, pipeline, gate = (reports[name] for name in ("training", "endpoint", "pipeline", "gate"))
    external, retention_report, blend = (reports[name] for name in ("external", "retention", "blend"))
    if (training_report["complete"] is not True or training_report["complete_training_audit"] is not True
            or training_report["through_step"] != 64 or training_report["accepted_updates"] != 58
            or training_report["best"]["step"] != 0 or training_report["independently_regenerated_training_tasks"] != 1792
            or training_report["independently_regenerated_rating_contexts"] != 128
            or training_report["independently_regenerated_structured_draws"] != 1536
            or training_report["admission_attempts"]["data_rejected_steps"] != []
            or training_report["autograd_replayed"] is not False
            or training_report["full_optimizer_trajectory_replayed"] is not False
            or training_report["confirmation_evaluated"] is not False
            or endpoint["complete"] is not True or endpoint["endpoint_grids"] != 56
            or endpoint["selected_grids"] != 0 or endpoint["cuda_initialized"] is not False
            or retention_report["complete"] is not True or retention_report["admitted_batches"] != 64
            or retention_report["reference_grids"] != 1664 or retention_report["component_grids"] != 3200
            or retention_report["component_role_counts"] != {"v8": 1536, "v29": 1664}
            or blend["complete"] is not True or blend["training_tasks_checked"] != 1792
            or blend["blend_tasks"] != 1664 or blend["raw_component_grids"] != 3328
            or blend["query_truth_used_for_routing"] is not False):
        raise ValueError("Preserve every original V51 task, component, outcome and audit limitation")
    if (pipeline["complete"] is not True or pipeline["gate"] != gate
            or pipeline["campaign"] is not None or pipeline["review"] is not None
            or gate["passed"] is not False or gate["confirmation_evaluated"] is not False
            or gate["selected_step"] != 0 or gate["real_benchmark_success"] is not False
            or external["complete"] is not True or external["outcome"] != "negative_no_eligible_checkpoint"
            or external["training_attempts"] != 64 or external["accepted_updates"] != 58
            or external["rejected_attempts"] != [32, 35, 42, 44, 55, 59]
            or external["original_workers_terminal"] is not True
            or any(external[name] is not False for name in ("goal_achieved", "confirmation_evaluated",
                "candidate_real_benchmarks_run", "model_inference_performed_by_closure", "gate_changed"))):
        raise ValueError("Require V51's exact stopped negative closure without confirmation or real data")


def require_completed_v51():
    evidence = {str(ROOT / name): digest for name, digest in CLOSURE_HASHES.items()}
    verify_hashes(evidence)
    reports = {name: read(path) for name, path in CLOSURE_DOCUMENTS.items()}
    validate_closed_v51(reports)
    for name in ("native_parent_v51", "native_parent_v51_audit_task_replay", "native_parent_v51_endpoint_replay",
            "native_parent_v51_confirmation", "native_parent_v51_pipeline"):
        path = ROOT / name / "status.json"
        state = read(path)
        if (state["state"] not in {"complete", "training_complete"}
                or diagnosis.original.process_identity(state["pid"]) is not None):
            raise ValueError("Stop every original V51 worker before a new feasibility experiment")
        evidence[str(path)] = sha256(path)
    if any((ROOT / "native_parent_v51" / name).exists() or (ROOT / "native_parent_v51" / name).is_symlink()
            for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth")):
        raise ValueError("The negative V51 closure must not publish a selected or qualified-best alias")
    analysis = read(ROOT / "all_imputation_column_mean_endpoint_analysis_v1/receipt.json")
    if (analysis["complete"] is not True or analysis["task_role_rows"] != 56
            or analysis["native_rating_weights_and_output_gradients_unchanged"] is not True
            or any(analysis[key] is not False for key in ("model_inference_performed", "cuda_initialized",
                "new_scientific_seeds", "training_performed", "checkpoint_selection_performed", "gate_changed", "real_data_used"))):
        raise ValueError("Preserve the complete synthetic-only objective diagnosis without an efficacy claim")
    for mapping in [report.get("artifact_sha256", {}) for report in reports.values()] + [
            reports["external"]["bound_sha256"], reports["external"]["registered_source_sha256"],
            analysis["bound_sha256"], analysis["output_sha256"]]:
        for path, digest in mapping.items():
            if path in evidence and evidence[path] != digest:
                raise ValueError("Every predecessor path must retain one immutable identity")
            evidence[path] = digest
    evidence.update({str(path): sha256(path) for path in CLOSURE_DOCUMENTS.values()})
    verify_hashes(evidence)
    return evidence


for name in ("reference_identities", "validate_saved_components", "validate_replay_archive"):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Probe helpers must bind the new all-imputation mean objective")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
RETENTION_REPLAY_SOURCE = probe_blueprint.RETENTION_REPLAY_SOURCE
exec(compile(RETENTION_REPLAY_SOURCE, __file__, "exec"), globals())

ALL_IMPUTATION_PROBE_SOURCE = probe_blueprint.NATIVE_PARENT_PROBE_SOURCE
for old, new in {
    "single_model_native_parent_retention as retention": "single_model_all_imputation_column_mean as retention",
    "native_parent_retention_preflight_v1": "all_imputation_column_mean_preflight_v1",
    "1321100901, 1322100901": "1351100901, 1352100901",
    "require_completed_v50": "require_completed_v51",
}.items():
    if old not in ALL_IMPUTATION_PROBE_SOURCE:
        raise ValueError(f"Missing explicit all-imputation mean probe adaptation: {old}")
    ALL_IMPUTATION_PROBE_SOURCE = ALL_IMPUTATION_PROBE_SOURCE.replace(old, new)
ALL_IMPUTATION_PROBE_SOURCE = replace_assignment(ALL_IMPUTATION_PROBE_SOURCE, "POLICY_TEXT", POLICY_TEXT)
registration_source = "REGISTER_SOURCE = replace_once(REGISTER_SOURCE, " + repr(
    '    sources.update(Path(module.__file__).resolve()') + ", " + repr(
    '    sources.update({REPO / "tests/test_single_model_all_imputation_column_mean.py", '
    'REPO / "tests/test_single_model_all_imputation_column_mean_preflight.py"})\n'
    '    sources.update(Path(module.__file__).resolve()') + ")"
ALL_IMPUTATION_PROBE_SOURCE = replace_once(ALL_IMPUTATION_PROBE_SOURCE,
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_source + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
exec(compile(ALL_IMPUTATION_PROBE_SOURCE, __file__, "exec"), globals())
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
