"""Preserve six closed nonselected artifacts before an all-imputation-blend trial."""

import inspect
import math
from pathlib import Path

import torch

import relocate_single_model_uniform_v8_capacity as capacity_blueprint
from preflight_single_model_uniform_v8_retention import replace_function
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = (
    "Storage only: preserve the fully audited unselected V44 step64 optimizer tail, the completed "
    "all-imputation-blend feasibility vector, and all four original nonpromotable point-alignment "
    "proposal vectors for V8/projected80 and composite-only/composite-plus-point protections. "
    "Keep every accepted/rejected diagnostic fact and all original serialized bytes/canonical paths. "
    "Keep V44's distinct selected step50, every original teacher and other selected model physically "
    "on checkpoint storage. Require stopped owners, no open readers, source hashes, complete CPU "
    "audits, copy/fsync/hash/mmap verification and atomic canonical symlinks, followed by independent "
    "rechecking of all six files. Restore18 GiB plus2432 MiB full-training admission; keep18/6/50 GiB "
    "reserves. UUID-bound scratch is machine-local, not external or provider-persistent backup. "
    "No training, selection, new scientific seed, qualification retry or inference change.")


def require_stopped():
    for name, expected in {
        "uniform_v8_retention_v44": "training_complete",
        "uniform_v8_retention_v44_pipeline": "complete",
        "uniform_v8_retention_v44_endpoint_replay": "complete",
        "uniform_v8_retention_v44_confirmation": "complete",
        "uniform_v8_retention_v44_benchmarks": "complete",
        "uniform_v8_retention_v44_review": "complete",
        "all_imputation_blend_preflight_v1": "complete",
        "point_alignment_v1": "complete",
    }.items():
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or probe.diagnosis.original.process_identity(current["pid"]) is not None:
            raise ValueError("Require every original producer stopped with its unchanged complete outcome")
    if probe.diagnosis.original.process_identity(3609882) is not None:
        raise ValueError("Require the original feasibility chain to stop before preservation")


def validate_payload(payload, kind, *, full_parameters=77998496):
    if kind in {"vector", "point_proposal"}:
        value = payload if kind == "vector" else payload.get("parameters")
        if (not isinstance(value, torch.Tensor) or value.shape != (full_parameters,)
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Preserve every finite FP32 coordinate of the full nonpromotable vector")
        if kind == "point_proposal" and (
                set(payload) != {"margin", "accepted", "parameters", "solution", "actual_products", "actual_norm", "adverse_tasks"}
                or type(payload["accepted"]) is not bool or not isinstance(payload["solution"], dict)
                or payload["margin"] != .0001 or len(payload["actual_products"]) not in {14, 42}
                or not all(math.isfinite(value) for value in payload["actual_products"])
                or not math.isfinite(payload["actual_norm"]) or payload["actual_norm"] < 0
                or type(payload["adverse_tasks"]) is not int or payload["adverse_tasks"] < 0):
            raise ValueError("Preserve original point-proposal metadata without promoting the diagnostic")
    elif kind == "tail":
        current = read(ROOT / "uniform_v8_retention_v44/status.json")
        if (payload["step"] != 64 or payload["attempted_batches"] != 64
                or payload["gradient_updates_performed"] != 57 or payload["best_step"] != 50
                or payload["best_score"] != current["best"]["score"]
                or payload["config_sha256"] != sha256(ROOT / "uniform_v8_retention_v44/config.json")
                or payload["independent_confirmation_passed"] is not False or payload["preflight_only"] is not False
                or payload["full_network_training"] is not True or not payload["model"] or not payload["optimizer"]):
            raise ValueError("Preserve only the exact unselected V44 terminal optimizer state")
        if (SELECTED.is_symlink() or SELECTED.stat().st_dev != ROOT.stat().st_dev or sha256(SELECTED) != SELECTED_SHA256):
            raise ValueError("Never move or change the original selected checkpoint")
        selected = torch.load(SELECTED, map_location="cpu", weights_only=True, mmap=True)
        if (selected["step"] != 50 or "optimizer" in selected or set(selected["model"]) != set(payload["model"])
                or all(torch.equal(value, selected["model"][name]) for name, value in payload["model"].items())):
            raise ValueError("Preserve the distinct original selection, not a promoted tail")
    else:
        raise ValueError("Only the three prospectively declared nonselected artifact kinds may move")


SOURCE = inspect.getsource(capacity_blueprint)
SOURCE = replace_once(SOURCE, "Preserve three closed nonselected artifacts before a uniform-reference full trial.",
    "Preserve six closed nonselected artifacts before an all-imputation-blend full trial.")
SOURCE = replace_once(SOURCE, "import preflight_single_model_uniform_v8_retention as probe",
    "import preflight_single_model_all_imputation_blend_teacher as probe")
SOURCE = SOURCE.replace("uniform_v8_capacity", "all_blend_capacity")
SOURCE = SOURCE.replace("family_balanced_retention_v43", "uniform_v8_retention_v44")
SOURCE = replace_assignment(SOURCE, "SELECTED_SHA256", "4653589ca38296ea47af94004353a4eebe8a0bb660573e14240cebc12fb4bed0")
SOURCE = replace_assignment(SOURCE, "PROBE_CONFIG_SHA256", "18ef8a6008c355889d6b622d4c2e04010292e239a2127d20772016bf95a39c0f")
SOURCE = replace_assignment(SOURCE, "POLICY", POLICY_TEXT)
SOURCE = replace_assignment(SOURCE, "TARGETS", {
    "uniform_v8_retention_v44/latest_checkpoint.pth": "tail",
    "all_imputation_blend_preflight_v1/candidate_parameters.pt": "vector",
    **{f"point_alignment_v1/{state}/{method}_attempt_0.pt": "point_proposal"
        for state in ("v8", "projected80") for method in ("composite_only", "composite_and_points")},
})
SOURCE = replace_function(SOURCE, "require_stopped", inspect.getsource(require_stopped))
SOURCE = replace_function(SOURCE, "validate_payload", inspect.getsource(validate_payload))
SOURCE = SOURCE.replace('rate_audit = read(ROOT / "decoder_rate_probe_v1_audit_recovery1/audit.json")',
    'point_audit = read(ROOT / "point_alignment_v1/audit.json")\n'
    '    point_result = read(ROOT / "point_alignment_v1/result.json")')
SOURCE = replace_once(SOURCE,
    'tail_audit["best"]["step"] != 25\n'
    '            or not rate_audit["complete"] or rate_audit["checkpoint_selected"] is not False\n'
    '            or rate_audit["prediction_grids"] != 112 or rate_audit["training_tasks"] != 14',
    'tail_audit["best"]["step"] != 50\n'
    '            or not point_audit["complete"] or point_audit["cuda_initialized"] is not False\n'
    '            or point_result["complete"] is not True or point_result["synthetic_tasks"] != 14\n'
    '            or any(point_result[key] is not False for key in ("training_restarted", "trained_checkpoint_created",\n'
    '                "validation_or_confirmation_used", "real_benchmarks_used", "finite_step_accuracy_assessed", "goal_achieved"))')
SOURCE = replace_once(SOURCE, '**rate_audit["artifact_sha256"]', '**point_audit["input_sha256"]')
SOURCE = replace_once(SOURCE, '    paths = [ROOT / name for name in (',
    '    point_config = read(ROOT / "point_alignment_v1/config.json")\n'
    '    if read(ROOT / "point_alignment_v1/registration.json")["config_sha256"] != sha256(ROOT / "point_alignment_v1/config.json"):\n'
    '        raise ValueError("Preserve the original registered point-alignment diagnosis")\n'
    '    verify_hashes(point_audit["input_sha256"])\n'
    '    verify_hashes(point_config["source_sha256"])\n'
    '    verify_hashes(point_config["input_sha256"])\n'
    '    paths = [ROOT / name for name in (')
SOURCE = replace_once(SOURCE,
    '"uniform_v8_retention_v44_audit/audit.json", "uniform_v8_retention_v44_orphan_recovery_v1/result.json",\n'
    '        "decoder_rate_probe_v1/config.json", "decoder_rate_probe_v1/status.json",\n'
    '        "decoder_rate_probe_v1_audit_recovery1/audit.json", "decoder_rate_probe_v1_audit_recovery1/status.json"',
    '"uniform_v8_retention_v44_audit/audit.json", "uniform_v8_retention_v44_pipeline/result.json",\n'
    '        "point_alignment_v1/config.json", "point_alignment_v1/status.json",\n'
    '        "point_alignment_v1/audit.json", "point_alignment_v1/result.json",\n'
    '        "point_alignment_v1/v8/result.json", "point_alignment_v1/projected80/result.json"')
SOURCE = replace_once(SOURCE, '"canonical_artifacts_checked": 3', '"canonical_artifacts_checked": 6')
SOURCE = replace_once(SOURCE, 'exact three audited single-link nonselected artifacts', 'exact six audited single-link nonselected artifacts')
SOURCE = replace_once(SOURCE, 'Require the exact accepted uniform-reference probe', 'Require the exact accepted all-imputation-blend probe')
SOURCE = replace_once(SOURCE, 'or not gate["passed"] or not gate["preflight_only"]',
    'or not gate["passed"] or not gate["preflight_only"]\n'
    '            or probe_audit["teacher_replay"]["blend_tasks"] != 26\n'
    '            or probe_audit["teacher_replay"]["component_grids"] != 52')
exec(compile(SOURCE, __file__, "exec"), globals())
