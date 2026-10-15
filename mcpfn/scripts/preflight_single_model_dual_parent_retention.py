"""One fresh fixed-mean retention probe after V49's audited negative closure."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import diagnose_single_model_parent_structured_initializer as diagnosis
import preflight_single_model_parent_structured_retention as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_dual_parent_retention as cohort_retention
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
DIAGNOSTIC_HASHES = {
    "config.json": "88f477e5abc634453a204023a100062fc33eb273c8123d268158c7237a0319cf",
    "initializer.json": "a1368dd4fc6b3acb82055520d1cf4dff953b689033ddac6fe00f58c33470e315",
    "audit.json": "448911e78c33a3e5d9674fda6f3c207c9c5dcd8d7d22f3571ca9e3574c9f0fda",
    "RESULTS.md": "4f14cb81a1fe98cac8d985d5fb88de61ac999e563e5bb02f0236ce9d339fb70f",
}
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility update after V49's complete audited "
    "negative closure and284-grid actual V29 initializer attribution. Start from unchanged "
    "actual V29, never V49 or a probe endpoint. " + cohort_retention.POLICY
    + " This registration authorizes only one feasibility update and independent audits. "
    "Preserve77,998,496 trainable parameters/238 tensors,16 frozen buffers,84 raw gradients, "
    "30 grouped protections,fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm cap and "
    "exact rollback. Independently replay52 original imputation-teacher component grids and "
    "all50 retention components (two general,24 V8,24 V29), plus all26 final retention grids. "
    "Keep24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB "
    "probe budgets. No seed retry, full trial, probe warm start, new confirmation, consumed-bank "
    "reselection, changed evaluation or benchmark-success claim. All original22 development "
    "guards,32 fresh confirmation comparisons and frozen real reviews remain mandatory."
)


def require_completed_v49():
    diagnosis.verify()
    evidence = {str(diagnosis.OUTPUT / name): digest for name, digest in DIAGNOSTIC_HASHES.items()}
    verify_hashes(evidence)
    report = read(diagnosis.OUTPUT / "audit.json")
    state = read(diagnosis.OUTPUT / "status.json")
    if (state["state"] != "complete" or diagnosis.original.process_identity(state["pid"]) is not None
            or report["complete"] is not True or report["prediction_grids"] != 284
            or len(report["comparisons"]) != 32 or report["post_hoc_consumed_bank"] is not True
            or report["goal_success_not_assessed"] is not True
            or any(report[key] is not False for key in ("cuda_initialized", "training_performed",
                "confirmation_repeated", "checkpoint_selection_performed", "real_benchmarks_evaluated"))):
        raise ValueError("Require complete stopped V49 attribution without repurposing consumed confirmation")
    expected = {("V8", family, "common_nrmse") for family in ("mar", "mnar_skip_logic")}
    for field in ("initializer_failed", "candidate_failed"):
        actual = {(row["reference"], row["group"], row["metric"])
                  for row in report["comparisons"] if row[field]}
        if actual != expected:
            raise ValueError("Preserve both inherited failures and every passing comparison in V49")
    verify_hashes(report["artifact_sha256"])
    evidence.update(report["artifact_sha256"])
    evidence.update({str(diagnosis.OUTPUT / name): sha256(diagnosis.OUTPUT / name)
                     for name in ("registration.json", "status.json")})
    evidence.update({str(path): sha256(path) for path in (diagnosis.CLOSURE, diagnosis.TRAINING_AUDIT,
        diagnosis.REPLAY / "gate.json", diagnosis.REPLAY / "audit.json",
        diagnosis.original.CONFIRMATION / "gate.json", diagnosis.original.CONFIRMATION / "audit.json")})
    return evidence


def reference_identities():
    return {"general": {"checkpoint": str(GENERAL), "sha256": cohort_retention.REFERENCE_SHA256S["general"]},
        "v8": {"checkpoint": str(probe_blueprint.training.alignment.fixed.PARENT),
            "sha256": cohort_retention.REFERENCE_SHA256S["v8"]},
        "v29": {"checkpoint": str(probe_blueprint.PARENT), "sha256": cohort_retention.REFERENCE_SHA256S["v29"]}}


def validate_saved_components(episode, payload):
    role = retention.reference_role(episode.metadata)
    if payload["component_weights"] != retention.reference_weights(role):
        raise ValueError("Never fit or alter the fixed retention coefficients")
    combined = retention.combine_predictions(role, payload["components"])
    for component in payload["components"].values():
        retention.validate_target(episode, component)
    if not composite.byte_equal(combined, payload["prediction"]):
        raise ValueError("The saved retention target must be the exact declared FP32 combination")


def validate_replay_archive(archive, role, payload):
    names = retention.reference_weights(role)
    expected = {"prediction", *(f"component_{name}" for name in names)}
    if set(archive.files) != expected:
        raise ValueError("Independently replay every component, not just the final retention target")
    if not np.array_equal(archive["prediction"], payload["prediction"].numpy()):
        raise ValueError("The independently replayed combined retention target changed")
    for name in names:
        if not np.array_equal(archive[f"component_{name}"], payload["components"][name].numpy()):
            raise ValueError("The independently replayed retention component changed")


def replay_cohort_targets(output):
    replay_original_teachers(output)
    config = verify(output)
    if read(output / "result.json").get("data_admission_only"):
        return
    manifest, records = retention_records(output)
    directory = output / "retention_replay"
    directory.mkdir()
    status(output, "replaying_retention_teachers")
    model, optimizer, criterion, teachers, sampler = training.alignment.mixture.original_setup(config["base"])
    del optimizer, teachers, sampler
    identities = reference_identities()
    references = retention.make_references(model, identities)
    del model
    verified = {role: retention.verify_model(reference, identities[role]) for role, reference in references.items()}
    if verified != manifest["reference"]:
        raise ValueError("Independently load all three exact original retention checkpoints")
    rows = []
    component_counts = Counter()
    for task, record, episode, payload in records:
        role = retention.reference_role(episode.metadata)
        components = {}
        with torch.no_grad(), compute_context(torch.device("cuda")):
            for name in retention.reference_weights(role):
                components[name] = teacher_predictions(references[name], episode.source.cuda(), criterion,
                    **STREAM_OPTIONS).float().cpu()
        for name, prediction in components.items():
            if not composite.byte_equal(prediction, payload["components"][name]):
                raise ValueError("Independent retention-component inference must reproduce every full grid")
            component_counts[name] += 1
        prediction = retention.combine_predictions(role, components)
        if not composite.byte_equal(prediction, payload["prediction"]):
            raise ValueError("Independent fixed-mean retention targets must reproduce exactly")
        path = directory / f'reference_{record["seed"]}.npz'
        atomic_npz(path, prediction=prediction.numpy(),
            **{f"component_{name}": value.numpy() for name, value in components.items()})
        rows.append({"seed": record["seed"], "artifact": str(path), "sha256": sha256(path),
            "reference_role": role, "identity": retention.target_identity(role, identities)})
    if dict(component_counts) != {"general": 2, "v8": 24, "v29": 24}:
        raise ValueError("Replay all50 retention components and all26 final grids")
    if any(retention.verify_model(reference, identities[role]) != verified[role]
            for role, reference in references.items()):
        raise ValueError("All independently replayed references must remain frozen")
    write_json(directory / "manifest.json", {"complete": True, "pid": os.getpid(), "records": rows,
        "reference": verified, "input_manifest_sha256": sha256(output / "retention_targets/manifest.json"),
        "component_role_counts": dict(component_counts), "component_grids": sum(component_counts.values())})
    status(output, "awaiting_independent_audit")


DUAL_PROBE_SOURCE = probe_blueprint.PARENT_PROBE_SOURCE
for old, new in {
    "single_model_parent_structured_retention as retention": "single_model_dual_parent_retention as retention",
    "parent_structured_retention_preflight_v1": "dual_parent_retention_preflight_v1",
    "1261100901, 1262100901": "1291100901, 1292100901",
    "require_completed_v48": "require_completed_v49",
    '{"general": 2, "v29": 24}': '{"general": 2, "structured_mean": 24}',
    'identities[record["reference_role"]]': 'retention.target_identity(record["reference_role"], identities)',
    'reference_identities()[record["reference_role"]]':
        'retention.target_identity(record["reference_role"], reference_identities())',
}.items():
    if old not in DUAL_PROBE_SOURCE:
        raise ValueError(f"Missing explicit dual-parent probe adaptation: {old}")
    DUAL_PROBE_SOURCE = DUAL_PROBE_SOURCE.replace(old, new)
DUAL_PROBE_SOURCE = replace_assignment(DUAL_PROBE_SOURCE, "POLICY_TEXT", POLICY_TEXT)
DUAL_PROBE_SOURCE = replace_once(DUAL_PROBE_SOURCE,
    'Path(reference_identities()["v29"]["checkpoint"])',
    'Path(reference_identities()["v29"]["checkpoint"]), Path(reference_identities()["v8"]["checkpoint"])')
DUAL_PROBE_SOURCE = replace_once(DUAL_PROBE_SOURCE,
    '            retention.validate_target(episode, payload["prediction"])',
    '            retention.validate_target(episode, payload["prediction"])\n'
    '            validate_saved_components(episode, payload)')
DUAL_PROBE_SOURCE = replace_once(DUAL_PROBE_SOURCE,
    '            if archive.files != ["prediction"] or not np.array_equal(archive["prediction"], payload["prediction"].numpy()):\n'
    '                raise ValueError("Independent retention replay must reproduce every prediction")',
    '            validate_replay_archive(archive, record["reference_role"], payload)')
DUAL_PROBE_SOURCE = replace_once(DUAL_PROBE_SOURCE,
    '            or len(replay["records"]) != len(records)):',
    '            or len(replay["records"]) != len(records)\n'
    '            or replay["component_role_counts"] != {"general": 2, "v8": 24, "v29": 24}\n'
    '            or replay["component_grids"] != 50):')
DUAL_PROBE_SOURCE = replace_once(DUAL_PROBE_SOURCE,
    '"eligibility_decisions": 28, "reference_grids": len(records),',
    '"eligibility_decisions": 28, "reference_grids": len(records), "component_grids": 50,\n'
    '        "component_role_counts": replay["component_role_counts"],')
registration_source = "REGISTER_SOURCE = replace_once(REGISTER_SOURCE, " + repr(
    '    sources.update(Path(module.__file__).resolve()') + ", " + repr(
    '    sources.update({REPO / "tests/test_single_model_dual_parent_retention.py", '
    'REPO / "tests/test_single_model_dual_parent_retention_preflight.py"})\n'
    '    sources.update(Path(module.__file__).resolve()') + ")"
DUAL_PROBE_SOURCE = replace_once(DUAL_PROBE_SOURCE,
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_source + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
exec(compile(DUAL_PROBE_SOURCE, __file__, "exec"), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    if blend_replay.storage_accounting(result)["reserved_checkpoint_bytes"] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError("Keep complete original teacher-component replay and capacity")
    return result


main = types.FunctionType(probe_blueprint.main.__code__, globals(), "main", probe_blueprint.main.__defaults__)


if __name__ == "__main__":
    main()
