"""Prospective all-imputation blend feasibility probe after complete V44 closure."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import preflight_single_model_uniform_v8_retention as previous_probe
from preflight_single_model_uniform_v8_retention import diagnosis, replace_function
import single_model_all_imputation_blend_replay as blend_replay
import single_model_all_imputation_blend_teacher as new_teacher
import single_model_uniform_v8_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once


ARITHMETIC = {"blas_threads": 24, "torch_threads": 4}
ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
CLOSED_REPORTS = {
    "uniform_v8_retention_v44/config.json": "60d8edf0c9cc94ecbfd7d97418b5190922c150361f0df31997e595991628e11b",
    "uniform_v8_retention_v44_audit/audit.json": "b1a691b6165d79390b5e30dbc639ca4cce57c813cec4003da7d83a1192990e40",
    "uniform_v8_retention_v44_endpoint_replay/audit.json": "04c7bf866c84c35f4fb9c984876853f7cfa29eb392e5ebe9a118829fe6b698aa",
    "uniform_v8_retention_v44_confirmation/gate.json": "07292e38f90c057bdc70ccbec4727d023b1af1172960106e8abe9394b86f11c4",
    "uniform_v8_retention_v44_confirmation/audit.json": "1e940683fd4b433527001e4e4f8f752a13fd920b741f57f9d2c4fe8888aea2c4",
    "uniform_v8_retention_v44_pipeline/result.json": "f312dc2019b558a8c9fcdd14f116820e7fd832d94631e6b9a63631d7b1f0b15b",
    "uniform_v8_retention_v44_benchmarks/campaign_audit.json": "72ada3c269c7c646033becea499dc7d3cce0d20aa745922fa6bdb3103bd859f3",
    "uniform_v8_retention_v44_review/result.json": "ac9219b3dcf8ccb2f35249efef4f939479b7b2f9b3043d36aacdb5eebafa9c20",
}
SELECTED_SHA256 = "4653589ca38296ea47af94004353a4eebe8a0bb660573e14240cebc12fb4bed0"
BLEND_COMPONENT_ALLOWANCE = blend_replay.storage_accounting(previous_probe.batch())["reserved_checkpoint_bytes"]
POLICY_TEXT = (
    "One separately registered fresh28-task feasibility probe only after V44's full original training, "
    "qualification, all three real suites and complete review have closed. Keep V29 step25/98 inherited "
    "updates as the unchanged initializer to isolate teacher assignment, never initialize from V44's "
    "real-benchmark results or a feasibility endpoint. Change structured-imputation training targets "
    "from general to the same observed-only blend already used on legacy tasks, uniformly over all "
    "families/components; keep rating-specialist targets. " + new_teacher.POLICY_TEXT
    + " Preserve original priors, supervised .5/.25/.25 masses, observed normalization, Huber/NLL/CDF/"
    "query/point losses, uniform V8 weight10 count-shrunk equal-column retention and every original "
    "task weight. Keep238 trainable tensors/77,998,496 parameters,16 frozen buffers,84 raw gradients, "
    "30 protections,fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm bound and exact rollback. "
    "Bind all52 independent raw teacher component grids and26 V8 references; ratings exclude retention. "
    "Preserve24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB "
    "probe budgets. Preserve all22 development and32 confirmation comparisons and all frozen real "
    "suites for any later independently registered full trial. No scientific retry, hidden-truth "
    "routing, seed reuse, relaxed gate or inference ensemble. Acceptance establishes feasibility only.")


def validate_completed(training_audit, replay, gate, confirmation_audit, closure, campaign, review):
    expected = {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 57, "training_episodes": 1792,
        "prediction_grids": 1368, "full_network_training": True, "fixed_half_training": True,
        "displacement_fraction": .5, "independently_regenerated_training_tasks": 1792,
        "independently_regenerated_rating_contexts": 128, "independently_regenerated_structured_draws": 1536,
        "full_optimizer_trajectory_replayed": False, "autograd_replayed": False}
    if (any(training_audit.get(key) != value for key, value in expected.items())
            or training_audit["best"]["step"] != 50
            or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or training_audit["admission_attempts"]["data_rejected_steps"] != []
            or training_audit["admission_attempts"]["optimizer_attempts"] != 64
            or replay["complete"] is not True or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or gate["passed"] is not True or gate["selected_step"] != 50 or gate["cases"] != 284
            or gate["checkpoint_sha256"] != SELECTED_SHA256 or gate["confirmation_evaluated"] is not True
            or confirmation_audit["complete"] is not True or confirmation_audit["prediction_grids"] != 1136
            or confirmation_audit["cuda_initialized"] is not False
            or closure["complete"] is not True or closure["gate"] != gate
            or closure["campaign"] != str(ROOT / "uniform_v8_retention_v44_benchmarks")
            or closure["review"] != str(ROOT / "uniform_v8_retention_v44_review")
            or campaign["all_evaluations_complete"] is not True or set(campaign["audits"]) != {"tabular", "recsys", "llm"}
            or campaign["candidate"]["checkpoint_sha256"] != SELECTED_SHA256
            or review["complete"] is not True or review["strict_point_estimate_parity"] is not False
            or review["all_primary_point_estimates_retained"] is not False
            or review["required_comparisons"] != 392 or review["primary_comparisons"] != 18
            or review["primary_regressions"] != 14 or review["required_regressions"] != 241
            or review["comparison_tolerance"] != 0 or review["practical_equivalence_margin"] is not None
            or review["cuda_initialized"] is not False or review["all_raw_predictions_reaudited"] is not True
            or review["original_campaign_hashes_unchanged"] is not True
            or review["goal_success_not_assessed"] is not True):
        raise ValueError("Require complete unchanged V44 qualification and all adverse real-suite comparisons")


def require_completed_v44():
    if CLOSED_REPORTS is None or len(CLOSED_REPORTS) != 8:
        raise ValueError("Do not register before binding all eight complete immutable V44 reports")
    identities = {str(ROOT / name): digest for name, digest in CLOSED_REPORTS.items()}
    verify_hashes(identities)
    config, training_audit, replay, gate, confirmation_audit, closure, campaign, review = [
        read(ROOT / name) for name in CLOSED_REPORTS]
    validate_completed(training_audit, replay, gate, confirmation_audit, closure, campaign, review)
    for name in ("uniform_v8_retention_v44", "uniform_v8_retention_v44_endpoint_replay",
            "uniform_v8_retention_v44_confirmation", "uniform_v8_retention_v44_pipeline",
            "uniform_v8_retention_v44_benchmarks", "uniform_v8_retention_v44_review"):
        path = ROOT / name / "status.json"
        state = read(path)
        if (state["state"] not in {"complete", "training_complete"}
                or diagnosis.original.process_identity(state["pid"]) is not None):
            raise ValueError("Every original V44 producer must stop before a new scientific probe")
        identities[str(path)] = sha256(path)
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = path.read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if any(argument.rsplit(b"/", 1)[-1] == b"audit_single_model_uniform_v8_retention.py" for argument in arguments):
            raise ValueError("Require the original full auditor to have stopped")
    verify_hashes(config["source_sha256"])
    for report in (training_audit, replay, confirmation_audit):
        verify_hashes(report["artifact_sha256"])
    for record in campaign["audits"].values():
        identities[record["path"]] = record["sha256"]
    review_directory = ROOT / "uniform_v8_retention_v44_review"
    identities[str(review_directory / "comparisons.csv")] = review["comparisons_sha256"]
    identities[str(review_directory / "config.json")] = review["config_sha256"]
    verify_hashes(identities)
    review_registration = read(review_directory / "registration.json")
    review_config = read(review_directory / "config.json")
    if (review_registration["config_sha256"] != review["config_sha256"]
            or review_config["candidate"] != campaign["candidate"] or review["candidate"] != campaign["candidate"]
            or review["model_inventory"] != {"tensors": 254, "elements": 77998688, "changed_tensors": 238}
            or any(review_config[key] is not False for key in (
                "training_performed", "checkpoint_selection_performed", "inference_performed", "original_artifacts_modified"))):
        raise ValueError("Bind complete CPU review, exact single-checkpoint identity and unchanged campaign")
    verify_hashes(review_config["source_sha256"])
    verify_hashes(review_config["input_sha256"])
    return identities


for name in ("reference_identities", "replay_cohort_targets"):
    function = getattr(previous_probe, name)
    if function.__closure__ is not None:
        raise ValueError("Do not retain predecessor runtime closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

SOURCE = previous_probe.SOURCE
for old, new in {
    "import single_model_legacy_blend_teacher as teacher_policy": "import single_model_all_imputation_blend_teacher as teacher_policy",
    "uniform_v8_retention_preflight_v1": "all_imputation_blend_preflight_v1",
    "1111100901, 1112100901": "1141100901, 1142100901",
    "require_completed_v43": "require_completed_v44",
    "blend_replay.ALLOWANCE": "BLEND_COMPONENT_ALLOWANCE",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit all-imputation-blend probe adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
old = '    sources.update(Path(module.__file__).resolve()'
new = ('    sources.update({REPO / "tests/test_single_model_all_imputation_blend_teacher.py", '
    'REPO / "tests/test_single_model_all_imputation_blend_full_replay.py", '
    'REPO / "tests/test_single_model_all_imputation_blend_preflight.py"})\n' + old)
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    f'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, {old!r}, {new!r})\n'
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
exec(compile(SOURCE, __file__, "exec"), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result["tasks"])
    if blend_replay.storage_accounting(result)["reserved_checkpoint_bytes"] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError("Account for every component grid before registering the exact batch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    stage = parser.parse_args().stage
    with diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if stage == "register":
            register(OUTPUT)
            return
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error), failed_stage=stage)
                raise


if __name__ == "__main__":
    main()
