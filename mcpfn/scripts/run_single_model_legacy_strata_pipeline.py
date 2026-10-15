"""Qualify V29 without omitting rejected admissions or fabricating optimizer endpoints."""

import hashlib
import inspect

import run_single_model_anchor_rehearsal_pipeline as template
from run_single_model_anchor_rehearsal_pipeline import chain_template, compact_cache, diagnostics, top_two
import single_model_admission_attempts as attempts
from single_model_registered_adaptation import replace_assignment, replace_once


MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in template.STAGE_WRITE_BUDGETS}
AUDITOR_COMMAND = "python scripts/audit_single_model_legacy_strata.py"
POLICY_TEXT = ("Wait for the exact V29 trainer/auditor chain and independently qualify the sole selected "
    "trained boundary. Require every scheduled structured admission draw, every admitted full task/teacher "
    "digest and rating context, all64 attempts and1368 original/V27-parent/development grids. Preserve "
    "all22 original V8 guards and step0 ineligibility. Distinguish data-only rejection from finite optimizer "
    "rejection, derive exact task counts, and materialize/replay the last actual optimizer endpoint, even "
    "when its ordinal precedes64. If no optimizer attempt exists, independently verify the empty-endpoint "
    "proof and skip neural endpoint replay without fabricating grids or selecting the initializer. Otherwise "
    "replay every before/candidate endpoint grid and the sole selected152-case state. Confirm once on284 "
    "new reserved synthetic cases, audit1136 grids, and only then publish and run every frozen tabular, "
    "four-dataset recommender and LLM comparison/review. Preserve original teachers, normalization, masks, "
    "individual regressions,18/6/50 GiB reserves and stage budgets. Scratch remains machine-local, not "
    "provider-persistent storage or external backup. No ensemble, source change, scientific retry, relaxed "
    "gate, reselection, hidden required regression or historical full-autograd replay claim.")


exec(compile(inspect.getsource(template.template.template.template.require_stage_capacity), __file__, "exec"), globals())
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_anchor_rehearsal as training_audit': 'import audit_single_model_legacy_strata as training_audit',
    'import train_single_matrixformer_anchor_rehearsal as training': 'import train_single_matrixformer_legacy_strata as training',
    'REPO / "tests/test_single_model_anchor_training.py"': 'REPO / "tests/test_single_model_legacy_strata_training.py"',
    'REPO / "tests/test_single_model_anchor_pipeline.py"': 'REPO / "tests/test_single_model_legacy_strata_pipeline.py"',
    'audit["training_episodes"] != 1792': 'audit["training_episodes"] != audit["admission_attempts"]["training_tasks"]',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, f'ROOT / "anchor_rehearsal_v28_{suffix}"', f'ROOT / "legacy_strata_v29_{suffix}"')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

source = replace_once(inspect.getsource(template.template.chain_is_live),
    "train_single_matrixformer_general_imputation_teacher.py --stage train; ",
    "train_single_matrixformer_legacy_strata.py --stage train; ")
exec(compile(source, __file__, "exec"), globals())
for name in ("process_identity", "register_chain_identity", "wait_for_training_audit"):
    source = inspect.getsource(getattr(chain_template, name))
    if name == "register_chain_identity":
        source = replace_once(source, "train_single_matrixformer_compact_continuation.py", "train_single_matrixformer_legacy_strata.py")
        source = replace_once(source, "audit_single_model_compact_continuation.py", "audit_single_model_legacy_strata.py")
        source = replace_once(source, "V23 one-shot", "V29 one-shot")
    elif name == "wait_for_training_audit":
        source = replace_once(source, 'if process_identity(expected["pid"]) != expected:', 'if not chain_is_live(expected):')
    exec(compile(source, __file__, "exec"), globals())


def require_compact_audit(report):
    directory = AUDIT.with_name(AUDIT.name + "_task_replay")
    result, current = read(directory / "result.json"), read(directory / "status.json")
    rows, raw = training_audit.read_prefix(training.OUTPUT / "training.jsonl", range(1, training.STEPS + 1))
    counts = training_audit.admission_inventory(read(training.OUTPUT / "config.json"), rows)
    if (current["state"] != "complete" or result["complete"] is not True
            or result["tasks"] != counts["training_tasks"] or result["audit_sha256"] != sha256(AUDIT / "audit.json")
            or report["admission_attempts"] != counts or report["accepted_updates"] != counts["accepted_updates"]
            or report["teacher_identity_checked_tasks"] != counts["training_tasks"]
            or report["training_teacher_policy"] != training.teacher_policy.POLICY
            or report["independently_regenerated_training_tasks"] != counts["training_tasks"]
            or report["independently_regenerated_rating_contexts"] != counts["rating_contexts"]
            or report["independently_regenerated_structured_draws"] != 24 * training.STEPS
            or report["exact_complete_task_and_teacher_digest_checks"] is not True
            or report["decoder_only_training"] is not False or report.get("top_block_training") is not True
            or report.get("top_two_block_training") is not True
            or report["autograd_replayed"] is not False or report["full_optimizer_trajectory_replayed"] is not False):
        raise ValueError("Require every scheduled admission and every admitted task, with exact independent counts")
    verify_hashes(read(directory / "config.json")["source_sha256"])


def actual_endpoint_step():
    counts = read(AUDIT / "audit.json")["admission_attempts"]
    descriptor = read(training.OUTPUT / "endpoint.json")
    attempts.check_endpoint_descriptor(descriptor, counts, sha256(training.OUTPUT / "config.json"))
    if descriptor["optimizer_attempt_step"] is None:
        raise ValueError("There is no optimizer endpoint to materialize or replay")
    return descriptor["optimizer_attempt_step"]


for name in ("materialize_endpoint_tasks", "check_materialized_task"):
    source = inspect.getsource(getattr(chain_template, name))
    source = source.replace("training.STEPS", "actual_endpoint_step()")
    source = source.replace('config["batches"][-1]["tasks"]', 'config["batches"][actual_endpoint_step() - 1]["tasks"]')
    exec(compile(source, __file__, "exec"), globals())

replay_actual, audit_replay_actual = replay, audit_replay


def replay(output):
    report = prerequisite(output)
    if report["admission_attempts"]["last_optimizer_attempt"] is not None:
        replay_actual(output)
        return
    if (read(REPLAY / "status.json")["state"] != "registered" or report["best"]["step"] != 0
            or report["accepted_updates"] != 0 or report["raw_endpoint"]["unchanged_initializer_checked"] is not True):
        raise ValueError("Skip endpoint inference only for an audited entirely data-rejected unchanged model")
    write_json(REPLAY / "result.json", {"no_optimizer_endpoint": True, "selected": None,
        "endpoint_grids": 0, "selected_grids": 0, "training_audit_sha256": sha256(AUDIT / "audit.json"),
        "student_inference_performed": False, "goal_success_not_assessed": True})
    status(REPLAY, "awaiting_independent_audit")


def audit_replay(output):
    report = prerequisite(output)
    if report["admission_attempts"]["last_optimizer_attempt"] is not None:
        audit_replay_actual(output)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("The empty-endpoint audit must be CPU-only")
    if read(REPLAY / "status.json")["state"] != "awaiting_independent_audit" or (REPLAY / "audit.json").exists():
        raise ValueError("Independently audit the empty-endpoint proof once")
    result = read(REPLAY / "result.json")
    if result != {"no_optimizer_endpoint": True, "selected": None, "endpoint_grids": 0, "selected_grids": 0,
                  "training_audit_sha256": sha256(AUDIT / "audit.json"), "student_inference_performed": False,
                  "goal_success_not_assessed": True} or report["best"]["step"] != 0:
        raise ValueError("An empty endpoint cannot fabricate predictions or a selected model")
    hashes = {str(path): sha256(path) for path in (AUDIT / "audit.json", REPLAY / "result.json")}
    write_json(REPLAY / "audit.json", {"complete": True, "no_optimizer_endpoint": True, "endpoint_grids": 0,
        "selected_grids": 0, "artifact_sha256": hashes, "student_inference_performed": False})
    write_json(REPLAY / "gate.json", {"passed": False, "selected_validation_passed": False,
        "reason": "No optimizer attempt exists; unchanged initialization is ineligible",
        "audit_sha256": sha256(REPLAY / "audit.json")})
    status(REPLAY, "complete")


if __name__ == "__main__":
    main()
