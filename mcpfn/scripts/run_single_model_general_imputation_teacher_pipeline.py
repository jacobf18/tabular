"""Qualify V27 without changing scientific gates or confusing shell exec with process death."""

from copy import deepcopy
import hashlib
import inspect

import run_single_model_top_two_blocks_pipeline as template
import run_single_model_compact_continuation_pipeline as chain_template
from run_single_model_top_two_blocks_pipeline import compact_cache, top_two, diagnostics
from single_model_registered_adaptation import replace_once, replace_assignment


MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in template.STAGE_WRITE_BUDGETS}
AUDITOR_COMMAND = "python scripts/audit_single_model_general_imputation_teacher.py"
POLICY_TEXT = ("Bind the existing one-shot V27 train/audit chain, never duplicate a scientific stage. Require64 "
    "attempts,1792 independently regenerated complete task/teacher payloads,128 rating contexts,1368 original "
    "reference/V26-parent/development grids,62 final-two-block/decoder coordinates,192 frozen tensors and all84 "
    "endpoint gradients with unchanged28-row proposal alias. Preserve all22 original V8 guards, original general "
    "and specialist references, initializer ineligibility, and every individual regression. V26 is only an "
    "audited unqualified initializer, never a teacher or new baseline. Materialize28 final tasks, reconstruct "
    "and independently CPU-audit the actual endpoint, and replay the sole selected152-case development state "
    "without reselection. Confirm once on284 newly reserved synthetic cases under unchanged V8/general gates; "
    "independently audit1136 grids before publishing a best alias. Only then run all frozen tabular, four-dataset "
    "recommender and LLM comparisons/review. Preserve18/6/50 GiB reserves and actual-volume stage write budgets. "
    "Only registered synthetic diagnostics use UUID-bound machine-local scratch, not provider-persistent "
    "storage or external backup. No inference ensemble, hidden required regression, gate relaxation, consumed "
    "confirmation reuse, scientific retry or historical full-autograd replay claim. A same-PID/same-start-time "
    "exec from the registered shell to its exact final auditor in the original working directory is a valid "
    "orchestration transition, not a scientific restart. Reject changed process identity or any other exec.")


exec(compile(inspect.getsource(template.template.require_stage_capacity), __file__, "exec"), globals())

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_top_two_blocks as training_audit':
        'import audit_single_model_general_imputation_teacher as training_audit',
    'import train_single_matrixformer_top_two_blocks as training':
        'import train_single_matrixformer_general_imputation_teacher as training',
    'audit["prediction_grids"] != 1216': 'audit["prediction_grids"] != 1368',
    'REPO / "tests/test_single_model_top_two_blocks_trial.py"':
        'REPO / "tests/test_single_model_general_imputation_training.py"',
    'REPO / "tests/test_single_model_top_two_blocks_pipeline.py"':
        'REPO / "tests/test_single_model_general_imputation_pipeline.py"',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, f'ROOT / "top_two_blocks_v26_{suffix}"',
                                  f'ROOT / "general_imputation_teacher_v27_{suffix}"')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


def chain_is_live(expected):
    observed = process_identity(expected["pid"])
    if observed == expected:
        return True
    if (observed is None or observed["pid"] != expected["pid"]
            or observed["start_ticks"] != expected["start_ticks"]
            or observed["command"] != AUDITOR_COMMAND
            or not expected["command"].endswith("; " + AUDITOR_COMMAND)
            or "train_single_matrixformer_general_imputation_teacher.py --stage train; " not in expected["command"]):
        return False
    try:
        return Path(f'/proc/{observed["pid"]}/cwd').resolve(strict=True) == REPO
    except (FileNotFoundError, ProcessLookupError):
        return False


for name in ("process_identity", "register_chain_identity", "wait_for_training_audit", "require_compact_audit",
             "materialize_endpoint_tasks", "check_materialized_task"):
    source = inspect.getsource(getattr(chain_template, name))
    if name == "register_chain_identity":
        source = replace_once(source, "train_single_matrixformer_compact_continuation.py",
                              "train_single_matrixformer_general_imputation_teacher.py")
        source = replace_once(source, "audit_single_model_compact_continuation.py",
                              "audit_single_model_general_imputation_teacher.py")
        source = replace_once(source, "V23 one-shot", "V27 one-shot")
    elif name == "wait_for_training_audit":
        source = replace_once(source, 'if process_identity(expected["pid"]) != expected:', 'if not chain_is_live(expected):')
    elif name == "require_compact_audit":
        if source.count("2688") != 3 or source.count("192") != 1:
            raise ValueError("Adapt every complete training-task and rating-context count")
        source = source.replace("2688", "1792").replace("192", "128")
        source = replace_once(source, 'or report["autograd_replayed"] is not False',
            'or report["decoder_only_training"] is not False or report.get("top_block_training") is not True\n'
            '            or report.get("top_two_block_training") is not True or report["autograd_replayed"] is not False')
    exec(compile(source, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
