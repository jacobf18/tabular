"""Qualify V28 once under every unchanged synthetic and frozen real benchmark gate."""

from copy import deepcopy
import hashlib
import inspect

import run_single_model_general_imputation_teacher_pipeline as template
from run_single_model_general_imputation_teacher_pipeline import chain_template, compact_cache, diagnostics, top_two
from single_model_registered_adaptation import replace_assignment, replace_once


MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in template.STAGE_WRITE_BUDGETS}
AUDITOR_COMMAND = "python scripts/audit_single_model_anchor_rehearsal.py"
POLICY_TEXT = ("Wait for the exact V28 anchor-rehearsal trainer/auditor chain, then independently qualify the "
    "sole selected trained boundary. Require all1792 regenerated task/teacher payloads,128 rating contexts,64 "
    "attempts and1368 complete original/V27-parent/development prediction grids,62 two-block/decoder coordinates, "
    "192 frozen state tensors and all84 terminal gradients with the exact28-row proposal alias. V27 selected "
    "step10 is only an unqualified initializer, never a teacher or new retention baseline. Preserve all22 "
    "original V8 guards, original general/specialist references, step0 ineligibility and individual regressions. "
    "Materialize all28 endpoint tasks, reconstruct/replay and CPU-audit the actual endpoint and sole selected152 "
    "development cases. Confirm once on284 newly reserved cases, audit all1136 grids under unchanged gates, "
    "and only then publish a best alias and run every frozen tabular, four-dataset recommender and LLM "
    "comparison/review. Preserve18/6/50 GiB reserves and actual-volume write budgets. Only registered synthetic "
    "diagnostics and raw endpoint evidence use UUID-bound local scratch, not external backup or provider-"
    "persistent storage. Preserve valid same-PID/same-start shell-to-exact-auditor exec handoff identity. No "
    "inference ensemble, source change, repeated scientific stage, reselection, relaxed gate, consumed-bank "
    "reuse, hidden required regression or historical full-autograd replay claim.")


exec(compile(inspect.getsource(template.template.template.require_stage_capacity), __file__, "exec"), globals())

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_general_imputation_teacher as training_audit':
        'import audit_single_model_anchor_rehearsal as training_audit',
    'import train_single_matrixformer_general_imputation_teacher as training':
        'import train_single_matrixformer_anchor_rehearsal as training',
    'REPO / "tests/test_single_model_general_imputation_training.py"':
        'REPO / "tests/test_single_model_anchor_training.py"',
    'REPO / "tests/test_single_model_general_imputation_pipeline.py"':
        'REPO / "tests/test_single_model_anchor_pipeline.py"',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, f'ROOT / "general_imputation_teacher_v27_{suffix}"',
                                  f'ROOT / "anchor_rehearsal_v28_{suffix}"')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

source = replace_once(inspect.getsource(template.chain_is_live),
    "train_single_matrixformer_general_imputation_teacher.py --stage train; ",
    "train_single_matrixformer_anchor_rehearsal.py --stage train; ")
exec(compile(source, __file__, "exec"), globals())

for name in ("process_identity", "register_chain_identity", "wait_for_training_audit", "require_compact_audit",
             "materialize_endpoint_tasks", "check_materialized_task"):
    source = inspect.getsource(getattr(chain_template, name))
    if name == "register_chain_identity":
        source = replace_once(source, "train_single_matrixformer_compact_continuation.py", "train_single_matrixformer_anchor_rehearsal.py")
        source = replace_once(source, "audit_single_model_compact_continuation.py", "audit_single_model_anchor_rehearsal.py")
        source = replace_once(source, "V23 one-shot", "V28 one-shot")
    elif name == "wait_for_training_audit":
        source = replace_once(source, 'if process_identity(expected["pid"]) != expected:', 'if not chain_is_live(expected):')
    elif name == "require_compact_audit":
        if source.count("2688") != 3 or source.count("192") != 1:
            raise ValueError("Adapt every complete task and rating-context count")
        source = source.replace("2688", "1792").replace("192", "128")
        source = replace_once(source, 'or report["autograd_replayed"] is not False',
            'or report["decoder_only_training"] is not False or report.get("top_block_training") is not True\n'
            '            or report.get("top_two_block_training") is not True or report["autograd_replayed"] is not False')
    exec(compile(source, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
