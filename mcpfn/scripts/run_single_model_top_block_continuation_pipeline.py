"""Qualify V25 through its existing training/audit chain and unchanged scientific gates."""

from copy import deepcopy
import hashlib
import inspect

import run_single_model_top_block_pipeline as template
import single_model_compact_training_cache as compact_cache
import single_model_top_block_training as top_block
import single_model_scratch_diagnostics as diagnostics
from single_model_registered_adaptation import replace_once, replace_assignment


MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in template.STAGE_WRITE_BUDGETS}
POLICY_TEXT = ("Bind the existing one-shot V25 train/audit chain and wait for all96 attempts and its independent "
    "2688-task/192-rating-context audit; never start another trainer or auditor. Require1368 reference/parent/"
    "development grids,34 last-block/decoder coordinates,220 frozen tensors, all22 original V8 guards and the "
    "complete84-gradient endpoint with unchanged28-row proposal alias. Materialize only28 final tasks with exact "
    "typed task/teacher digests, reconstruct the actual GPU endpoint and independently CPU-audit all grids. "
    "Replay only the selected152-case development checkpoint without reselection. Confirm once on the reserved "
    "284 fresh cases against original V8/general and independently audit all1136 grids before publishing a best "
    "alias. Only then run every frozen tabular, four-dataset recommender and LLM comparison/review. V24 remains "
    "an unqualified training input, never a teacher or new retention baseline. New synthetic endpoint/confirmation "
    "diagnostics use prospectively registered scratch links; selected/latest/published checkpoints and the real "
    "campaign remain on checkpoint storage. Preserve18/6/50 GiB checkpoint/root/scratch reserves and declared "
    "per-volume budgets including checkpoint publication. Scratch is not provider-persistent storage or external "
    "backup. No retry, fallback, relaxed scientific gate, consumed-confirmation reuse, hidden regression, "
    "inference ensemble or historical full-autograd claim.")


def require_stage_capacity(label):
    if label not in STAGE_WRITE_BUDGETS:
        raise ValueError("Every qualification stage needs its registered per-volume budget")
    diagnostics.require_capacity(label)
    diagnostics.verify(read(training.OUTPUT / "config.json")["diagnostic_storage"], "qualification")


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_top_block_training as training_audit':
        'import audit_single_model_top_block_continuation as training_audit',
    'import train_single_matrixformer_top_block as training':
        'import train_single_matrixformer_top_block_continuation as training',
    'OUTPUT = ROOT / "top_block_v24_pipeline"': 'OUTPUT = ROOT / "top_block_continuation_v25_pipeline"',
    'AUDIT = ROOT / "top_block_v24_audit"': 'AUDIT = ROOT / "top_block_continuation_v25_audit"',
    'REPLAY = ROOT / "top_block_v24_endpoint_replay"': 'REPLAY = ROOT / "top_block_continuation_v25_endpoint_replay"',
    'CONFIRMATION = ROOT / "top_block_v24_confirmation"': 'CONFIRMATION = ROOT / "top_block_continuation_v25_confirmation"',
    'CAMPAIGN = ROOT / "top_block_v24_benchmarks"': 'CAMPAIGN = ROOT / "top_block_continuation_v25_benchmarks"',
    'REVIEW = ROOT / "top_block_v24_review"': 'REVIEW = ROOT / "top_block_continuation_v25_review"',
    'audit["through_step"] != 64': 'audit["through_step"] != 96',
    'audit["attempted_batches"] != 64 or audit["training_episodes"] != 1792':
        'audit["attempted_batches"] != 96 or audit["training_episodes"] != 2688',
    'audit["prediction_grids"] != 1216': 'audit["prediction_grids"] != 1368',
    'REPO / "tests/test_single_model_top_block_trial.py"':
        'REPO / "tests/test_single_model_top_block_continuation.py"',
    'REPO / "tests/test_single_model_top_block_pipeline.py"':
        'REPO / "tests/test_single_model_top_block_continuation_pipeline.py"',
    '    for directory in (REPLAY, CONFIRMATION):\n        directory.mkdir()':
        '    diagnostics.prepare(read(training.OUTPUT / "config.json")["diagnostic_storage"], "qualification")\n'
        '    for directory in (REPLAY, CONFIRMATION):',
    '    return config\n':
        '    diagnostics.verify(read(training.OUTPUT / "config.json")["diagnostic_storage"], "qualification")\n'
        '    return config\n',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("process_identity", "register_chain_identity", "wait_for_training_audit", "require_compact_audit",
             "materialize_endpoint_tasks", "check_materialized_task"):
    source = inspect.getsource(getattr(template.template, name))
    if name == "register_chain_identity":
        source = replace_once(source, "train_single_matrixformer_compact_continuation.py",
                              "train_single_matrixformer_top_block_continuation.py")
        source = replace_once(source, "audit_single_model_compact_continuation.py",
                              "audit_single_model_top_block_continuation.py")
        source = replace_once(source, "V23 one-shot", "V25 one-shot")
    if name == "require_compact_audit":
        source = replace_once(source, 'or report["autograd_replayed"] is not False',
            'or report["decoder_only_training"] is not False or report.get("top_block_training") is not True\n'
            '            or report["autograd_replayed"] is not False')
    exec(compile(source, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
