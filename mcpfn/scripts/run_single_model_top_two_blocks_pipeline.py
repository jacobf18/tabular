"""Qualify V26 with the original synthetic gates and entire frozen real campaign."""

from copy import deepcopy
import hashlib
import inspect

import run_single_model_top_block_continuation_pipeline as template
import run_single_model_compact_continuation_pipeline as chain_template
import single_model_compact_training_cache as compact_cache
import single_model_top_two_block_training as top_two
import single_model_scratch_diagnostics as diagnostics
from single_model_registered_adaptation import replace_once, replace_assignment


MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in template.STAGE_WRITE_BUDGETS}
POLICY_TEXT = ("Bind the existing V26 train/audit chain, never duplicate it. Require all64 attempts,1792 complete "
    "independently regenerated task/teacher payloads,128 rating contexts,1216 original reference/development grids, "
    "62 final-two-block/decoder coordinates,192 frozen tensors and the complete84-gradient optimizer endpoint "
    "with unchanged28-row proposal alias. Preserve all22 original V8 guards and initialization eligibility rules. "
    "Materialize only28 final tasks, reconstruct/replay the actual GPU endpoint, independently CPU-audit all "
    "grids and replay only the selected152-case development state without reselection. Confirm once on284 new "
    "reserved cases against original V8/general, independently audit1136 grids and only then publish a best alias. "
    "Run every frozen tabular, four-dataset recommender and LLM comparison/review without omitting regressions. "
    "Keep ordinary checkpoints and real campaigns on checkpoint storage; only new registered synthetic diagnostic "
    "directories use UUID-bound machine-local scratch, not provider-persistent storage or external backup. "
    "Preserve18/6/50 GiB reserves and declared actual-volume write budgets, including publication. No historical "
    "full-autograd claim, inference ensemble, retry, fallback, relaxed gate, consumed-confirmation reuse or "
    "replacement of original teachers/baselines.")


exec(compile(inspect.getsource(template.require_stage_capacity), __file__, "exec"), globals())

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_top_block_continuation as training_audit':
        'import audit_single_model_top_two_blocks as training_audit',
    'import train_single_matrixformer_top_block_continuation as training':
        'import train_single_matrixformer_top_two_blocks as training',
    'OUTPUT = ROOT / "top_block_continuation_v25_pipeline"': 'OUTPUT = ROOT / "top_two_blocks_v26_pipeline"',
    'AUDIT = ROOT / "top_block_continuation_v25_audit"': 'AUDIT = ROOT / "top_two_blocks_v26_audit"',
    'REPLAY = ROOT / "top_block_continuation_v25_endpoint_replay"': 'REPLAY = ROOT / "top_two_blocks_v26_endpoint_replay"',
    'CONFIRMATION = ROOT / "top_block_continuation_v25_confirmation"': 'CONFIRMATION = ROOT / "top_two_blocks_v26_confirmation"',
    'CAMPAIGN = ROOT / "top_block_continuation_v25_benchmarks"': 'CAMPAIGN = ROOT / "top_two_blocks_v26_benchmarks"',
    'REVIEW = ROOT / "top_block_continuation_v25_review"': 'REVIEW = ROOT / "top_two_blocks_v26_review"',
    'audit["through_step"] != 96': 'audit["through_step"] != 64',
    'audit["attempted_batches"] != 96 or audit["training_episodes"] != 2688':
        'audit["attempted_batches"] != 64 or audit["training_episodes"] != 1792',
    'audit["prediction_grids"] != 1368': 'audit["prediction_grids"] != 1216',
    'audit.get("top_block_training") is not True':
        'audit.get("top_block_training") is not True or audit.get("top_two_block_training") is not True',
    'REPO / "tests/test_single_model_top_block_continuation.py"': 'REPO / "tests/test_single_model_top_two_blocks_trial.py"',
    'REPO / "tests/test_single_model_top_block_continuation_pipeline.py"':
        'REPO / "tests/test_single_model_top_two_blocks_pipeline.py"',
    'view = top_block.TopBlockParameterView(model)': 'view = top_two.TopTwoBlockParameterView(model)',
    'result["frozen"]["frozen_tensors"] != 220': 'result["frozen"]["frozen_tensors"] != 192',
    'checkpoint.get("top_block_training") is not True':
        'checkpoint.get("top_block_training") is not True or checkpoint.get("top_two_block_training") is not True',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("process_identity", "register_chain_identity", "wait_for_training_audit", "require_compact_audit",
             "materialize_endpoint_tasks", "check_materialized_task"):
    source = inspect.getsource(getattr(chain_template, name))
    if name == "register_chain_identity":
        source = replace_once(source, "train_single_matrixformer_compact_continuation.py",
                              "train_single_matrixformer_top_two_blocks.py")
        source = replace_once(source, "audit_single_model_compact_continuation.py",
                              "audit_single_model_top_two_blocks.py")
        source = replace_once(source, "V23 one-shot", "V26 one-shot")
    elif name == "require_compact_audit":
        if source.count("2688") != 3 or source.count("192") != 1:
            raise ValueError("Adapt every registered complete task and rating-context count")
        source = source.replace("2688", "1792").replace("192", "128")
        source = replace_once(source, 'or report["autograd_replayed"] is not False',
            'or report["decoder_only_training"] is not False or report.get("top_block_training") is not True\n'
            '            or report.get("top_two_block_training") is not True or report["autograd_replayed"] is not False')
    exec(compile(source, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
