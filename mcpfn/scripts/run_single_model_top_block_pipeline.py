"""Qualify the existing V24 last-block trial without repeating training or its audit."""

from copy import deepcopy
import hashlib
import inspect

import run_single_model_compact_continuation_pipeline as template
import single_model_compact_training_cache as compact_cache
import single_model_top_block_training as top_block
from single_model_registered_adaptation import replace_once, replace_assignment


MINIMUM_FREE = 18 * 1024**3
STAGE_WRITE_BUDGETS = dict(template.STAGE_WRITE_BUDGETS)
POLICY_TEXT = ("Wait for all64 V24 attempts and the existing one-shot train/audit chain's independent1792-task "
    "regeneration audit; never launch a duplicate trainer or auditor. Preserve the34-coordinate last-block/decoder "
    "inventory,220 frozen state tensors, all22 original V8 guards,1216 development/reference grids and the complete "
    "84-row terminal evidence with its unchanged28-row Adam proposal. Materialize only28 final task payloads, "
    "verify complete typed task/teacher digests, reconstruct the GPU endpoint and independently CPU-score every "
    "grid. Replay the sole selected152-case development checkpoint without reselection. Confirm once on the "
    "registered284 fresh cases and independently audit all1136 grids under unchanged V8/general retention gates. "
    "Only then publish a best alias and run every frozen tabular, four-dataset recommender and LLM benchmark and "
    "required comparison. Preserve18 GiB checkpoint plus declared stage budgets,6 GiB root and50 GiB local scratch. "
    "No inference ensemble, retry, fallback, threshold relaxation, reuse of consumed confirmation, hidden "
    "regression or historical autograd-replay claim. Scratch is machine-local, not provider-persistent backup.")


def require_stage_capacity(label):
    if label not in STAGE_WRITE_BUDGETS:
        raise ValueError("Every qualification stage needs its declared capacity budget")
    for path, reserve, budget in ((ROOT, MINIMUM_FREE, STAGE_WRITE_BUDGETS[label]),
                                 (Path("/"), 6 * 1024**3, 0),
                                 (training.SCRATCH.parent, 50 * 1024**3, 0)):
        free = shutil.disk_usage(path).free
        if free - budget < reserve:
            raise OSError(f"Preserve {reserve} bytes before {label} on {path}: free={free}, budget={budget}")


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_compact_continuation as training_audit':
        'import audit_single_model_top_block_training as training_audit',
    'import train_single_matrixformer_compact_continuation as training':
        'import train_single_matrixformer_top_block as training',
    'OUTPUT = ROOT / "compact_continuation_v23_pipeline"': 'OUTPUT = ROOT / "top_block_v24_pipeline"',
    'AUDIT = ROOT / "compact_continuation_v23_audit"': 'AUDIT = ROOT / "top_block_v24_audit"',
    'REPLAY = ROOT / "compact_continuation_v23_endpoint_replay"': 'REPLAY = ROOT / "top_block_v24_endpoint_replay"',
    'CONFIRMATION = ROOT / "compact_continuation_v23_confirmation"': 'CONFIRMATION = ROOT / "top_block_v24_confirmation"',
    'CAMPAIGN = ROOT / "compact_continuation_v23_benchmarks"': 'CAMPAIGN = ROOT / "top_block_v24_benchmarks"',
    'REVIEW = ROOT / "compact_continuation_v23_review"': 'REVIEW = ROOT / "top_block_v24_review"',
    'audit["through_step"] != 96': 'audit["through_step"] != 64',
    'audit["attempted_batches"] != 96 or audit["training_episodes"] != 2688':
        'audit["attempted_batches"] != 64 or audit["training_episodes"] != 1792',
    'not audit["decoder_only_training"] or audit["prediction_grids"] != 1368':
        'audit["decoder_only_training"] or audit.get("top_block_training") is not True or audit["prediction_grids"] != 1216',
    'REPO / "tests/test_single_model_compact_continuation_audit.py"':
        'REPO / "tests/test_single_model_top_block_trial.py"',
    'REPO / "tests/test_single_model_compact_continuation_pipeline.py"':
        'REPO / "tests/test_single_model_top_block_pipeline.py"',
    'view = decoder.DecoderParameterView(model)': 'view = top_block.TopBlockParameterView(model)',
    'result["frozen"]["frozen_tensors"] != 248 or not result["frozen"]["backbone_and_buffers_bitwise_unchanged"]':
        'result["frozen"]["frozen_tensors"] != 220 or not result["frozen"]["unselected_parameters_and_all_buffers_bitwise_unchanged"]',
    'Replayed nondecoder state must remain frozen': 'Replayed unselected parameters and every buffer must remain frozen',
    'if checkpoint["step"] != audit["best"]["step"] or checkpoint["gradient_updates_performed"] < 1:':
        'if (checkpoint["step"] != audit["best"]["step"] or checkpoint["gradient_updates_performed"] < 1\n'
        '                or checkpoint["decoder_only_training"] or checkpoint.get("top_block_training") is not True):',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
frozen_expression = 'decoder.frozen_state_equal(parent, model.state_dict(), mutable_names)'
if ADAPTED_SOURCE.count(frozen_expression) != 2:
    raise ValueError("Adapt both endpoint and selected-checkpoint frozen-state certificates")
ADAPTED_SOURCE = ADAPTED_SOURCE.replace(frozen_expression,
    'training.preflight.frozen_inventory(parent, model.state_dict(), mutable_names)')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("process_identity", "register_chain_identity", "wait_for_training_audit", "require_compact_audit",
             "materialize_endpoint_tasks", "check_materialized_task"):
    helper_source = inspect.getsource(getattr(template, name))
    if name == "register_chain_identity":
        helper_source = replace_once(helper_source, "train_single_matrixformer_compact_continuation.py",
                                     "train_single_matrixformer_top_block.py")
        helper_source = replace_once(helper_source, "audit_single_model_compact_continuation.py",
                                     "audit_single_model_top_block_training.py")
        helper_source = replace_once(helper_source, "V23 one-shot", "V24 one-shot")
    if name == "require_compact_audit":
        if helper_source.count("2688") != 3 or helper_source.count("192") != 1:
            raise ValueError("Adapt all complete compact task and rating-context audit counts")
        helper_source = helper_source.replace("2688", "1792").replace("192", "128")
        helper_source = replace_once(helper_source, 'or report["autograd_replayed"] is not False',
            'or report["decoder_only_training"] is not False or report.get("top_block_training") is not True\n'
            '            or report["autograd_replayed"] is not False')
    exec(compile(helper_source, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
