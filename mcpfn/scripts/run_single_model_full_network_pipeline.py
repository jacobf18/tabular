"""Qualify one V34 checkpoint under the unchanged synthetic and frozen-real comparisons."""

from copy import deepcopy
import hashlib
import inspect
import types

import run_single_model_top_three_blocks_pipeline as template
from run_single_model_top_three_blocks_pipeline import attempts, chain_template, compact_cache
import single_model_full_network_training as full_network
import single_model_full_network_storage as diagnostics
from single_model_registered_adaptation import replace_assignment, replace_once


MINIMUM_FREE = diagnostics.RESERVES["checkpoint"]
STAGE_WRITE_BUDGETS = {name: dict(diagnostics.WRITE_BUDGETS[name]) for name in template.STAGE_WRITE_BUDGETS}
AUDITOR_COMMAND = "python scripts/audit_single_model_full_network.py"
POLICY_TEXT = ("Wait for the exact V34 trainer/auditor chain. Initialize only from unqualified V29 step25; "
    "train all238 original parameter tensors while preserving16 buffers and unchanged single-network inference. "
    "Retain current priors, teachers, objectives,30 point constraints and all84 raw/proposal gradients. Require "
    "every admission draw, admitted complete teacher/task digest and rating context,64 attempts and1368 "
    "reference/parent/development grids. Independently audit all optimizer coordinates, materialize all28 "
    "actual endpoint tasks and replay56 endpoint and152 sole-selected grids. Keep all22 original development "
    "guards, strict minimax selection, earlier ties and ineligible step0. No selection means no confirmation, "
    "fallback or fabricated predictions. Otherwise confirm once on284 fresh reserved cases and audit1136 grids "
    "against all32 unchanged comparisons. Only then publish and run every frozen tabular, four-dataset "
    "recommender and LLM comparison/review. Preserve18/6/50 GiB reserves and original qualification/publication/"
    "real-evaluation budgets. Only training storage allowances expand prospectively. Scratch is machine-local, "
    "not provider-persistent storage or backup. No inference ensemble, source mutation, scientific retry, gate "
    "relaxation, reselection, hidden regression or historical full-autograd replay claim. Recognize only the "
    "registered same-PID/start shell-to-auditor exec handoff; never duplicate the trainer or auditor.")


for name in ("require_stage_capacity", "process_identity", "wait_for_training_audit", "require_compact_audit",
             "actual_endpoint_step", "materialize_endpoint_tasks", "check_materialized_task"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding pipeline closure")
    code = function.__code__
    if name == "require_compact_audit":
        if code.co_consts.count("top_three_block_training") != 1:
            raise ValueError("Change only the explicitly registered optimizer scope flag")
        code = code.replace(co_consts=tuple("full_network_training" if value == "top_three_block_training" else value
                                           for value in code.co_consts))
    cloned = types.FunctionType(code, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_top_three_blocks as training_audit': 'import audit_single_model_full_network as training_audit',
    'import train_single_matrixformer_top_three_blocks as training': 'import train_single_matrixformer_full_network as training',
    'REPO / "tests/test_single_model_top_three_blocks_training.py"': 'REPO / "tests/test_single_model_full_network_trial.py"',
    'REPO / "tests/test_single_model_top_three_blocks_pipeline.py"': 'REPO / "tests/test_single_model_full_network_pipeline.py"',
    'view = top_three.TopThreeBlockParameterView(model)': 'view = full_network.FullNetworkParameterView(model)',
    'audit.get("top_three_block_training") is not True': 'audit.get("full_network_training") is not True',
    'checkpoint.get("top_three_block_training") is not True': 'checkpoint.get("full_network_training") is not True',
    'result["frozen"]["frozen_tensors"] != 164': 'result["frozen"]["frozen_tensors"] != 16',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, f'ROOT / "top_three_blocks_v33_{suffix}"', f'ROOT / "full_network_v34_{suffix}"')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


def chain_is_live(expected):
    observed = process_identity(expected["pid"])
    if observed == expected:
        return True
    if (observed is None or observed["pid"] != expected["pid"]
            or observed["start_ticks"] != expected["start_ticks"] or observed["command"] != AUDITOR_COMMAND
            or not expected["command"].endswith("; " + AUDITOR_COMMAND)
            or "train_single_matrixformer_full_network.py --stage train; " not in expected["command"]):
        return False
    try:
        return Path(f'/proc/{observed["pid"]}/cwd').resolve(strict=True) == REPO
    except (FileNotFoundError, ProcessLookupError):
        return False


source = inspect.getsource(chain_template.register_chain_identity)
for old, new in {
    "train_single_matrixformer_compact_continuation.py": "train_single_matrixformer_full_network.py",
    "audit_single_model_compact_continuation.py": "audit_single_model_full_network.py",
    "V23 one-shot": "V34 one-shot",
}.items():
    source = replace_once(source, old, new)
exec(compile(source, __file__, "exec"), globals())

replay_actual, audit_replay_actual = replay, audit_replay
for name in ("replay", "audit_replay"):
    function = getattr(template, name)
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)


if __name__ == "__main__":
    main()
