"""Qualify V31 without relaxing any synthetic or frozen-real comparison."""

from copy import deepcopy
import hashlib
import inspect
import types

import run_single_model_legacy_heavy_pipeline as template
from run_single_model_legacy_heavy_pipeline import attempts, chain_template, compact_cache, diagnostics, top_two
from single_model_registered_adaptation import replace_assignment, replace_once


MINIMUM_FREE = template.MINIMUM_FREE
STAGE_WRITE_BUDGETS = {name: dict(amounts) for name, amounts in template.STAGE_WRITE_BUDGETS.items()}
AUDITOR_COMMAND = "python scripts/audit_single_model_legacy_point.py"
POLICY_TEXT = ("Wait for the exact V31 trainer/auditor chain. Keep the unqualified V29 step25 initializer, "
    ".5/.25/.25 task masses and original teachers. Add point metrics to the Adam proposal while preserving "
    "all raw rows and separate legacy protection. Require every admission draw, admitted complete task/teacher "
    "digest and rating context, all64 attempts and1368 reference/parent/development grids. Preserve all22 "
    "original V8 guards, strict minimax selection and ineligible step0. Independently audit the actual full-row "
    "optimizer endpoint, materialize all28 actual tasks, replay the endpoint and sole eligible152-case "
    "selection. No selection means no confirmation consumption or fallback. Otherwise confirm once on284 "
    "fresh reserved cases and independently audit1136 grids. Only then publish and run every frozen tabular, "
    "four-dataset recommender and LLM comparison/review. Preserve18/6/50 GiB reserves and per-volume budgets. "
    "Scratch is machine-local, not provider-persistent storage or backup. No inference ensemble, altered "
    "registered source, scientific retry, gate change, reselection, hidden regression or historical full-"
    "autograd replay claim. Recognize only the registered same-PID/start shell-to-auditor exec handoff; "
    "never launch a duplicate trainer or auditor.")


for name in ("require_stage_capacity", "process_identity", "wait_for_training_audit", "require_compact_audit",
             "actual_endpoint_step", "materialize_endpoint_tasks", "check_materialized_task"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding pipeline closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import audit_single_model_legacy_heavy as training_audit': 'import audit_single_model_legacy_point as training_audit',
    'import train_single_matrixformer_legacy_heavy as training': 'import train_single_matrixformer_legacy_point as training',
    'REPO / "tests/test_single_model_legacy_heavy_training.py"': 'REPO / "tests/test_single_model_legacy_point_training.py"',
    'REPO / "tests/test_single_model_legacy_heavy_pipeline.py"': 'REPO / "tests/test_single_model_legacy_point_pipeline.py"',
    'training.paired.paired_displacement(': 'training.paired.strata.paired_displacement(',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for suffix in ("pipeline", "audit", "endpoint_replay", "confirmation", "benchmarks", "review"):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, f'ROOT / "legacy_heavy_v30_{suffix}"', f'ROOT / "legacy_point_v31_{suffix}"')
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
            or "train_single_matrixformer_legacy_point.py --stage train; " not in expected["command"]):
        return False
    try:
        return Path(f'/proc/{observed["pid"]}/cwd').resolve(strict=True) == REPO
    except (FileNotFoundError, ProcessLookupError):
        return False


source = inspect.getsource(chain_template.register_chain_identity)
for old, new in {
    "train_single_matrixformer_compact_continuation.py": "train_single_matrixformer_legacy_point.py",
    "audit_single_model_compact_continuation.py": "audit_single_model_legacy_point.py",
    "V23 one-shot": "V31 one-shot",
}.items():
    source = replace_once(source, old, new)
exec(compile(source, __file__, "exec"), globals())

replay_actual, audit_replay_actual = replay, audit_replay
for name in ("replay", "audit_replay"):
    function = getattr(template, name)
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)


if __name__ == "__main__":
    main()
