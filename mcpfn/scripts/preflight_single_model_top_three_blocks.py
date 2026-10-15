"""Probe a three-block point-protected update after V32's audited failed confirmation."""

import hashlib
import inspect
import types

import torch

import preflight_single_model_point_only as template
from preflight_single_model_point_only import (
    admission, preceding, rehearsal, replicated, schedule, storage, teacher_policy, StructuredSampler,
    sync_directory, composite, mixture, paired,
)
import run_single_model_point_only_pipeline as latest
import single_model_top_three_block_training as top_three
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES = ("transformer_blocks.5.", "transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
TRAINABLE_TENSORS, TRAINABLE_PARAMETERS, FROZEN_TENSORS = 90, 36036497, 164
PARENT_SHA256 = template.PARENT_SHA256
WRITE_BUDGETS, RESERVES = dict(template.WRITE_BUDGETS), dict(template.RESERVES)
SELECTED_V32_SHA256 = "056399d2487cd1d907be08a2c43cf9bdba805b9e5f04f05d0152f24d98dff66b"
STORAGE_ACCOUNTING = {"raw_gradient_bytes": 84 * 4 * TRAINABLE_PARAMETERS,
    "snapshot_vector_allowance_bytes": 8 * 4 * TRAINABLE_PARAMETERS,
    "ordinary_checkpoint_allowance_bytes": 2 * 4 * 77998496,
    "prediction_task_allowance_bytes": 1024**3,
    "checkpoint_artifact_allowance_bytes": 4 * TRAINABLE_PARAMETERS + 64 * 1024**2}
POLICY_TEXT = ("One fresh28-task synthetic optimizer feasibility probe after fully audited unsuccessful V32 "
    "confirmation. Initialize only from the same unqualified V29 step25 as V30/V31/V32, never V32 or a diagnostic "
    "endpoint. Expand only mutable training depth to existing blocks5/6/7 plus decoder:90 parameter tensors, "
    "36,036,497 parameters and164 bitwise-frozen state tensors, including all selected-prefix RoPE buffers. "
    "Preserve the ordinary77,998,496-parameter inference network and state layout. " + top_three.POLICY + " "
    + mixture.POLICY + " " + paired.POLICY + " " + composite.POLICY + " Retain original teachers, native-anchor "
    "rehearsal, observed-only normalization, all Huber/NLL/CDF/query/point losses, half-composite/half-point "
    "proposal weights, fresh5e-6 AdamW,0.1 inward margin,2x displacement bound and exact rollback. Retain all84 "
    "raw/proposal rows when metrics exist and30 point constraints, including adverse composite diagnostics. "
    "Independently regenerate24 admission draws, replay28 teachers, audit56 before/candidate grids and the "
    "actual optimizer/full model state. Data-only rejection does no model work and fabricates no metric. "
    "No validation, confirmation, nomination, real evaluation, consumed-seed reuse, scientific retry, inference "
    "ensemble or change to held-out gates. Diagnostic endpoints cannot initialize future training. This is "
    "feasibility, not accuracy or proof of a capacity bottleneck. Preserve18/6/50 GiB reserves and256 MiB/16 MiB/"
    "17 GiB checkpoint/root/scratch write budgets with explicit expanded full-row artifact accounting. Only "
    "new declared diagnostic checkpoint/raw evidence uses UUID-bound local scratch, not provider-persistent "
    "storage or external backup. Never modify any previously registered scientific source.")


def validate_latest(result, training_audit, replay_audit, confirmation_audit, gate, digest):
    if (result["complete"] is not True or result["gate"] != gate or gate["passed"] is not False
            or gate["confirmation_evaluated"] is not True or gate["selected_step"] != 10
            or gate["cases"] != 284 or gate["checkpoint_sha256"] != digest or digest != SELECTED_V32_SHA256
            or result["campaign"] is not None or result["review"] is not None
            or result["no_inference_ensemble"] is not True
            or training_audit["complete"] is not True or training_audit["complete_training_audit"] is not True
            or training_audit["through_step"] != 64 or training_audit["accepted_updates"] != 61
            or training_audit["best"]["step"] != 10 or training_audit["prediction_grids"] != 1368
            or training_audit["independently_regenerated_training_tasks"] != 1764
            or training_audit["independently_regenerated_rating_contexts"] != 126
            or training_audit["independently_regenerated_structured_draws"] != 1536
            or training_audit["admission_attempts"]["data_rejected_steps"] != [51]
            or training_audit["admission_attempts"]["optimizer_attempts"] != 63
            or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or training_audit["raw_endpoint"]["optimizer_attempt_step"] != 64
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 152 or confirmation_audit["complete"] is not True
            or confirmation_audit["cases"] != 284 or confirmation_audit["prediction_grids"] != 1136):
        raise ValueError("Require complete V32 failure, its data-only no-op and original independent audits")


source = inspect.getsource(template.require_predecessor)
source = replace_once(source, "original V31 scientific stage", "original V32 scientific stage")
source = replace_once(source, "unqualified V31", "unqualified V32")
exec(compile(source, __file__, "exec"), globals())

for name in ("require_space", "save_raw", "audit_raw", "expected_role", "check_routing", "audit_teacher_replay",
             "admit_batch", "audit_admission", "save_probe_checkpoint", "audit_probe_checkpoint"):
    function = inspect.unwrap(getattr(template, name))
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding preflight helper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

require_volume_space = require_space


def require_space():
    scratch_bytes = sum(value for key, value in STORAGE_ACCOUNTING.items() if key != "checkpoint_artifact_allowance_bytes")
    if (scratch_bytes > WRITE_BUDGETS["scratch"]
            or STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] > WRITE_BUDGETS["checkpoint"]):
        raise ValueError("Expanded full-row evidence must fit declared stage allowances beyond all reserves")
    require_volume_space()


def frozen_inventory(initial, current, mutable_names):
    expected = {row[0] for row in read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json")
                if row[0].startswith(PREFIXES)}
    if (set(initial) != set(current) or set(mutable_names) != expected or len(expected) != TRAINABLE_TENSORS
            or len(set(initial) - expected) != FROZEN_TENSORS):
        raise ValueError("Preserve the original full model,90 mutable parameters and164 frozen state tensors")
    changed, frozen = [], 0
    for name, before in initial.items():
        after = current[name]
        if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
            raise ValueError("Every model tensor must retain its finite original shape and dtype")
        equal = composite.byte_equal(before.detach().cpu(), after.detach().cpu())
        if name not in expected:
            frozen += 1
            if not equal:
                raise ValueError(f"Frozen parameter or buffer changed: {name}")
        elif not equal:
            changed.append(name)
    return {"frozen_tensors": frozen, "changed_trainable_tensors": changed,
            "unselected_parameters_and_all_buffers_bitwise_unchanged": True}


REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    "point-only-protection preflight directories": "three-block point-protected preflight directories",
    'REPO / "tests/test_single_model_point_only_preflight.py"': 'REPO / "tests/test_single_model_top_three_blocks_preflight.py"',
    'REPO / "tests/test_single_model_top_two_block_training.py"': 'REPO / "tests/test_single_model_top_three_block_training.py"',
    'config = {"policy": POLICY,':
        'config = {"policy": POLICY, "optimizer_view_policy": top_three.POLICY, "storage_accounting": STORAGE_ACCOUNTING,',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "point_only_preflight_v1"': 'OUTPUT = ROOT / "top_three_blocks_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/point_only_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/top_three_blocks_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 741100901, 742100901': 'DATA_SEED, GRADIENT_SEED = 771100901, 772100901',
    'view = top_two.TopTwoBlockParameterView(model)': 'view = top_three.TopThreeBlockParameterView(model)',
    'with top_two.capture_coordinates(model, view):': 'with top_three.capture_coordinates(model, view):',
    'or config["policy"] != POLICY':
        'or config["optimizer_view_policy"] != top_three.POLICY or config["storage_accounting"] != STORAGE_ACCOUNTING\n'
        '            or config["policy"] != POLICY',
    '# Point-only protected optimizer preflight': '# Three-block point-protected optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

audit_admitted = audit
function = inspect.unwrap(template.replay_admitted)
if function.__closure__ is not None:
    raise ValueError("Never retain a preceding teacher replay closure")
cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
cloned.__kwdefaults__ = function.__kwdefaults__
replay_admitted = torch.no_grad()(cloned)
for name in ("replay_targets", "audit"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding admission wrapper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
