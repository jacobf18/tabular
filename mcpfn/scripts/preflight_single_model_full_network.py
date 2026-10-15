"""Probe full-network point-protected training after V33's independently audited rejection."""

import hashlib
import inspect
import types

import torch

import preflight_single_model_top_three_blocks as template
from preflight_single_model_top_three_blocks import (
    admission, preceding, rehearsal, replicated, schedule, storage, teacher_policy, StructuredSampler,
    sync_directory, composite, mixture, paired,
)
import run_single_model_top_three_blocks_pipeline as latest
import single_model_full_network_training as full_network
from single_model_registered_adaptation import replace_assignment, replace_once


PREFIXES, PARENT_SHA256 = full_network.PREFIXES, template.PARENT_SHA256
TRAINABLE_TENSORS, TRAINABLE_PARAMETERS, FROZEN_TENSORS = 238, 77998496, 16
RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {"checkpoint": 384 * 1024**2, "root": 16 * 1024**2, "scratch": 32 * 1024**3}
STORAGE_ACCOUNTING = {"raw_gradient_bytes": 84 * 4 * TRAINABLE_PARAMETERS,
    "snapshot_vector_allowance_bytes": 8 * 4 * TRAINABLE_PARAMETERS,
    "ordinary_checkpoint_allowance_bytes": 2 * 4 * TRAINABLE_PARAMETERS,
    "prediction_task_allowance_bytes": 1024**3,
    "checkpoint_artifact_allowance_bytes": 4 * TRAINABLE_PARAMETERS + 64 * 1024**2}
POLICY_TEXT = ("One fresh28-task synthetic optimizer feasibility test after V33 completes64 accepted updates, "
    "independent full task/optimizer/development audit and56-grid endpoint replay without any eligible selection. "
    "Initialize only from the same audited unqualified V29 step25, never a V33 tail or diagnostic checkpoint. "
    "Expand only mutable training coordinates to the original encoder, all eight blocks and decoder:238 "
    "parameter tensors/77,998,496 parameters. Preserve all16 buffers bitwise and ordinary single-model inference. "
    + full_network.POLICY + " " + mixture.POLICY + " " + paired.POLICY + " " + composite.POLICY
    + " Preserve original general-imputation/specialist-rating teachers, native-anchor rehearsal, observed-only "
    "normalization, Huber/NLL/CDF/query/point losses, .5/.25/.25 masses, half-composite/half-point proposal, "
    "fresh5e-6 AdamW,0.1 margin,2x norm bound and exact rollback. Retain all84 raw/proposal rows when defined "
    "and30 point constraints, including every adverse individual and composite diagnostic. Independently "
    "regenerate24 admission draws, replay28 complete teacher grids and audit56 before/candidate grids and actual "
    "full optimizer/model state. Reject inadmissible data without model work, redraw or fabricated metrics. "
    "No validation, confirmation, model nomination, real evaluation, consumed-seed reuse, scientific retry, "
    "inference ensemble or gate change. This probes feasibility, not accuracy or proof of a capacity bottleneck. "
    "Diagnostic endpoints cannot initialize future training. Preserve18/6/50 GiB reserves; independently expanded "
    "write allowances are384 MiB checkpoint,16 MiB root and32 GiB scratch, including every full raw-gradient row "
    "and eight parameter vectors. Only declared diagnostics use UUID-bound machine-local scratch, not provider-"
    "persistent storage or external backup. No previously registered source or artifact changes.")


def validate_latest(result, training_audit, replay_audit, replay_gate, gate):
    if (result["complete"] is not True or result["gate"] != gate or gate["passed"] is not False
            or gate["confirmation_evaluated"] is not False or gate["selected_step"] != 0
            or gate["real_benchmark_success"] is not False or result["campaign"] is not None or result["review"] is not None
            or result["no_inference_ensemble"] is not True
            or training_audit["complete"] is not True or training_audit["complete_training_audit"] is not True
            or training_audit["through_step"] != 64 or training_audit["attempted_batches"] != 64
            or training_audit["accepted_updates"] != 64 or training_audit["best"]["step"] != 0
            or training_audit["prediction_grids"] != 1368 or training_audit["top_three_block_training"] is not True
            or training_audit["independently_regenerated_training_tasks"] != 1792
            or training_audit["independently_regenerated_rating_contexts"] != 128
            or training_audit["independently_regenerated_structured_draws"] != 1536
            or training_audit["admission_attempts"]["data_rejected_steps"] != []
            or training_audit["admission_attempts"]["optimizer_attempts"] != 64
            or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or training_audit["raw_endpoint"]["optimizer_attempt_step"] != 64
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 0 or replay_gate["endpoint_reproduced"] is not True
            or replay_gate["selected_validation_passed"] is not False or replay_gate["selected_step"] != 0):
        raise ValueError("Require complete independently audited V33 rejection without selection or confirmation")


def require_predecessor():
    evidence = template.require_predecessor()
    directories = (latest.training.OUTPUT, latest.OUTPUT, latest.REPLAY, latest.CONFIRMATION,
                   latest.AUDIT.with_name(latest.AUDIT.name + "_task_replay"))
    for directory in directories:
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or Path(f'/proc/{current["pid"]}').exists():
            raise ValueError("Finish and stop every original V33 scientific stage before another probe")
    report = latest.prerequisite(latest.OUTPUT)
    replay, replay_gate = [read(latest.REPLAY / name) for name in ("audit.json", "gate.json")]
    result, gate = read(latest.OUTPUT / "result.json"), read(latest.CONFIRMATION / "gate.json")
    validate_latest(result, report, replay, replay_gate, gate)
    if (result["training_audit_sha256"] != sha256(latest.AUDIT / "audit.json")
            or replay_gate["audit_sha256"] != sha256(latest.REPLAY / "audit.json")):
        raise ValueError("Bind both complete original independent audits")
    verify_hashes(replay["artifact_sha256"])
    forbidden = [latest.training.OUTPUT / name for name in ("selected_checkpoint.pth", "best_validation_checkpoint.pth")]
    forbidden.extend((latest.CAMPAIGN, latest.REVIEW, latest.CONFIRMATION / "audit.json", latest.CONFIRMATION / "manifest.json"))
    if any(path.exists() for path in forbidden):
        raise ValueError("Preserve V33's absent eligible selection and unconsumed confirmation, without a real campaign")
    paths = [directory / "status.json" for directory in directories]
    paths.extend(latest.OUTPUT / name for name in ("config.json", "registration.json", "result.json"))
    paths.extend([latest.training.OUTPUT / "config.json", latest.AUDIT / "audit.json", latest.CONFIRMATION / "gate.json"])
    paths.extend(latest.REPLAY / name for name in ("audit.json", "gate.json"))
    return {**evidence, **{str(path): sha256(path) for path in paths}}


for name in ("require_space", "require_volume_space", "save_raw", "audit_raw", "expected_role", "check_routing",
             "audit_teacher_replay", "admit_batch", "audit_admission", "save_probe_checkpoint", "audit_probe_checkpoint",
             "frozen_inventory"):
    function = inspect.unwrap(getattr(template, name))
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding preflight helper closure")
    code = function.__code__
    if name == "frozen_inventory":
        before = "Preserve the original full model,90 mutable parameters and164 frozen state tensors"
        if code.co_consts.count(before) != 1:
            raise ValueError("Adapt only the explicit frozen-inventory diagnostic, keeping every check")
        code = code.replace(co_consts=tuple("Preserve the original full model,238 mutable parameters and16 frozen buffers"
                                           if value == before else value for value in code.co_consts))
    cloned = types.FunctionType(code, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    "three-block point-protected preflight directories": "full-network point-protected preflight directories",
    'REPO / "tests/test_single_model_top_three_blocks_preflight.py"': 'REPO / "tests/test_single_model_full_network_preflight.py"',
    'REPO / "tests/test_single_model_top_three_block_training.py"': 'REPO / "tests/test_single_model_full_network_training.py"',
    '"optimizer_view_policy": top_three.POLICY': '"optimizer_view_policy": full_network.POLICY',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "top_three_blocks_preflight_v1"': 'OUTPUT = ROOT / "full_network_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_three_blocks_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/full_network_preflight_v1")',
    'DATA_SEED, GRADIENT_SEED = 771100901, 772100901': 'DATA_SEED, GRADIENT_SEED = 801100901, 802100901',
    'view = top_three.TopThreeBlockParameterView(model)': 'view = full_network.FullNetworkParameterView(model)',
    'with top_three.capture_coordinates(model, view):': 'with full_network.capture_coordinates(model, view):',
    'config["optimizer_view_policy"] != top_three.POLICY': 'config["optimizer_view_policy"] != full_network.POLICY',
    '# Three-block point-protected optimizer preflight': '# Full-network point-protected optimizer preflight',
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
