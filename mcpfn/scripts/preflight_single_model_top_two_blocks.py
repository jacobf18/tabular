"""Test one mixed-prior update of two final blocks after V25's rejected confirmation."""

import hashlib
import inspect

import torch

import preflight_single_model_top_block as template
import run_single_model_top_block_continuation_pipeline as preceding
import single_model_top_two_block_training as top_two
from preflight_single_model_top_block import composite, paired, replicated, storage, teacher_policy, schedule
from single_model_registered_adaptation import replace_once, replace_assignment


PREFIXES = ("transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
POLICY_TEXT = ("One fresh28-task optimizer feasibility test of the original final two transformer blocks plus "
    "decoder, after V25's complete unsuccessful independent qualification. Initialize from original qualified V8, "
    "not V25 or a diagnostic endpoint. Preserve the ordinary77,998,496-parameter network, train only62 original "
    "tensors/27,766,670 parameters and preserve192 other state tensors bitwise. Keep the existing first-batch "
    "shapes, .25 legacy/.25 structured/.5 ratings mass, MAR/skip emphasis, observed-only normalization and original "
    "Huber/NLL/CDF/query objectives. Keep original V8/general/specialist teachers, constant1e-5 proposal, paired0.1 "
    "protection,2x norm cap and exact parameter/moment/rate rollback. Save all84 raw protected gradients and the "
    "unchanged28-row composite prefix alias. Independently replay28 complete teacher grids and CPU-audit the "
    "optimizer endpoint plus56 before/candidate grids. The diagnostic checkpoint cannot initialize a future "
    "model. No validation, confirmation, model nomination, real benchmark, inference ensemble, guard relaxation "
    "or general memory-feasibility claim beyond this batch. Use fresh reserved training/backward streams only. "
    "Ordinary checkpoint and canonical evidence stay on checkpoint storage; only bulky raw endpoint diagnostics "
    "use machine-local scratch, not provider-persistent storage or external backup. Preserve18/6/50 GiB "
    "checkpoint/root/scratch reserves and1/0/16 GiB corresponding stage write budgets. No changes to V25 or any "
    "completed registered source, automatic retry or reuse of consumed confirmation.")


for name in ("require_space", "require_predecessor", "register_trial", "frozen_inventory", "save_raw", "audit_raw"):
    source = inspect.getsource(getattr(template, name))
    if name == "require_space":
        source = replace_once(source, "(SCRATCH.parent, 50, 12)", "(SCRATCH.parent, 50, 16)")
    elif name == "require_predecessor":
        source = replace_once(source, "terminal V23", "terminal V25")
    elif name == "register_trial":
        source = replace_once(source, "test_single_model_top_block_preflight.py", "test_single_model_top_two_blocks_preflight.py")
        source = replace_once(source, "test_single_model_top_block_training.py", "test_single_model_top_two_block_training.py")
        source = replace_once(source, "new declared top-block preflight", "new declared two-block preflight")
    elif name == "frozen_inventory":
        source = replace_once(source, "len(expected) != 34", "len(expected) != 62")
        source = replace_once(source, "and34 last-block/decoder tensors", "and62 final-two-block/decoder tensors")
    exec(compile(source, __file__, "exec"), globals())

for name in ("expected_role", "check_routing", "replay_targets", "audit_teacher_replay"):
    exec(compile(inspect.getsource(getattr(template.template, name)), __file__, "exec"), globals())

teacher_predictions = template.teacher_predictions
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "top_block_preflight_v1"': 'OUTPUT = ROOT / "top_two_blocks_preflight_v1"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_block_preflight_v1")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/top_two_blocks_preflight_v1")',
    'RATE = 2e-5': 'RATE = 1e-5',
    'DATA_SEED, GRADIENT_SEED = 481100901, 482100901': 'DATA_SEED, GRADIENT_SEED = 551100901, 552100901',
    'view = top_block.TopBlockParameterView(model)': 'view = top_two.TopTwoBlockParameterView(model)',
    'with top_block.capture_coordinates(model, view):': 'with top_two.capture_coordinates(model, view):',
    '# Last-block plus decoder optimizer preflight': '# Final-two-block plus decoder optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


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
