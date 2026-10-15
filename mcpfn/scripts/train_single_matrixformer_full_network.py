"""Train one ordinary V34 network with all original parameters and unchanged qualification."""

import hashlib
import types

import train_single_matrixformer_top_three_blocks as template
from train_single_matrixformer_top_three_blocks import (
    admission, attempts, baseline, compact_cache, composite, mixture, paired, rehearsal,
    replicated, storage, teacher_policy,
)
import single_model_full_network_training as full_network
import single_model_full_network_storage as diagnostics
from single_model_registered_adaptation import replace_assignment, replace_once


RATE, PREFIXES = template.RATE, full_network.PREFIXES
TRAINABLE_TENSORS, TRAINABLE_PARAMETERS, FROZEN_TENSORS = 238, 77998496, 16
PARENT_SHA256, PARENT_UPDATES = template.PARENT_SHA256, template.PARENT_UPDATES
STORAGE_ACCOUNTING = {
    "checkpoint": {"two_atomic_optimizer_checkpoints": 2 * (4 * TRAINABLE_PARAMETERS + 8 * TRAINABLE_PARAMETERS),
                   "selected_checkpoint": 4 * TRAINABLE_PARAMETERS, "metadata_and_audit_allowance": 64 * 1024**2},
    "root": {"logs_and_registration_allowance": 16 * 1024**2},
    "scratch": {"raw_gradient_bytes": 84 * 4 * TRAINABLE_PARAMETERS,
                "snapshot_vector_allowance": 8 * 4 * TRAINABLE_PARAMETERS,
                "ordinary_model_allowance": 2 * 4 * TRAINABLE_PARAMETERS,
                "prediction_task_allowance": 1024**3},
}
TRIAL_POLICY = ("Initialize only from the same audited unqualified V29 step25 with98 inherited updates, never "
    "V33's terminal state or a diagnostic checkpoint. Attempt64 fresh28-task synthetic mixed-prior batches with "
    "fresh5e-6 AdamW. Expand only optimizer scope to the original encoder, all eight blocks and decoder:238 "
    "parameter tensors/77,998,496 parameters, zero frozen parameters and16 bitwise-frozen original buffers. "
    "Inference remains one unchanged ordinary network without routing or ensemble. " + full_network.POLICY
    + " " + mixture.POLICY + " " + paired.POLICY + " " + composite.POLICY + " " + admission.POLICY + " " + attempts.POLICY
    + " Retain original general-imputation/specialist-rating teachers, native-anchor rehearsal, observed-only "
    "normalization, Huber/NLL/CDF/query/point objectives, .5/.25/.25 prior masses, half-composite/half-point "
    "proposal, all84 raw/proposal rows,30 point protections,0.1 margin,2x norm bound and exact rollback. Preserve "
    "every individual regression, all22 development and32 fresh confirmation comparisons, strict minimax "
    "selection, earlier ties and ineligible step0. Preserve18/6/50 GiB reserves with prospectively expanded "
    "2.25 GiB checkpoint/16 MiB root/32 GiB scratch training budgets and explicit full-coordinate atomic-write "
    "accounting. Keep original later-stage budgets and all ordinary checkpoints/real campaigns on checkpoint "
    "storage. Only declared synthetic diagnostics use UUID-bound machine-local scratch, not provider-persistent "
    "storage or backup. No scientific retry, source mutation, real-data selection, gate/reference change or "
    "benchmark-success claim from synthetic evidence.")
TRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION


for name in ("require_evidence", "constant_rate", "save_endpoint", "audit_endpoint", "save_last_endpoint",
             "replicated_batches", "validate_preflight"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding training helper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

BASE_PREFLIGHT_SOURCE = replace_once(template.BASE_PREFLIGHT_SOURCE,
    'result["backbone_parameters"] != 41961999', 'result["backbone_parameters"] != 0')
BASE_PREFLIGHT_SOURCE = replace_once(BASE_PREFLIGHT_SOURCE, "accepted three-block feasibility", "accepted full-network feasibility")
exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())


def require_capacity():
    for volume, amounts in STORAGE_ACCOUNTING.items():
        if sum(amounts.values()) > diagnostics.WRITE_BUDGETS["training"][volume]:
            raise ValueError("Full optimizer,84-row evidence and atomic replacements must fit registered budgets")
    diagnostics.storage.require_mount()
    diagnostics.require_capacity("training")


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_top_three_blocks as preflight': 'import preflight_single_model_full_network as preflight',
    'OUTPUT = ROOT / "top_three_blocks_v33"': 'OUTPUT = ROOT / "full_network_v34"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_three_blocks_v33")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/full_network_v34")',
    'DATA_SEED, GRADIENT_SEED = 781100901, 782100901': 'DATA_SEED, GRADIENT_SEED = 811100901, 812100901',
    'CONFIRMATION_SEEDS = (791100901, 792100901)': 'CONFIRMATION_SEEDS = (821100901, 822100901)',
    'view = top_three.TopThreeBlockParameterView(model)': 'view = full_network.FullNetworkParameterView(model)',
    'with top_three.capture_coordinates(model, view):': 'with full_network.capture_coordinates(model, view):',
    '"top_three_block_training": True': '"full_network_training": True',
    'protecting_top_three_block_update': 'protecting_full_network_update',
    'config["optimizer_view_policy"] != top_three.POLICY': 'config["optimizer_view_policy"] != full_network.POLICY',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'REPO / "scripts/audit_single_model_top_three_blocks.py"': 'REPO / "scripts/audit_single_model_full_network.py"',
    'REPO / "scripts/run_single_model_top_three_blocks_pipeline.py"': 'REPO / "scripts/run_single_model_full_network_pipeline.py"',
    'REPO / "tests/test_single_model_top_three_blocks_training.py"': 'REPO / "tests/test_single_model_full_network_trial.py"',
    'REPO / "tests/test_single_model_top_three_blocks_audit.py"': 'REPO / "tests/test_single_model_full_network_audit.py"',
    'REPO / "tests/test_single_model_top_three_blocks_pipeline.py"': 'REPO / "tests/test_single_model_full_network_pipeline.py"',
    'REPO / "tests/test_single_model_scratch_diagnostics.py"': 'REPO / "tests/test_single_model_full_network_storage.py"',
    '"optimizer_view_policy": top_three.POLICY': '"optimizer_view_policy": full_network.POLICY',
}.items():
    REGISTER_SOURCE = replace_once(REGISTER_SOURCE, old, new)
exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart a begun full-network continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
