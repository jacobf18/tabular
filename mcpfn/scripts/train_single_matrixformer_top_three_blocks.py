"""Train one ordinary V33 network with three mutable final blocks and fixed qualification."""

import hashlib
import inspect
import types

import train_single_matrixformer_point_only as template
import train_single_matrixformer_top_two_blocks as foundation
from train_single_matrixformer_point_only import (
    admission, attempts, baseline, compact_cache, composite, diagnostics, mixture, paired, rehearsal,
    replicated, storage, teacher_policy,
)
import single_model_top_three_block_training as top_three
from single_model_registered_adaptation import replace_assignment, replace_once


RATE = template.RATE
PREFIXES = ("transformer_blocks.5.", "transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
TRAINABLE_TENSORS, TRAINABLE_PARAMETERS, FROZEN_TENSORS = 90, 36036497, 164
PARENT_SHA256, PARENT_UPDATES = template.PARENT_SHA256, template.PARENT_UPDATES
STORAGE_ACCOUNTING = {
    "checkpoint": {"two_atomic_optimizer_checkpoints": 2 * (4 * 77998496 + 8 * TRAINABLE_PARAMETERS),
                   "selected_checkpoint": 4 * 77998496, "metadata_and_audit_allowance": 64 * 1024**2},
    "root": {"logs_and_registration_allowance": 16 * 1024**2},
    "scratch": {"raw_gradient_bytes": 84 * 4 * TRAINABLE_PARAMETERS,
                "snapshot_vector_allowance": 8 * 4 * TRAINABLE_PARAMETERS,
                "ordinary_model_allowance": 2 * 4 * 77998496,
                "prediction_task_allowance": 1024**3},
}
TRIAL_POLICY = ("Initialize only from the same audited unqualified V29 step25 with98 inherited updates, never "
    "V32 or a diagnostic checkpoint. Attempt64 fresh28-task synthetic mixed-prior batches with fresh5e-6 AdamW. "
    "Expand training depth to existing blocks5/6/7 and decoder:90 tensors/36,036,497 parameters, preserving164 "
    "frozen state tensors including all selected-prefix RoPE buffers. Inference remains one ordinary77,998,496-"
    "parameter network, without ensemble or routing. " + top_three.POLICY + " " + mixture.POLICY + " "
    + paired.POLICY + " " + composite.POLICY + " " + admission.POLICY + " " + attempts.POLICY
    + " Preserve original general-imputation/specialist-rating teachers, native-anchor rehearsal, observed-only "
    "normalization, Huber/NLL/CDF/query/point objectives, .5/.25/.25 prior masses, full84-row proposal and30 "
    "point protections,0.1 margin,2x norm bound and exact rollback. Preserve every individual regression,22 "
    "development and32 fresh confirmation comparisons, strict minimax selection and ineligible step0. "
    "Preserve18/6/50 GiB reserves and original per-volume write budgets with expanded explicit accounting, "
    "including atomic optimizer checkpoint replacement. Ordinary checkpoints and frozen real campaigns "
    "remain on checkpoint storage; only declared synthetic diagnostics use UUID-bound machine-local scratch, "
    "not provider-persistent storage or backup. No scientific retry, real-data selection, source mutation, "
    "changed gate/reference or claim of success from synthetic evidence.")
TRIAL_QUALIFICATION = template.TRIAL_QUALIFICATION


for name in ("require_evidence", "constant_rate", "save_endpoint", "audit_endpoint", "save_last_endpoint",
             "replicated_batches", "validate_preflight"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding training helper closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


BASE_PREFLIGHT_SOURCE = inspect.getsource(foundation.validate_preflight)
for old, new in {
    "def validate_preflight(": "def base_validate_preflight(",
    'result["trainable_parameters"] != 27766670': 'result["trainable_parameters"] != TRAINABLE_PARAMETERS',
    'result["backbone_parameters"] != 50231826': 'result["backbone_parameters"] != 41961999',
    'result["frozen"]["frozen_tensors"] != 192': 'result["frozen"]["frozen_tensors"] != FROZEN_TENSORS',
    'len(result["frozen"]["changed_trainable_tensors"]) != 62':
        'len(result["frozen"]["changed_trainable_tensors"]) != TRAINABLE_TENSORS',
    "accepted two-block feasibility": "accepted three-block feasibility",
}.items():
    BASE_PREFLIGHT_SOURCE = replace_once(BASE_PREFLIGHT_SOURCE, old, new)
exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())


def require_capacity():
    for volume, amounts in STORAGE_ACCOUNTING.items():
        if sum(amounts.values()) > diagnostics.WRITE_BUDGETS["training"][volume]:
            raise ValueError("Expanded optimizer, full-row evidence and atomic replacements must fit original budgets")
    diagnostics.storage.require_mount()
    diagnostics.require_capacity("training")


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_point_only as preflight': 'import preflight_single_model_top_three_blocks as preflight',
    'OUTPUT = ROOT / "point_only_v32"': 'OUTPUT = ROOT / "top_three_blocks_v33"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/point_only_v32")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/top_three_blocks_v33")',
    'DATA_SEED, GRADIENT_SEED = 751100901, 752100901': 'DATA_SEED, GRADIENT_SEED = 781100901, 782100901',
    'CONFIRMATION_SEEDS = (761100901, 762100901)': 'CONFIRMATION_SEEDS = (791100901, 792100901)',
    'view = top_two.TopTwoBlockParameterView(model)': 'view = top_three.TopThreeBlockParameterView(model)',
    'with top_two.capture_coordinates(model, view):': 'with top_three.capture_coordinates(model, view):',
    '"top_two_block_training": True': '"top_three_block_training": True',
    'protecting_top_two_block_update': 'protecting_top_three_block_update',
    'or config["policy"] != POLICY':
        'or config["optimizer_view_policy"] != top_three.POLICY or config["storage_accounting"] != STORAGE_ACCOUNTING\n'
        '            or config["policy"] != POLICY',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'REPO / "scripts/audit_single_model_point_only.py"': 'REPO / "scripts/audit_single_model_top_three_blocks.py"',
    'REPO / "scripts/run_single_model_point_only_pipeline.py"': 'REPO / "scripts/run_single_model_top_three_blocks_pipeline.py"',
    'REPO / "tests/test_single_model_point_only_training.py"': 'REPO / "tests/test_single_model_top_three_blocks_training.py"',
    'REPO / "tests/test_single_model_point_only_audit.py"': 'REPO / "tests/test_single_model_top_three_blocks_audit.py"',
    'REPO / "tests/test_single_model_point_only_pipeline.py"': 'REPO / "tests/test_single_model_top_three_blocks_pipeline.py"',
    'config = {"policy": POLICY,':
        'config = {"policy": POLICY, "optimizer_view_policy": top_three.POLICY, "storage_accounting": STORAGE_ACCOUNTING,',
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
                raise ValueError("Never restart a begun three-block continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
