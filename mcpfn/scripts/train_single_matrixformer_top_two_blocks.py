"""Train V26 from V8 with two final blocks and decoder on fresh mixed-prior tasks."""

import hashlib
import inspect

import train_single_matrixformer_top_block as template
import single_model_top_two_block_training as top_two
import single_model_scratch_diagnostics as diagnostics
from train_single_matrixformer_top_block import baseline, compact_cache, composite, paired, replicated, storage, teacher_policy
from single_model_registered_adaptation import replace_once, replace_assignment


RATE = 1e-5
PREFIXES = ("transformer_blocks.6.", "transformer_blocks.7.", "decoder.")
TRIAL_POLICY = ("Initialize one ordinary network from original qualified V8 with fresh AdamW moments, never V25 "
    "or a diagnostic endpoint. Attempt64 fresh28-task batches, training only the existing final two transformer "
    "blocks and decoder:62 tensors/27,766,670 parameters; preserve192 other state tensors bitwise. Keep the "
    "unchanged77,998,496-parameter inference network, .25 legacy/.25 structured/.5 ratings mass, MAR/skip emphasis, "
    "shape/component/regime schedule, observed-only normalization and Huber/NLL/CDF/query objectives. Original "
    "V8 teaches legacy, original general teaches structured imputation and original specialist teaches ratings. "
    "Use constant1e-5 after the accepted independently audited two-block feasibility test, paired0.1 protection, "
    "2x norm cap and exact parameter/moment/rate rollback. Preserve every individual-draw regression. Store "
    "compact exact complete task/teacher digests and rating vectors, all84 terminal gradients and identical28-row "
    "proposal alias. Only synthetic diagnostic development/replay/confirmation and bulky raw endpoint evidence "
    "use registered UUID-bound local scratch, not provider-persistent storage or external backup. Ordinary "
    "selected/latest/published checkpoints and every real benchmark stay on checkpoint storage. Preserve18/6/50 "
    "GiB reserves and registered per-volume write budgets. No inference ensemble, new architecture, teacher or "
    "baseline replacement, real-data selection, retry or changes to original scientific gates.")
TRIAL_QUALIFICATION = ("Independently regenerate/digest-check all1792 complete task/teacher payloads and128 rating "
    "contexts, audit all64 attempts and1216 original reference/development grids,62 optimizer coordinates,192 "
    "frozen state tensors and complete terminal optimizer/84-gradient evidence. Do not claim historical full "
    "autograd/optimizer replay. Select only at0/10/25/50/64 under all22 original V8 guards and strict minimax "
    "improvement; unchanged step0 is ineligible and ties retain the earlier trained boundary. Independently "
    "reconstruct/replay the endpoint and sole selected152-case development state. Confirm once on284 fresh "
    "reserved cases under unchanged V8/general gates and audit all1136 grids. Only then publish a best alias and "
    "run every frozen tabular, four-dataset recommender and LLM comparison/review. No reselecting after failure, "
    "hidden required regression or claim of goal success from synthetic evidence.")


def require_capacity():
    diagnostics.storage.require_mount()
    diagnostics.require_capacity("training")


def validate_preflight(result, audit, gate, current, digest):
    if (current["state"] != "complete" or current.get("accepted") is not True
            or gate["passed"] is not True or gate["preflight_only"] is not True or gate["audit_sha256"] != digest
            or audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["teacher_replay"]["complete"] is not True or audit["teacher_replay"]["full_grids"] != 28
            or audit["teacher_replay"]["bitwise_equal"] is not True
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["raw_endpoint"]["accepted_updates"] != 1
            or audit["raw_endpoint"]["accepted"] is not True
            or audit["autograd_replayed"] is not False or audit["cuda_initialized"] is not False
            or result["preflight_only"] is not True or result["record"]["accepted"] is not True
            or result["record"]["step"] != 1 or result["record"]["rate"] != RATE
            or result["record"]["finite"]["all_retained"] is not True or result["record"]["finite"]["accepted"] is not True
            or result["trainable_parameters"] != 27766670 or result["backbone_parameters"] != 50231826
            or result["frozen"]["frozen_tensors"] != 192 or len(result["frozen"]["changed_trainable_tensors"]) != 62
            or result["frozen"]["unselected_parameters_and_all_buffers_bitwise_unchanged"] is not True
            or audit["frozen"] != result["frozen"]):
        raise ValueError("Require the complete independently audited accepted two-block feasibility test")


def require_evidence():
    preflight.verify(preflight.OUTPUT)
    preflight.require_predecessor()
    result, audit, gate, current = [read(preflight.OUTPUT / name)
                                  for name in ("result.json", "audit.json", "gate.json", "status.json")]
    validate_preflight(result, audit, gate, current, sha256(preflight.OUTPUT / "audit.json"))
    if Path(f'/proc/{current["pid"]}').exists():
        raise ValueError("Finish and stop the original preflight before registering training")
    verify_hashes(audit["artifact_sha256"])
    paths = [preflight.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "audit.json", "gate.json", "status.json")]
    return {str(path): sha256(path) for path in paths}


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_top_block as preflight': 'import preflight_single_model_top_two_blocks as preflight',
    'OUTPUT = ROOT / "top_block_v24"': 'OUTPUT = ROOT / "top_two_blocks_v26"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/top_block_v24_raw")':
        'SCRATCH = Path("/mnt/matrixformer_scratch/top_two_blocks_v26")',
    'DATA_SEED, GRADIENT_SEED = 491100901, 492100901': 'DATA_SEED, GRADIENT_SEED = 561100901, 562100901',
    'CONFIRMATION_SEEDS = (501100901, 502100901)': 'CONFIRMATION_SEEDS = (571100901, 572100901)',
    'view = top_block.TopBlockParameterView(model)': 'view = top_two.TopTwoBlockParameterView(model)',
    'with top_block.capture_coordinates(model, view):': 'with top_two.capture_coordinates(model, view):',
    '"top_block_training": True': '"top_block_training": True, "top_two_block_training": True',
    'teacher_policy.identities(config["base"], PARENT)':
        'teacher_policy.identities(config["base"], PARENT)\n'
        '            or config["parent_selection_qualified"] is not True\n'
        '            or config["diagnostic_storage"] != diagnostics.layout(OUTPUT, SCRATCH, BOUNDARIES)',
    '    return config\n': '    diagnostics.verify(config["diagnostic_storage"], "training")\n    return config\n',
    '    directory.mkdir()\n': '    diagnostics.require_empty_validation(config["diagnostic_storage"], directory)\n',
    '"protecting_top_block_update"': '"protecting_top_two_block_update"',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("constant_rate", "replicated_batches", "save_endpoint", "audit_endpoint"):
    exec(compile(inspect.getsource(getattr(template, name)), __file__, "exec"), globals())

REGISTER_SOURCE = inspect.getsource(template.register_trial)
for old, new in {
    'REPO / "scripts/audit_single_model_top_block_training.py"':
        'REPO / "scripts/audit_single_model_top_two_blocks.py",\n'
        '               REPO / "tests/test_single_model_scratch_diagnostics.py"',
    'REPO / "tests/test_single_model_top_block_trial.py"': 'REPO / "tests/test_single_model_top_two_blocks_trial.py"',
    '"parent_selection_qualified": True, "retention_baseline": str(PARENT),':
        '"parent_selection_qualified": True, "retention_baseline": str(PARENT),\n'
        '              "diagnostic_storage": diagnostics.layout(OUTPUT, SCRATCH, BOUNDARIES),',
    '    output.mkdir()\n': '    output.mkdir()\n    diagnostics.prepare(config["diagnostic_storage"], "training")\n',
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
                raise ValueError("Never restart a begun two-block trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
