"""Continue the audited V20 student on fresh mixed-prior tasks without changing its teachers."""

import hashlib

import train_single_matrixformer_structured_teacher as baseline
import recover_single_model_point_teacher_entry as recovery
import run_single_model_structured_teacher_pipeline as parent_pipeline
import single_model_paired_protection as paired
import single_model_replicated_decoder_batch as replicated
import single_model_shared_tensor_storage as shared_storage
import single_model_structured_teacher as teacher_policy
from preflight_single_model_point_inclusive import shared_gradients, RAW_GRADIENT_STORAGE
from single_model_registered_adaptation import replace_once, replace_assignment


previous = baseline.previous
constant_rate = baseline.constant_rate
PARENT_SHA256 = "2080659b45c5506af16516c05578c96726348c46cb032c5de3f02481eb15ff4c"
TRIAL_POLICY = ("Initialize one ordinary network from the audited V20 selected boundary64, which remains unqualified "
                "after failed independent confirmation. Continue for64 fresh28-task decoder-only attempts with "
                "fresh AdamW moments, constant2e-5 composite proposals, paired0.1 protection,2x norm cap and exact "
                "finite rollback. Preserve V20 priors, shapes, components, regimes, MAR/skip emphasis, .25 legacy/"
                ".25 structured/.5 ratings mass, Huber/NLL/CDF/query objectives and observed-only normalization. "
                "Original V8 remains the legacy teacher and retention/confirmation baseline; original general "
                "remains the structured teacher and original specialist the ratings teacher. The V20 student is "
                "not a teacher or a new retention baseline. Freeze its248 nondecoder state tensors. Preserve all "
                "raw gradients using exact shared storage. This tests more training from the stronger synthetic "
                "development state, not the unselected V21 tail. No new one-step feasibility diagnostic is used; "
                "the audited V20 algorithm and every fresh attempt's finite acceptance remain mandatory. Use fresh "
                "training/backward/confirmation seeds. No inference ensemble, benchmark-based selection, "
                "qualification relaxation or claim that V20's development selection is independent qualification.")
TRIAL_QUALIFICATION = (baseline.QUALIFICATION + " V22 initializes from V20, but all original V8 retention and "
                       "confirmation baselines remain unchanged. Independently audit an additional152 V20-parent "
                       "development grids and verify initialization parity, yielding1368 full scheduled/reference "
                       "grids. Step zero is never eligible, even when the unqualified parent passes development.")


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in previous.batches()]


def validate_history(parent, finished, endpoint, checkpoint_digest):
    if (not parent.get("complete") or parent.get("gate", {}).get("passed") is not False
            or parent.get("gate", {}).get("confirmation_evaluated") is not True
            or parent.get("campaign") is not None or parent.get("review") is not None
            or not finished.get("complete") or finished.get("gate", {}).get("passed") is not False
            or finished.get("gate", {}).get("selected_step") != 0
            or finished.get("gate", {}).get("confirmation_evaluated") is not False
            or finished.get("campaign") is not None or finished.get("review") is not None
            or not endpoint.get("complete") or endpoint.get("endpoint_grids") != 56
            or checkpoint_digest != PARENT_SHA256):
        raise ValueError("Require audited unqualified V20 and completed unchanged V21 nonselection")


def require_history():
    for output in (parent_pipeline.OUTPUT, recovery.OUTPUT):
        terminal = read(output / "status.json")
        if terminal["state"] != "complete" or Path(f'/proc/{terminal["pid"]}').exists():
            raise ValueError("Complete both original qualification lifecycles before registering a continuation")
    recovery.gateway.verify_recovery()
    recovery.pipeline.prerequisite(recovery.OUTPUT)
    parent_audit = parent_pipeline.prerequisite(parent_pipeline.OUTPUT)
    endpoint = read(recovery.pipeline.REPLAY / "audit.json")
    validate_history(read(parent_pipeline.OUTPUT / "result.json"), read(recovery.OUTPUT / "result.json"),
                     endpoint, sha256(PARENT))
    if (parent_audit["best"]["step"] != 64 or parent_audit["accepted_updates"] != 56
            or parent_audit["artifact_sha256"].get(str(PARENT)) != PARENT_SHA256):
        raise ValueError("Bind the actual audited56-update V20 selected checkpoint")
    verify_hashes(endpoint["artifact_sha256"])
    for folder, field, expected in ((parent_pipeline.REPLAY, "endpoint_grids", 56),
                                     (parent_pipeline.CONFIRMATION, "prediction_grids", 1136)):
        report = read(folder / "audit.json")
        if not report["complete"] or report[field] != expected:
            raise ValueError("Preserve V20's independently audited endpoint and complete failed confirmation")
        verify_hashes(report["artifact_sha256"])
    for folder in (parent_pipeline.training.OUTPUT, recovery.pipeline.training.OUTPUT):
        if (folder / "best_validation_checkpoint.pth").exists():
            raise ValueError("Neither preceding candidate is an independently qualified replacement")
    paths = [PARENT, parent_pipeline.AUDIT / "audit.json", parent_pipeline.TEACHERS / "audit.json",
             parent_pipeline.REPLAY / "audit.json", parent_pipeline.CONFIRMATION / "audit.json",
             parent_pipeline.CONFIRMATION / "gate.json", recovery.pipeline.AUDIT / "audit.json",
             recovery.pipeline.TEACHERS / "audit.json", recovery.pipeline.REPLAY / "audit.json",
             recovery.pipeline.CONFIRMATION / "gate.json", recovery.OUTPUT / "recovery_audit.json",
             recovery.OUTPUT / "recovery_config.json", recovery.OUTPUT / "recovery_registration.json"]
    paths.extend(output / name for output in (parent_pipeline.OUTPUT, recovery.OUTPUT)
                 for name in ("config.json", "registration.json", "status.json", "result.json"))
    return {str(path): sha256(path) for path in paths}


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
start = ADAPTED_SOURCE.index("    preflight.verify(preflight.OUTPUT)", ADAPTED_SOURCE.index("def register("))
end = ADAPTED_SOURCE.index("    previous = sorted(", start)
ADAPTED_SOURCE = ADAPTED_SOURCE[:start] + "    history_hashes = require_history()\n" + ADAPTED_SOURCE[end:]
for old, new in {
    'OUTPUT = ROOT / "structured_teacher_v20"': 'OUTPUT = ROOT / "teacher_continuation_v22"',
    'INITIAL_RUN = original.alignment.fixed.PRODUCER': 'INITIAL_RUN = ROOT / "structured_teacher_v20"',
    'PARENT = preflight.PARENT': 'PARENT = INITIAL_RUN / "selected_checkpoint.pth"',
    'DATA_SEED, GRADIENT_SEED = 381100901, 382100901': 'DATA_SEED, GRADIENT_SEED = 441100901, 442100901',
    'CONFIRMATION_SEEDS = (391100901, 392100901)': 'CONFIRMATION_SEEDS = (451100901, 452100901)',
    'teacher_policy.identities(config["base"], PARENT)':
        'teacher_policy.identities(config["base"], original.alignment.fixed.PARENT)\n'
        '            or config["parent_selection_qualified"] is not False\n'
        '            or config["raw_gradient_storage"] != RAW_GRADIENT_STORAGE',
    'teacher_policy.identities(base, PARENT)': 'teacher_policy.identities(base, original.alignment.fixed.PARENT)',
    'paths = [PARENT, INITIAL_RUN / "validation_0.json",': 'paths = [PARENT, INITIAL_RUN / "validation_64.json",',
    'REPO / "tests/test_single_model_structured_teacher_training.py"':
        'REPO / "tests/test_single_model_teacher_continuation.py"',
    '"parent_selection_qualified": True': '"parent_selection_qualified": False, "raw_gradient_storage": RAW_GRADIENT_STORAGE',
    '"input_sha256": {**base["input_sha256"],': '"input_sha256": {**base["input_sha256"], **history_hashes,',
    'never promotion of the unchanged V8 training input': 'never promotion of the unqualified V20 training input',
    '"parent_training_updates": 0': '"parent_training_updates": 56',
    'read(INITIAL_RUN / "validation_0.json")["student"]': 'read(INITIAL_RUN / "validation_64.json")["student"]',
    'original.save_tensor({"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)].cpu(),':
        'shared_storage.save({"snapshot": error.snapshot, "record": error.record, "gradients": shared_gradients(gradients[:len(entries)], error.snapshot),',
    'payload = {"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)].cpu(),':
        'payload = {"snapshot": snapshot, "record": record, "gradients": shared_gradients(gradients[:len(entries)], snapshot),',
    'record["raw_endpoint_sha256"] = original.save_tensor(payload, output / "raw_final_update.pt")':
        'record["raw_endpoint_sha256"] = shared_storage.save(payload, output / "raw_final_update.pt")',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


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
                raise ValueError("Never restart the registered continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
