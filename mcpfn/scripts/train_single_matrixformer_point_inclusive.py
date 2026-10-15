"""Train a fresh V8-start single model with point-inclusive half-rate paired updates."""

import hashlib

import train_single_matrixformer_structured_teacher as baseline
import compare_single_model_point_proposals as matched
import single_model_paired_protection as paired
import single_model_point_inclusive_proposal as inclusive
import single_model_replicated_decoder_batch as replicated
import single_model_shared_tensor_storage as shared_storage
import single_model_structured_teacher as teacher_policy
from preflight_single_model_point_inclusive import shared_gradients, RAW_GRADIENT_STORAGE
from single_model_registered_adaptation import replace_once, replace_assignment


previous = baseline.previous
MATCHED_AUDIT_SHA256 = "c62bfd895655ff17d108be1bb76bfa47a2eb3a3d4d988c84b76d58c28e6f9f84"
MATCHED_RESULT_SHA256 = "885dbc5169266de868cb53e150b0358312be9d593b2bf5e22aab4eefb846b224"
TRIAL_POLICY = ("Initialize a single ordinary network directly from original V8, never a diagnostic candidate or V20. "
                "Retain V20's64 attempts,28 paired tasks, .25 legacy/.25 structured/.5 ratings mixture, all priors, "
                "shapes, components, regimes, MAR/skip emphasis, observed-only normalization, frozen248 backbone "
                "state tensors and original V8/general/specialist teacher routing. Change the decoder Adam proposal "
                "to half original Huber/NLL/CDF/query composite and half defined ground-truth point gradients at "
                "constant1e-5. Retain paired protection,0.1 margin,2x norm cap, finite acceptance and exact rollback. "
                "Initial and rollback rate is1e-5, unlike the2e-5 preceding rate of the one-step matched diagnostic; "
                "the actual tested proposal rate was1e-5. Require its audited accepted point_1e-5 control, not the "
                "rejected original2e-5 preflight. Both half-rate controls improved28/28 training metrics; the "
                "point-inclusive control was better on15 and worse on13, not uniformly dominant. Preserve all raw "
                "gradients through lossless shared storage. Fresh streams, no inference ensemble, architecture "
                "change, held-out gate relaxation or matched-seed full-training causal claim. All real comparisons "
                "remain frozen and synthetic training improvements alone cannot establish goal success.")


def constant_rate(step):
    baseline.constant_rate(step)
    return 1e-5


def replicated_batches():
    return [replicated.replicate_batch(batch, DATA_SEED, GRADIENT_SEED) for batch in previous.batches()]


def matched_acceptance(audit):
    rows = audit.get("rows", [])
    if (not audit.get("complete") or audit.get("matched_cells") != 4 or audit.get("new_controls") != 3
            or audit.get("prediction_grids") != 224 or audit.get("selection_performed") is not False
            or audit.get("cuda_initialized") is not False or len(rows) != 4
            or [{key: row.get(key) for key in ("name", "method", "rate", "reused")} for row in rows] != matched.registered_cases()):
        raise ValueError("Require the complete independent224-grid matched diagnostic, without selection")
    selected = next(row for row in rows if row["name"] == "point_1e-5")
    if (not selected["accepted"] or selected["reason"] != "accepted"
            or selected["paired_improved"] != 28 or selected["paired_worsened"] != 0):
        raise ValueError("Require the accepted half-rate point-inclusive control, never the rejected original preflight")
    return selected


def require_matched():
    for name, expected in (("audit.json", MATCHED_AUDIT_SHA256), ("result.json", MATCHED_RESULT_SHA256)):
        if sha256(matched.OUTPUT / name) != expected:
            raise ValueError("Preserve the original matched feasibility evidence")
    config = matched.verify()
    terminal = read(matched.OUTPUT / "status.json")
    result = read(matched.OUTPUT / "result.json")
    if (terminal["state"] != "complete" or Path(f'/proc/{terminal["pid"]}').exists()
            or not result["complete"] or Path(f'/proc/{result["run_pid"]}').exists()
            or config["parent"] != str(PARENT)):
        raise ValueError("Require separate completed matched controls from the original V8 parent")
    audit = read(matched.OUTPUT / "audit.json")
    matched_acceptance(audit)
    verify_hashes(audit["artifact_sha256"])
    return {**config["input_sha256"], **audit["artifact_sha256"],
            **{str(matched.OUTPUT / name): sha256(matched.OUTPUT / name)
               for name in ("config.json", "registration.json", "status.json", "audit.json", "result.json")}}


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
start = ADAPTED_SOURCE.index("    preflight.verify(preflight.OUTPUT)", ADAPTED_SOURCE.index("def register("))
end = ADAPTED_SOURCE.index("    previous = sorted(", start)
ADAPTED_SOURCE = ADAPTED_SOURCE[:start] + "    feasibility_hashes = require_matched()\n" + ADAPTED_SOURCE[end:]
for old, new in {
    'OUTPUT = ROOT / "structured_teacher_v20"': 'OUTPUT = ROOT / "point_inclusive_v21"',
    'DATA_SEED, GRADIENT_SEED = 381100901, 382100901': 'DATA_SEED, GRADIENT_SEED = 411100901, 412100901',
    'CONFIRMATION_SEEDS = (391100901, 392100901)': 'CONFIRMATION_SEEDS = (421100901, 422100901)',
    'config["learning_rate_schedule"] != {"constant": 2e-5}': 'config["learning_rate_schedule"] != {"constant": 1e-5}',
    'config["protection_source_sha256"] != paired.PROTECTED_SOURCE_SHA256':
        'config["protection_source_sha256"] != inclusive.PROTECTED_SOURCE_SHA256\n'
        '            or config["proposal_policy"] != inclusive.POLICY\n'
        '            or config["proposal_audit_source_sha256"] != inclusive.AUDIT_SOURCE_SHA256\n'
        '            or config["raw_gradient_storage"] != RAW_GRADIENT_STORAGE\n'
        '            or config["feasibility_audit_sha256"] != MATCHED_AUDIT_SHA256',
    'REPO / "tests/test_single_model_structured_teacher_training.py"': 'REPO / "tests/test_single_model_point_inclusive_training.py"',
    '"learning_rate_schedule": {"constant": 2e-5}': '"learning_rate_schedule": {"constant": 1e-5}',
    '"protection_source_sha256": paired.PROTECTED_SOURCE_SHA256,':
        '"protection_source_sha256": inclusive.PROTECTED_SOURCE_SHA256,\n'
        '              "proposal_policy": inclusive.POLICY, "proposal_audit_source_sha256": inclusive.AUDIT_SOURCE_SHA256,\n'
        '              "raw_gradient_storage": RAW_GRADIENT_STORAGE, "feasibility_audit_sha256": MATCHED_AUDIT_SHA256,',
    '"input_sha256": {**base["input_sha256"],': '"input_sha256": {**base["input_sha256"], **feasibility_hashes,',
    'record, snapshot = paired.protected_update(': 'record, snapshot = inclusive.protected_update(',
    'original.save_tensor({"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)].cpu(),':
        'shared_storage.save({"snapshot": error.snapshot, "record": error.record, "gradients": shared_gradients(gradients[:len(entries)], error.snapshot),',
    'payload = {"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)].cpu(),':
        'payload = {"snapshot": snapshot, "record": record, "gradients": shared_gradients(gradients[:len(entries)], snapshot),',
    'record["raw_endpoint_sha256"] = original.save_tensor(payload, output / "raw_final_update.pt")':
        'record["raw_endpoint_sha256"] = shared_storage.save(payload, output / "raw_final_update.pt")',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", TRIAL_POLICY)
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
                raise ValueError("Never restart the registered point-inclusive trial")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
