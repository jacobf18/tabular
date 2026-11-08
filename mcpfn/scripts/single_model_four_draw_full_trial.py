"""Bind V57 to complete feasibility, capacity and unchanged full-model qualification."""

from pathlib import Path

import preflight_single_model_four_draw as probe
import recover_single_model_four_draw_mean_audit as recovery
import single_model_balanced_native_long_trial as template
import single_model_four_draw_trial_budget as budget
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256


ARITHMETIC = template.ARITHMETIC
CAPACITY_RECEIPT_SHA256 = "26e5449dbf823059a5331197fbb0b586fa1df331ca2f01bac2a15f25ec538f24"
FEASIBILITY_SHA256 = "a6923bb5cd25758545fbf3124a9f3ccd4792c9b13c133fc3fec305a715e3a6c0"
MEAN_SHA256 = "bcc8fbc602a28391fec084ffe7e800501ecd32b98def457470b26ebb1947072a"
POLICY = (
    "One separately registered V57 trial, initialized only from actual V29 step25/98 inherited "
    "updates, never a diagnostic or closed-trial endpoint. Pair the original256 fresh balanced "
    "metadata batches into128 four-draw56-task attempts, retaining7168 scheduled tasks, every "
    "requested shape/orientation/component and half of each original supervised/effective "
    "retention weight. Preserve.5/.25/.25 native/structured/rating masses, the original individual "
    "standardized teacher Huber20/delta1/NLL/CDF/query/point losses, native V29 and FP32 half "
    "V8/V29 structured retention, no rating retention and original observed-only normalization. "
    "Frozen observed-blend imputation teachers and the specialist rating teacher are training-only; "
    "inference remains one ordinary77,998,496-parameter network,238 trainable tensors and16 frozen "
    "buffers. Preserve each defined raw gradient, up to168 vectors, via complete FP32 CPU capture "
    "and byte-checked CUDA staging. Release CPU capture before unchanged CUDA AdamW/quarter "
    "projection,.1 inward margin,2x norm cap and exact rollback. Keep four independent native "
    "draw guards and structured/rating four-draw means,34 grouped slots, all individual adverse "
    "metrics and truth-defined nulls. The accepted feasibility update had seven adverse individual "
    "point metrics, one undefined NRMSE and16 adverse mean terms; feasibility does not establish "
    "generalization. Use fresh5e-6 AdamW and consume every rejection without replacement or "
    "adaptive stopping. Preserve eight boundaries0/5/13/25/32/64/96/128 (source25 explicitly rounds "
    "up to26), all22 original development guards, strict minimax/earlier ties/ineligible0 and one "
    "selected checkpoint. Independently audit every admitted task, teacher/reference component, "
    "transfer and terminal optimizer endpoint; no historical full-autograd claim. Require284 fresh "
    "confirmation cases/all32 comparisons before every frozen MissBench/UCI/four-RecSys/matched-LLM "
    "suite and full480-comparison review,392 required/18 primary. No real-data selection, family "
    "exception, relaxed gate, benchmark subset, router or inference ensemble. Bind complete positive "
    "feasibility and independent five-file capacity preservation before registration. Keep4 GiB/64 "
    "MiB/192 GiB checkpoint/root/scratch training allowances,18/6/50 GiB reserves,176+32 GiB host "
    "working/reserve admission, four Torch/24 BLAS threads and.9 allocator cap. Preserve every prior "
    "failure and frozen artifact. Scratch is machine-local, not external backup."
)


def validate_host_admission(memory):
    if (memory["working_allowance_bytes"] != probe.HOST_WORKING_BYTES
            or memory["reserve_bytes"] != probe.HOST_RESERVE_BYTES
            or type(memory["available_bytes"]) is not int
            or memory["available_bytes"] < probe.HOST_WORKING_BYTES + probe.HOST_RESERVE_BYTES
            or memory["source"] != "/proc/meminfo:MemAvailable"):
        raise ValueError("Require the complete unchanged host working allowance and reserve")


def require_probe(current_probe, validate_preflight):
    if current_probe is not probe:
        raise ValueError("Bind the original complete four-draw feasibility attempt")
    config = probe.verify(probe.OUTPUT)
    recovered = recovery.verify()
    evidence = probe.require_predecessor()
    identities = {str(recovery.OUTPUT / "result.json"): FEASIBILITY_SHA256,
        str(recovery.mean.OUTPUT / "audit.json"): MEAN_SHA256}
    verify_hashes(identities)
    report = probe.read(recovery.OUTPUT / "result.json")
    result, audit, gate, retention, transfer, current = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "gradient_transfer_audit.json", "status.json")]
    recovery.validate_completed(result, audit, retention, transfer)
    validate_preflight(result, audit, gate, current, sha256(probe.OUTPUT / "audit.json"))
    if (report["complete"] is not True or report["feasibility_passed"] is not True
            or report["all_original_failures_preserved"] is not True or report["full_trial_authorized"] is not False
            or probe.read(recovery.OUTPUT / "status.json")["state"] != "complete"
            or Path(f'/proc/{current["pid"]}').exists()
            or config["anchor_retention"] != probe.retention.SPECIFICATION
            or config["native_teacher_mean"] != probe.retention.MEAN_SPECIFICATION
            or audit["teacher_replay"]["blend_tasks"] != 52 or audit["teacher_replay"]["component_grids"] != 104
            or audit["teacher_replay"]["observed_weights_refitted"] is not True):
        raise ValueError("Require the complete stopped probe and all original losses, teachers and adverse outcomes")
    for mapping in (config["source_sha256"], config["input_sha256"], recovered["source_sha256"], recovered["input_sha256"],
            report["artifact_sha256"], audit["artifact_sha256"], retention["artifact_sha256"]):
        verify_hashes(mapping)
    paths = [probe.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "audit.json", "retention_audit.json", "gradient_transfer_audit.json")]
    paths.extend(recovery.OUTPUT / name for name in ("config.json", "registration.json", "status.json"))
    return {**evidence, **identities, **config["input_sha256"], **recovered["input_sha256"],
        **{str(path): sha256(path) for path in paths}}


def storage_evidence(capacity, receipt_sha256):
    if receipt_sha256 != CAPACITY_RECEIPT_SHA256:
        raise ValueError("Bind the independently completed original five-file capacity preparation")
    return capacity.require_completed(receipt_sha256)
