"""Account for every scheduled attempt without treating data-only rejection as optimization."""

import math


POLICY = ("Every scheduled training attempt has an independently reproducible admission record. Empty-query "
    "or empty-observation structured data rejects the entire attempt before task capture, teacher inference "
    "or optimizer work. Log a data_admission_rejected event without losses, gradients, point metrics, rate "
    "or a fabricated task manifest. Keep its original ordinal and all scheduled development boundaries; "
    "never redraw, replace or redistribute its mass. Count captured tasks and rating contexts only for "
    "admitted optimizer attempts. Preserve the last actual optimizer snapshot and its raw gradients until "
    "the end, even when terminal scheduled attempts are data-only. Bind that actual endpoint separately "
    "from the terminal ordinal and checkpoint. If no attempt is admitted, there is no optimizer endpoint "
    "and all model/optimizer state must remain the initializer. No held-out gate or benchmark scope changes.")

DATA_ONLY_FIELDS = {"step", "attempt_kind", "accepted", "accepted_updates", "reason", "admission_manifest",
                    "admission_sha256", "seconds", "peak_gib"}


def check_attempt(batch, record, admission, accepted_before):
    if (record["step"] != batch["step"] or admission["step"] != batch["step"]
            or type(admission["eligible"]) is not bool
            or admission["task_count"] != len(batch["tasks"])
            or [row["task"] for row in admission["tasks"]] != batch["tasks"]
            or type(record["accepted"]) is not bool
            or type(record["accepted_updates"]) is not int
            or record["accepted_updates"] != accepted_before + int(record["accepted"])
            or not math.isfinite(record["seconds"]) or record["seconds"] < 0
            or not math.isfinite(record["peak_gib"]) or record["peak_gib"] < 0):
        raise ValueError("Bind every original task, ordinal and cumulative accepted-update count")
    if admission["eligible"]:
        if record["attempt_kind"] != "optimizer_attempt":
            raise ValueError("Admitted data must account for a real optimizer attempt")
    elif (set(record) != DATA_ONLY_FIELDS or record["attempt_kind"] != "data_admission_rejected"
            or record["reason"] != "data_admission_rejected" or record["accepted"] is not False):
        raise ValueError("Data-only rejection must not fabricate a normal gradient, metric or optimizer record")
    return record["accepted_updates"]


def inventory(batches, records, admissions):
    if (not batches or len(batches) != len(records) or len(records) != len(admissions)
            or [batch["step"] for batch in batches] != list(range(1, len(batches) + 1))):
        raise ValueError("Account for the complete contiguous scheduled attempt prefix")
    accepted, admitted_steps, rejected_steps, tasks, ratings = 0, [], [], 0, 0
    for batch, record, admission in zip(batches, records, admissions):
        accepted = check_attempt(batch, record, admission, accepted)
        if admission["eligible"]:
            admitted_steps.append(batch["step"])
            tasks += len(batch["tasks"])
            ratings += sum(task["specification"]["domain"] == "recsys" for task in batch["tasks"])
        else:
            rejected_steps.append(batch["step"])
    return {"scheduled_attempts": len(batches), "optimizer_attempts": len(admitted_steps),
            "admitted_steps": admitted_steps, "data_rejected_steps": rejected_steps,
            "training_tasks": tasks, "rating_contexts": ratings, "accepted_updates": accepted,
            "last_optimizer_attempt": admitted_steps[-1] if admitted_steps else None}


def check_endpoint_descriptor(descriptor, counts, config_digest):
    if (set(descriptor) != {"terminal_step", "optimizer_attempt_step", "accepted_updates", "raw_endpoint_sha256",
                            "config_sha256"}
            or descriptor["terminal_step"] != counts["scheduled_attempts"]
            or descriptor["optimizer_attempt_step"] != counts["last_optimizer_attempt"]
            or descriptor["accepted_updates"] != counts["accepted_updates"]
            or descriptor["config_sha256"] != config_digest):
        raise ValueError("Distinguish the final scheduled ordinal from the last actual optimizer attempt")
    digest = descriptor["raw_endpoint_sha256"]
    if counts["last_optimizer_attempt"] is None:
        if digest is not None or counts["accepted_updates"] != 0:
            raise ValueError("No admitted task means no invented gradient endpoint or accepted update")
    elif not isinstance(digest, str) or len(digest) != 64 or any(letter not in "0123456789abcdef" for letter in digest):
        raise ValueError("Bind the complete last actual raw optimizer endpoint")
