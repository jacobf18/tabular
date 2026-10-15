"""Audit every V29 admission attempt, admitted task and actual terminal optimizer state."""

from contextvars import ContextVar
import hashlib
import inspect
import sys
import time

import audit_single_model_anchor_rehearsal as template
from audit_single_model_anchor_rehearsal import compact_cache, compact_template, compressed_endpoint
import single_model_admission_attempts as attempts
import single_model_training_admission as admission
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_legacy_strata_audit_replay", default=None)


def admission_inventory(config, rows):
    saved = [training.read(training.OUTPUT / "data_admission" / f'step_{batch["step"]:03d}.json')
             for batch in config["batches"][:len(rows)]]
    return attempts.inventory(config["batches"][:len(rows)], rows, saved)


def replay_admission(run, batch, row, accepted, hashes):
    active = REPLAY.get()
    if active is None:
        raise ValueError("Require an independent admission/task replay runtime")
    path = run / "data_admission" / f'step_{batch["step"]:03d}.json'
    if row["admission_manifest"] != str(path) or row["admission_sha256"] != sha256(path):
        raise ValueError("Bind the complete original admission record before replay")
    saved = training.read(path)
    checker = admission.TrainingBatchAdmission(active["original_sampler"], batch)
    if admission.content_digest(checker.record) != admission.content_digest(saved):
        raise ValueError("Preserve every scheduled structured mask, raw-array digest, episode digest and decision")
    attempts.check_attempt(batch, row, saved, accepted)
    hashes[str(path)] = sha256(path)
    active["sampler"] = checker
    active["admission_steps"].append(batch["step"])
    active["structured_draws"] += saved["structured_draws"]
    if not checker.eligible:
        checker.assert_consumed()
        directory = run / "training_batches" / f'step_{batch["step"]:03d}'
        if directory.exists() or directory.is_symlink():
            raise ValueError("A data-only attempt cannot contain captured model/teacher training tasks")
    return checker


def check_empty_optimizer_endpoint(latest, parent):
    if (latest["gradient_updates_performed"] != 0 or latest["optimizer"]["state"]
            or latest["model"].keys() != parent.keys()
            or any(not training.composite.byte_equal(latest["model"][name], value) for name, value in parent.items())):
        raise ValueError("An all-data-rejected trial must preserve the full initializer and empty optimizer")


def audit_final_endpoint(run, config, rows, counts, latest, parent, selected, rollback_rates, hashes):
    descriptor_path = run / "endpoint.json"
    descriptor = training.read(descriptor_path)
    attempts.check_endpoint_descriptor(descriptor, counts, sha256(run / "config.json"))
    hashes[str(descriptor_path)] = sha256(descriptor_path)
    step = descriptor["optimizer_attempt_step"]
    raw_path = run / "raw_final_update.pt"
    if step is None:
        if raw_path.exists() or raw_path.is_symlink():
            raise ValueError("Do not fabricate raw gradients when no data batch was admitted")
        check_empty_optimizer_endpoint(latest, parent)
        return {"optimizer_attempt_step": None, "terminal_step": descriptor["terminal_step"],
                "accepted_updates": 0, "gradient_vectors": 0, "unchanged_initializer_checked": True}
    record = rows[step - 1]
    hashes[str(raw_path)] = descriptor["raw_endpoint_sha256"]
    verify_hashes({str(raw_path): hashes[str(raw_path)]})
    payload = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)
    if (payload["config_sha256"] != sha256(run / "config.json") or payload["batch"] != config["batches"][step - 1]
            or payload["record"] != record or payload["entries"] != training.read(record["task_manifest"])["entries"]
            or payload["task_manifest"] != record["task_manifest"]):
        raise ValueError("Bind the last actual optimizer attempt, not a later data-only scheduled ordinal")
    hashes.update(training.audit_endpoint(payload, raw_path))
    result = checks.check_endpoint(payload, record, [row[2] for row in selected], config["base"]["weight_decay"],
                                   rollback_rates[step], latest["optimizer"])
    expected = torch.cat([latest["model"][name].flatten() for name, shape, size in selected])
    if not torch.equal(expected, payload["snapshot"]["after"]):
        raise ValueError("Terminal checkpoint must equal the accepted or rolled-back actual endpoint")
    return {**result, "optimizer_attempt_step": step, "terminal_step": descriptor["terminal_step"]}


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_anchor_rehearsal as training': 'import train_single_matrixformer_legacy_strata as training',
    'import single_model_paired_protection as checks': 'import single_model_legacy_stratum_protection as checks',
    '    boundaries = [step for step in training.BOUNDARIES if step <= through_step]':
        '    counts = admission_inventory(config, rows)\n    boundaries = [step for step in training.BOUNDARIES if step <= through_step]',
    '        step = batch["step"]\n        manifest_path':
        '        step = batch["step"]\n        checker = replay_admission(run, batch, row, accepted, hashes)\n'
        '        if not checker.eligible:\n            accepted_by_step[step] = accepted\n            continue\n        manifest_path',
    '        if entries != manifest["entries"]:': '        checker.assert_consumed()\n        if entries != manifest["entries"]:',
    '"training_episodes": 28 * through_step, "teacher_identity_checked_tasks": 28 * through_step,':
        '"training_episodes": counts["training_tasks"], "teacher_identity_checked_tasks": counts["training_tasks"],\n'
        '               "admission_attempts": counts, "independently_regenerated_structured_draws": REPLAY.get()["structured_draws"],',
    'The V28 initialization must reproduce the audited V27 step10 prediction grids':
        'The V29 initialization must reproduce the audited V27 step10 prediction grids',
    '# Anchor-rehearsal training audit': '# Admission-aware legacy-stratum training audit',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
start = ADAPTED_SOURCE.index('        raw_path = run / "raw_final_update.pt"')
end = ADAPTED_SOURCE.index('        hashes.update({str(run / name)', start)
ADAPTED_SOURCE = ADAPTED_SOURCE[:start] + (
    '        endpoint_result = audit_final_endpoint(run, config, rows, counts, latest, parent, selected, rollback_rates, hashes)\n'
    ) + ADAPTED_SOURCE[end:]
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
exec(compile(inspect.getsource(compact_template.load_regenerated_task), __file__, "exec"), globals())


def regeneration_evidence(through_step):
    active = REPLAY.get()
    config = training.read(training.OUTPUT / "config.json")
    rows, raw = read_prefix(training.OUTPUT / "training.jsonl", range(1, through_step + 1))
    counts = admission_inventory(config, rows)
    expected = {str(training.OUTPUT / "training_batches" / f"step_{step:03d}" / f"task_{index:02d}.pt")
                for step in counts["admitted_steps"] for index in range(28)}
    if (active is None or active["seen"] != expected or active["rating_contexts"] != counts["rating_contexts"]
            or active["admission_steps"] != list(range(1, through_step + 1))
            or active["structured_draws"] != 24 * through_step):
        raise ValueError("Regenerate every admission draw and every admitted full task/teacher, never a subset")
    actual = {str(path) for path in (training.OUTPUT / "training_batches").glob("step_*/*.pt")}
    if actual != expected:
        raise ValueError("Task capsules must cover exactly the admitted schedule without data-only replacements")
    return {str(active["output"] / name): sha256(active["output"] / name) for name in ("config.json", "tasks.jsonl")}


RUN_AUDIT_SOURCE = inspect.getsource(compact_template.run_audit)
for old, new in {
    '    directory = output.with_name(output.name + "_task_replay")':
        '    rows, raw = read_prefix(training.OUTPUT / "training.jsonl", range(1, training.STEPS + 1))\n'
        '    counts = admission_inventory(config, rows)\n    directory = output.with_name(output.name + "_task_replay")',
    '    directory.mkdir()': '    directory.mkdir()\n    (directory / "tasks.jsonl").touch()',
    '"training_tasks": 28 * training.STEPS,': '"training_tasks": counts["training_tasks"], "admission_attempts": counts,',
    '"backend": training.original.alignment.mixture.backend, "seen": set(), "rating_contexts": 0}':
        '"backend": training.original.alignment.mixture.backend, "seen": set(), "rating_contexts": 0,\n'
        '                      "original_sampler": sampler, "admission_steps": [], "structured_draws": 0}',
    'report["independently_regenerated_training_tasks"] != 28 * training.STEPS':
        'report["independently_regenerated_training_tasks"] != counts["training_tasks"]',
    'report["independently_regenerated_rating_contexts"] != 2 * training.STEPS':
        'report["independently_regenerated_rating_contexts"] != counts["rating_contexts"]',
}.items():
    RUN_AUDIT_SOURCE = replace_once(RUN_AUDIT_SOURCE, old, new)
exec(compile(RUN_AUDIT_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "legacy_strata_v29_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
