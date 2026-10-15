"""Train V29 with admission-aware attempts and separate legacy original/anchor protection."""

import ast
import hashlib
import inspect

import train_single_matrixformer_anchor_rehearsal as template
from train_single_matrixformer_anchor_rehearsal import (
    baseline, compact_cache, composite, diagnostics, rehearsal, replicated, storage, teacher_policy, top_two,
)
import single_model_admission_attempts as attempts
import single_model_legacy_stratum_protection as paired
import single_model_training_admission as admission
from single_model_registered_adaptation import replace_assignment, replace_once


RATE, PREFIXES, PARENT_SHA256, PARENT_UPDATES = template.RATE, template.PREFIXES, template.PARENT_SHA256, template.PARENT_UPDATES
TRIAL_POLICY = ("Warm-start one ordinary77,998,496-parameter model from audited but unqualified V27 step10, "
    "never V28 or a diagnostic endpoint. Attempt64 fresh28-task mixed-prior batches at5e-6 fresh AdamW, "
    "training62 final-two-block/decoder tensors and preserving192 state tensors bitwise. Keep original "
    "general-imputation/specialist-rating teachers, .25 legacy/.25 structured/.5 ratings mass, original "
    "Huber/NLL/CDF/query objectives, observed-only normalization and native latent shapes. "
    + rehearsal.POLICY + " Override only legacy paired training protection: " + paired.POLICY + " "
    + admission.POLICY + " " + attempts.POLICY + " Keep compact complete task/teacher digests for every "
    "admitted task, all individual regressions and the full last actual gradient endpoint. Ordinary models "
    "stay on checkpoint storage; only declared synthetic diagnostics use UUID-bound local scratch, not "
    "provider-persistent storage or backup. Preserve18/6/50 GiB reserves and existing stage budgets. "
    "No inference ensemble, scientific retry, altered original source or real-data model selection.")
TRIAL_QUALIFICATION = ("Independently regenerate every scheduled structured admission draw, including rejected "
    "attempts, then digest-check every admitted full task/teacher payload and rating context. Audit all64 "
    "attempts, exact data-only no-ops,1368 original/V27-parent/development grids and all frozen model tensors. "
    "Audit the last actual optimizer endpoint, or prove unchanged initializer and empty optimizer if none "
    "exists; never fabricate a terminal gradient. Preserve all22 original V8 guards, original references, "
    "development boundaries0/10/25/50/64, ineligible step0 and earlier-boundary ties. Independently replay "
    "the actual endpoint and sole selected152-case state. Confirm once on284 fresh reserved synthetic cases "
    "and independently audit1136 grids. Only then publish a best alias and run every frozen tabular, "
    "four-dataset recommender and LLM comparison/review. No reselection, relaxed gate, omitted case or claim "
    "of goal success from synthetic evidence; no historical full-autograd replay claim.")


for name in ("require_capacity", "require_evidence"):
    exec(compile(inspect.getsource(getattr(template.framework, name)), __file__, "exec"), globals())


def validate_preflight(result, audit, gate, current, digest):
    template.validate_preflight(result, audit, gate, current, digest)
    if (result["data_admission_only"] is not False or audit["data_admission"]["complete"] is not True
            or audit["data_admission"]["eligible"] is not True or audit["data_admission"]["structured_draws"] != 24
            or result["record"]["projection"]["scope"] != "legacy_strata_other_paired_means"
            or len(result["record"]["projection"]["groups"]) != 45):
        raise ValueError("Require the actual independently audited admission-aware legacy-stratum preflight")


def save_last_endpoint(output, endpoint, gradients, terminal_step, accepted):
    descriptor = {"terminal_step": terminal_step, "optimizer_attempt_step": None,
                  "accepted_updates": accepted, "raw_endpoint_sha256": None,
                  "config_sha256": sha256(output / "config.json")}
    if endpoint is not None:
        if endpoint["record"]["accepted_updates"] != accepted:
            raise ValueError("Only data-only no-ops may follow the last captured optimizer attempt")
        payload = {**endpoint, "gradients": gradients[:len(endpoint["entries"])].cpu(),
                   "config_sha256": descriptor["config_sha256"]}
        descriptor["optimizer_attempt_step"] = endpoint["batch"]["step"]
        descriptor["raw_endpoint_sha256"] = save_endpoint(payload, output / "raw_final_update.pt")
    elif accepted:
        raise ValueError("Never accept updates without a real captured optimizer endpoint")
    write_json(output / "endpoint.json", descriptor)


ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import preflight_single_model_anchor_small_step as preflight': 'import preflight_single_model_legacy_strata as preflight',
    'OUTPUT = ROOT / "anchor_rehearsal_v28"': 'OUTPUT = ROOT / "legacy_strata_v29"',
    'SCRATCH = Path("/mnt/matrixformer_scratch/anchor_rehearsal_v28")': 'SCRATCH = Path("/mnt/matrixformer_scratch/legacy_strata_v29")',
    'DATA_SEED, GRADIENT_SEED = 631100901, 632100901': 'DATA_SEED, GRADIENT_SEED = 661100901, 662100901',
    'CONFIRMATION_SEEDS = (641100901, 642100901)': 'CONFIRMATION_SEEDS = (671100901, 672100901)',
    'or config["policy"] != POLICY':
        'or config["attempt_policy"] != attempts.POLICY or config["admission_policy"] != admission.POLICY\n'
        '            or config["policy"] != POLICY',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
node = next(node for node in ast.parse(ADAPTED_SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == "train")
TRAIN_SOURCE = ast.get_source_segment(ADAPTED_SOURCE, node)
TRAIN_SOURCE = replace_once(TRAIN_SOURCE, 'best, accepted, stop = {"step": 0, "score": anchor}, 0, []',
    'best, accepted, stop = {"step": 0, "score": anchor}, 0, []\n    last_endpoint = None')
start = TRAIN_SOURCE.index('            directory = output / "training_batches"')
end = TRAIN_SOURCE.index('            append_json(output / "training.jsonl", record)')
block = TRAIN_SOURCE[start:end]
terminal_start = block.index('            if step == STEPS or stop:')
block = block[:terminal_start]
block = replace_once(block, 'teachers, sampler, gradients)', 'teachers, checker, gradients)')
block = replace_once(block, '            paired.validate_layout(', '            checker.assert_consumed()\n            paired.validate_layout(')
block += ('            record.update(attempt_kind="optimizer_attempt", admission_manifest=str(admission_path),\n'
          '                          admission_sha256=sha256(admission_path))\n'
          '            last_endpoint = {"snapshot": snapshot, "record": record, "entries": entries, "batch": batch,\n'
          '                             "task_manifest": str(directory / "manifest.json")}\n'
          '            del snapshot\n')
replacement = ('            checker = admission.TrainingBatchAdmission(sampler, batch)\n'
    '            admission_path = output / "data_admission" / f"step_{step:03d}.json"\n'
    '            write_json(admission_path, checker.record)\n'
    '            if checker.eligible:\n' + ''.join('    ' + line + '\n' for line in block.splitlines()) +
    '            else:\n'
    '                checker.assert_consumed()\n'
    '                record = {"step": step, "attempt_kind": "data_admission_rejected", "accepted": False,\n'
    '                          "accepted_updates": accepted, "reason": "data_admission_rejected",\n'
    '                          "admission_manifest": str(admission_path), "admission_sha256": sha256(admission_path),\n'
    '                          "seconds": time.monotonic() - started, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}\n')
TRAIN_SOURCE = TRAIN_SOURCE[:start] + replacement + TRAIN_SOURCE[end:]
TRAIN_SOURCE = replace_once(TRAIN_SOURCE, '            del snapshot\n            if step in BOUNDARIES:', '            if step in BOUNDARIES:')
TRAIN_SOURCE = replace_once(TRAIN_SOURCE, '    verify_registration(output)\n',
    '        save_last_endpoint(output, last_endpoint, gradients, step, accepted)\n    verify_registration(output)\n')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, ast.get_source_segment(ADAPTED_SOURCE, node), TRAIN_SOURCE)
for name, value in (("POLICY", TRIAL_POLICY), ("QUALIFICATION", TRIAL_QUALIFICATION)):
    ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, name, value)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("constant_rate", "save_endpoint", "audit_endpoint"):
    exec(compile(inspect.getsource(getattr(template.framework.template, name)), __file__, "exec"), globals())
exec(compile(inspect.getsource(template.replicated_batches), __file__, "exec"), globals())

REGISTER_SOURCE = template.REGISTER_SOURCE
for old, new in {
    'REPO / "scripts/audit_single_model_anchor_rehearsal.py"': 'REPO / "scripts/audit_single_model_legacy_strata.py"',
    'REPO / "tests/test_single_model_anchor_training.py"':
        'REPO / "tests/test_single_model_legacy_strata_training.py", REPO / "tests/test_single_model_admission_attempts.py",\n'
        '               REPO / "tests/test_single_model_legacy_strata_audit.py", REPO / "tests/test_single_model_legacy_strata_pipeline.py"',
    'config = {"policy": POLICY,': 'config = {"policy": POLICY, "attempt_policy": attempts.POLICY, "admission_policy": admission.POLICY,',
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
                raise ValueError("Never restart a begun admission-aware continuation")
            try:
                train(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
