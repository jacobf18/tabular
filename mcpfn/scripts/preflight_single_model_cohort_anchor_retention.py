"""Probe cohort-specific frozen retention after V41's closed negative confirmation."""

import ast
from collections import Counter
import fcntl
from pathlib import Path

import diagnose_single_model_legacy_blend_initializer as diagnosis
import preflight_single_model_legacy_blend_teacher as previous_probe
import single_model_cohort_anchor_retention as cohort_retention
import single_model_legacy_blend_replay as blend_replay
from single_model_registered_adaptation import replace_assignment, replace_once


ARITHMETIC = {"blas_threads": 24, "torch_threads": 4}
POLICY_TEXT = ("One fresh28-task feasibility probe after V41's complete64-attempt/56-update training, "
    "independent audits, selected step50,284-case failed confirmation and complete284-grid V29 initializer "
    "diagnosis. Preserve all five failed comparisons and every adverse diagnostic result. "
    + cohort_retention.POLICY + " This registration authorizes one probe and independent replays, not full "
    "training, confirmation or a real campaign. Keep V29 step25/98 inherited updates as initializer; never "
    "warm-start a scientific trial from this probe. Keep238 tensors/77,998,496 trainable parameters,16 "
    "frozen buffers,84 raw gradients,30 protections, fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x "
    "norm bound and exact rollback. Preserve the.9 allocator cap and18/6/50 GiB reserves with512 MiB/16 "
    "MiB/32 GiB write allowances. The existing full26-grid reference allowance covers both identities; "
    "measure the additional frozen model's actual memory rather than assuming feasibility. Pin24 BLAS/4 "
    "Torch threads in every stage, including independent teacher reconstruction. Replay both original "
    "legacy components, exact observed routing, every structured/rating teacher and both frozen retention "
    "references independently. No changed priors, loss formula, inference ensemble, relaxed guards, "
    "consumed-seed reuse, scientific retry, reference substitution or benchmark-success claim.")


def reference_identities():
    return {"v8": {"checkpoint": str(previous_probe.training.alignment.fixed.PARENT),
                    "sha256": cohort_retention.REFERENCE_SHA256S["v8"]},
            "v29": {"checkpoint": str(previous_probe.PARENT), "sha256": previous_probe.PARENT_SHA256}}


def validate_completed(training_audit, replay, replay_gate, closure, diagnostic):
    required = {"complete": True, "complete_training_audit": True, "through_step": 64,
        "attempted_batches": 64, "accepted_updates": 56, "prediction_grids": 1368,
        "full_network_training": True, "fixed_half_training": True, "displacement_fraction": .5,
        "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
        "independently_regenerated_structured_draws": 1536, "confirmation_evaluated": False}
    failures = {(row["baseline"], row["group"], row["metric"]) for row in closure["failures"]}
    expected = {("V8_retention_baseline", "imputation", metric)
        for metric in ("missbench_nrmse", "uci_nrmse", "common_nrmse", "rmse")}
    expected.add(("V8_retention_baseline", "mnar_two_phase_subset", "common_nrmse"))
    if (any(training_audit.get(key) != value for key, value in required.items())
            or training_audit["best"]["step"] != 50
            or training_audit["admission_attempts"]["data_rejected_steps"] != []
            or training_audit["admission_attempts"]["optimizer_attempts"] != 64
            or training_audit["raw_endpoint"]["adam"]["gradient_vectors"] != 84
            or training_audit["raw_endpoint"]["optimizer_attempt_step"] != 64
            or replay["complete"] is not True or replay["endpoint_grids"] != 56 or replay["selected_grids"] != 152
            or replay_gate["endpoint_reproduced"] is not True or replay_gate["selected_validation_passed"] is not True
            or replay_gate["selected_step"] != 50 or closure["complete"] is not True
            or closure["passed_original_confirmation"] is not False or closure["prediction_grids"] != 1136
            or closure["cases"] != 284 or closure["selected_step"] != 50 or len(closure["comparisons"]) != 32
            or len(closure["failures"]) != 5 or failures != expected
            or diagnostic["complete"] is not True or diagnostic["prediction_grids"] != 284
            or len(diagnostic["comparisons"]) != 22 or diagnostic["cuda_initialized"] is not False
            or diagnostic["post_hoc_consumed_bank"] is not True
            or any(diagnostic[name] is not False for name in ("checkpoint_selection_performed", "confirmation_repeated",
                "training_performed", "real_benchmarks_evaluated"))):
        raise ValueError("Require the complete V41 failure and full consumed-bank initializer diagnosis")


def require_completed_v41():
    diagnosis.require_closed_failure()
    original = diagnosis.original
    paths = {
        original.training.OUTPUT / "config.json": "64dee12d225b00d7c31f4b7dbb144e601b916cc99fc269347e7268443ddfc683",
        original.AUDIT / "audit.json": "356672096ce19ee456858b88e42aecd204ba39b2fc9d97d79d62734c934f28e2",
        diagnosis.recovery.REPLAY / "audit.json": "d810fe7a6a1afd55caa35b5d6a197aa908f6cbe1a88b68407bb0ce3cec48c258",
        original.CONFIRMATION / "audit.json": "a5730c034c5f752840c29d25371172c291dc46d9d170dcc85dfb3f71c7106145",
        original.CONFIRMATION / "gate.json": diagnosis.CONFIRMATION_GATE_SHA256,
        diagnosis.CLOSURE: diagnosis.CLOSURE_SHA256,
        diagnosis.OUTPUT / "config.json": "ddc1ef933d55d7706665f7228feba5225bbd615593ecbaf3b34ed8d05bd6dc0e",
        diagnosis.OUTPUT / "audit.json": "213e8cd65d42d85ba63a0ec6dad532665f580c4de4a9f54eb0fdfd96ece140cc",
    }
    expected = {str(path): digest for path, digest in paths.items()}
    verify_hashes(expected)
    training_audit = read(original.AUDIT / "audit.json")
    replay = read(diagnosis.recovery.REPLAY / "audit.json")
    diagnostic = read(diagnosis.OUTPUT / "audit.json")
    validate_completed(training_audit, replay, read(diagnosis.recovery.REPLAY / "gate.json"),
        read(diagnosis.CLOSURE), diagnostic)
    require_auditor_stopped()
    for directory in (original.training.OUTPUT, diagnosis.recovery.OUTPUT,
            diagnosis.recovery.REPLAY, original.CONFIRMATION, diagnosis.OUTPUT):
        current = read(directory / "status.json")
        if current["state"] not in {"complete", "training_complete"} or original.process_identity(current["pid"]) is not None:
            raise ValueError("Stop every predecessor producer before registering the fresh probe")
        path = directory / "status.json"
        expected[str(path)] = sha256(path)
    for report in (training_audit, replay, diagnostic):
        verify_hashes(report["artifact_sha256"])
    config = read(diagnosis.OUTPUT / "config.json")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return expected


def require_auditor_stopped(proc=Path("/proc")):
    for path in proc.glob("[0-9]*/cmdline"):
        try:
            arguments = path.read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if any(argument.rsplit(b"/", 1)[-1] == b"audit_single_model_legacy_blend_teacher.py" for argument in arguments):
            raise ValueError("The original audit has no status file; require its producer to be stopped")


def replay_cohort_targets(output):
    replay_original_teachers(output)
    config = verify(output)
    if read(output / "result.json").get("data_admission_only"):
        return
    manifest, records = retention_records(output)
    directory = output / "retention_replay"
    directory.mkdir()
    status(output, "replaying_retention_teachers")
    model, optimizer, criterion, teachers, sampler = training.alignment.mixture.original_setup(config["base"])
    del optimizer, teachers, sampler
    identities = reference_identities()
    references = retention.make_references(model, identities)
    del model
    verified = {role: retention.verify_model(reference, identities[role]) for role, reference in references.items()}
    if verified != manifest["reference"]:
        raise ValueError("Independently load both exact original retention checkpoints")
    rows = []
    for task, record, episode, payload in records:
        role = retention.reference_role(episode.metadata)
        with torch.no_grad(), compute_context(torch.device("cuda")):
            prediction = teacher_predictions(references[role], episode.source.cuda(), criterion, **STREAM_OPTIONS).float().cpu()
        if not composite.byte_equal(prediction, payload["prediction"]):
            raise ValueError("Independent cohort-reference inference must reproduce every full grid exactly")
        path = directory / f'reference_{record["seed"]}.npz'
        atomic_npz(path, prediction=prediction.numpy())
        rows.append({"seed": record["seed"], "artifact": str(path), "sha256": sha256(path),
            "reference_role": role, "identity": identities[role]})
    if any(retention.verify_model(reference, identities[role]) != verified[role]
            for role, reference in references.items()):
        raise ValueError("Both independently replayed references must remain frozen")
    write_json(directory / "manifest.json", {"complete": True, "pid": os.getpid(), "records": rows,
        "reference": verified, "input_manifest_sha256": sha256(output / "retention_targets/manifest.json")})
    status(output, "awaiting_independent_audit")


def replace_function(source, name, replacement):
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError("Require one undecorated top-level function for a declared probe adaptation")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement + "\n" + "".join(lines[node.end_lineno:])


SOURCE = previous_probe.SOURCE.split('\nif __name__ == "__main__":')[0]
SOURCE = replace_function(SOURCE, "validate_latest", "")
SOURCE = replace_function(SOURCE, "require_v40_predecessor", "")
SOURCE = replace_function(SOURCE, "require_predecessor", "def require_predecessor():\n    return require_completed_v41()")
SOURCE = replace_function(SOURCE, "replay_targets", "def replay_targets(output):\n    return replay_cohort_targets(output)")
for old, new in {
    "import single_model_all_imputation_retention as retention": "import single_model_cohort_anchor_retention as retention",
    "legacy_blend_teacher_preflight_v1": "cohort_anchor_retention_preflight_v1",
    "1021100901, 1022100901": "1051100901, 1052100901",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing cohort-probe source adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)

REGISTER_ADAPTATIONS = [
    ('    paths = [PARENT,', '    paths = [Path(reference_identities()["v8"]["checkpoint"]), PARENT,'),
    ('config = {"policy": POLICY,', 'config = {"policy": POLICY, "retention_references": reference_identities(), "arithmetic": ARITHMETIC,'),
    ('    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_cohort_anchor_retention.py", REPO / "tests/test_single_model_cohort_anchor_retention_preflight.py"})\n'
        '    sources.update(Path(module.__file__).resolve()'),
]
registration_patch = "\n".join(f"REGISTER_SOURCE = replace_once(REGISTER_SOURCE, {old!r}, {new!r})"
    for old, new in REGISTER_ADAPTATIONS)
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_patch + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
RUN_ADAPTATIONS = [
    ('        anchor = deepcopy(model).eval().requires_grad_(False)',
        '        identities = reference_identities()\n        references = retention.make_references(model, identities)'),
    ('        identity = {"checkpoint": str(PARENT), "sha256": PARENT_SHA256}\n', ''),
    ('output / "retention_targets", anchor, identity,', 'output / "retention_targets", references, identities,'),
    ('        del anchor', '        del references'),
    ('or config["anchor_retention"] != retention.SPECIFICATION',
        'or config["anchor_retention"] != retention.SPECIFICATION\n'
        '            or config["retention_references"] != reference_identities() or config["arithmetic"] != ARITHMETIC'),
]
runtime_patch = "\n".join(f"ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, {old!r}, {new!r})" for old, new in RUN_ADAPTATIONS)
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)',
    runtime_patch + '\nADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)')
for old, new in [
    ('\n    identity = {"checkpoint": str(PARENT), "sha256": PARENT_SHA256}\n', '\n    identities = reference_identities()\n'),
    ('manifest["reference"] != {**identity, "state_tensors": 254, "bitwise_equal": True}',
        'manifest["reference"] != {role: {**identity, "state_tensors": 254, "bitwise_equal": True} for role, identity in identities.items()}'),
    ('or record["eligible"] is not retention.eligible(episode.metadata)',
        'or record["eligible"] is not retention.eligible(episode.metadata)\n'
        '                or record["reference_role"] != retention.reference_role(episode.metadata)'),
    ('payload["identity"] != identity or payload["specification"]',
        'payload["identity"] != identities[record["reference_role"]] or payload["reference_role"] != record["reference_role"]\n'
        '                    or payload["specification"]'),
    ('if not records:', 'if len(records) != 26 or set(record["reference_role"] for task, record, episode, payload in records) != {"v8", "v29"}:'),
    ('if replayed != {"seed": record["seed"], "artifact": str(path), "sha256": sha256(path)}:',
        'if replayed != {"seed": record["seed"], "artifact": str(path), "sha256": sha256(path),\n'
        '                "reference_role": record["reference_role"], "identity": reference_identities()[record["reference_role"]]}:'),
    ('row = {"task": task["name"], "seed": record["seed"], "queries": int(episode.mask.sum())}',
        'row = {"task": task["name"], "seed": record["seed"], "queries": int(episode.mask.sum()), "reference_role": record["reference_role"]}'),
    ('"eligibility_decisions": 28, "reference_grids": len(records),',
        '"eligibility_decisions": 28, "reference_grids": len(records), "reference_role_counts": dict(Counter(record["reference_role"] for task, record, episode, payload in records)),'),
]:
    if SOURCE.count(old) != 1:
        raise ValueError(f"Require one cohort audit adaptation, found {SOURCE.count(old)}: {old}")
    SOURCE = replace_once(SOURCE, old, new)
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
exec(compile(SOURCE, __file__, "exec"), globals())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    stage = parser.parse_args().stage
    with diagnosis.recovery.previous.original_arithmetic():
        if stage == "register":
            register(OUTPUT)
            return
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error), failed_stage=stage)
                raise


if __name__ == "__main__":
    main()
