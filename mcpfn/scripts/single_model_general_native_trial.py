"""Qualify one native-general-retention student under the unchanged full protocol."""

import inspect

import single_model_stronger_mean_trial as predecessor
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
POLICY = (
    "A separately registered V47 trial initializes only from unchanged V29 step25/98 inherited updates, "
    "never from V46 or a probe. After V46's complete negative confirmation, all284-case initializer "
    "attribution and accepted independently audited native-general probe, change only native-imputation "
    "training retention from V8 to the original general parent. Every structured family retains V8 "
    "uniformly and ratings have no retention callback. Assignment uses validated synthetic cohort "
    "metadata only. Replace the unused V29 reference allocation with general while retaining actual "
    "V29 initialization. Preserve all four inherited V46 failures and10/22 worsened diagnostic metrics. "
    "Preserve the new probe's19/26 increased retention penalties and weighted total .19032076897416328 "
    "to .1943091234977801. Feasibility is not held-out accuracy; this change does not establish a MAR "
    "remedy. Keep mean20, observed-blend imputation/specialist rating teachers, all priors/.5/.25/.25 "
    "masses, observed-only normalization, Huber/NLL/CDF/query/point losses, weight10 count-shrunk "
    "equal-column equal-family retention, fresh5e-6 AdamW, fixed-half displacement,.1 margin,84 raw "
    "gradients,30 grouped protections,2x norm bound and exact rollback. Attempt64 fresh batches with "
    "no replacement draws. Keep boundaries0/10/25/50/64, all22 original development guards, strict "
    "minimax/earlier ties/ineligible0, sole selection and284 genuinely fresh confirmation cases with32 "
    "original comparisons before all frozen real suites and full review. Replay every original task, "
    "teacher/component and reference grid, up to128 general-native and1536 V8-structured grids. Keep "
    "one ordinary77,998,496-parameter network and16 frozen buffers at inference without routing or "
    "ensembling. Preserve18/6/50 GiB reserves,2432 MiB/16 MiB/40 GiB budgets,24 BLAS/4 Torch threads "
    "and.9 allocator cap. Scratch is machine-local, not external backup. Coordinators never initialize "
    "CUDA; full training/task-reference audits require visible GPU, saved-grid audits hide it. No "
    "real-data selection, consumed-seed reuse, scientific retry, guard relaxation or aggregate-only success."
)
PROBE_HASHES = {
    "config.json": "636930ea0c19dfb1a9a2c964696052e9537857f06a08d4e43ef8bcec704e729d",
    "result.json": "5339016cf12f7c413c99cbc284fda0e68de984209791d9ad31994cf8d067f81e",
    "audit.json": "ca4647bb0cbe73a0e0bb59e60860943755bdb16f89d4a0211132165100a85473",
    "gate.json": "bd1e83b83b372e3c90d896ba5e91e23f536011a1816e110b332cc64239c5905a",
    "retention_audit.json": "6bd01d1a8575f98ce35c2b030a9a9369d79b0b64e60b7ad9d07b4a9710834609",
}


def require_probe(probe, validate_preflight):
    probe.verify(probe.OUTPUT)
    evidence = probe.require_completed_v46()
    identities = {str(probe.OUTPUT / name): digest for name, digest in PROBE_HASHES.items()}
    probe.verify_hashes(identities)
    result, audit, gate, retention, config, current = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")]
    validate_preflight(result, audit, gate, current, PROBE_HASHES["audit.json"])
    if (probe.diagnosis.original.process_identity(current["pid"]) is not None
            or config["mean_distillation"] != probe.stronger.SPECIFICATION
            or config["anchor_retention"] != probe.retention.SPECIFICATION
            or retention["complete"] is not True or retention["eligibility_decisions"] != 28
            or retention["reference_grids"] != 26 or retention["reference_role_counts"] != {"general": 2, "v8": 24}
            or audit["teacher_replay"]["blend_tasks"] != 26 or audit["teacher_replay"]["component_grids"] != 52
            or audit["teacher_replay"]["observed_weights_refitted"] is not True):
        raise ValueError("Bind the complete stopped native-general probe, never V46's older recovery")
    for mapping in (config["source_sha256"], config["input_sha256"], audit["artifact_sha256"], retention["artifact_sha256"]):
        probe.verify_hashes(mapping)
    paths = [probe.OUTPUT / name for name in ("registration.json", "status.json")]
    return {**evidence, **identities, **config["input_sha256"],
            **{str(path): probe.sha256(path) for path in paths}}


STORAGE_EVIDENCE_SOURCE = inspect.getsource(predecessor.storage_evidence)
for old, new in (("!= 5", "!= 3"), ("2138934894", "1652558524"), ("five-file", "three-file")):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, "exec"), globals())


def adapt_training(source):
    return predecessor.adapt_training(source)


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    return replace_once(source, "    sources.update(Path(module.__file__).resolve()",
        '    sources.update({REPO / "tests/test_single_model_general_native_trial.py", '
        'REPO / "tests/test_single_model_general_native_full_audit.py", '
        'REPO / "tests/test_single_model_general_native_pipeline.py", '
        'REPO / "tests/test_single_model_general_native_retention_training.py"})\n'
        "    sources.update(Path(module.__file__).resolve()")


def adapt_recipe(source):
    return predecessor.adapt_recipe(source)
