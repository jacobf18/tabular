"""Qualify the native-column student under the original complete protocol."""

import single_model_general_native_trial as predecessor
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
POLICY = (
    "A separately registered V48 trial initializes only from unchanged V29 step25/98 inherited "
    "updates, never V47 or a probe. After the audited negative V47 closure, complete five-boundary "
    "diagnosis and independently audited native-column probe, change only native teacher-mean "
    "Huber cell weighting to the existing count-shrunk equal-column coefficients on all eight "
    "native components. Keep nominal mean20, raw-residual delta1, one unit-variance pseudocount, "
    "1e-6 floor, detached synthetic target moments and every original query. Structured/rating "
    "callbacks stay unchanged. Preserve general-native/V8-structured retention, observed-blend "
    "imputation/specialist rating teachers, NLL/CDF/query/point objectives, all priors/.5/.25/.25 "
    "masses and normalization. The probe improves30 grouped and56 individual training-batch point "
    "checks but worsens19/26 retention penalties and16/28 teacher-mean terms; preserve those adverse "
    "outcomes and do not infer held-out improvement or a causal comparison across distinct seeds. "
    "Use fresh5e-6 AdamW, fixed-half displacement,.1 margin,84 raw gradients,30 grouped protections, "
    "2x norm cap and exact rollback. Attempt64 fresh batches without replacement. Keep boundaries "
    "0/10/25/50/64, all22 development guards, strict minimax/earlier ties/ineligible0, sole selection "
    "and284 genuinely fresh confirmation cases with32 original comparisons before every frozen "
    "real suite and complete review. Independently replay every original task, teacher/component "
    "and reference grid, including up to128 general-native and1536 V8-structured grids. One ordinary "
    "77,998,496-parameter network and16 frozen buffers at inference without routing or ensembling. "
    "Preserve18/6/50 GiB reserves,2432 MiB/16 MiB/40 GiB budgets,24 BLAS/4 Torch threads and.9 "
    "allocator cap. Scratch is machine-local, not external backup. Coordinators never initialize "
    "CUDA; full task/reference audits have GPU visibility and saved-grid audits hide CUDA. No "
    "real-data selection, seed reuse, scientific retry, guard relaxation or aggregate-only success."
)
PROBE_HASHES = {
    "config.json": "216fa8bf6745f7bc9f37c06ac89b720dc4d1ebca153ec116a3c576360399f0c2",
    "result.json": "0c9802d9539d45d993373b8e89e9c3ee924efb0ca65502606641285e60f0c81f",
    "audit.json": "0ae006744c0afc59cbd0e81bf6f6260cc6842eaaae4e85dacbbc872772611fbd",
    "gate.json": "3315e57743580c9ae724dd9262fa6a9070245448f4781fbfd2192422ddf7b887",
    "retention_audit.json": "18e4a277bd295a85bdf3f00f1e40505ab76d25347e6d2055e010be2723f1fe6a",
}
MEAN_AUDIT_SHA256 = "1cce742f2c8351f1b06dbbf479be258fc83104adc7a12e854ed6e07118d26089"


def require_probe(probe, validate_preflight):
    probe.verify(probe.OUTPUT)
    evidence = probe.require_completed_v47()
    identities = {str(probe.OUTPUT / name): digest for name, digest in PROBE_HASHES.items()}
    probe.verify_hashes(identities)
    result, audit, gate, retention, config, current = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "gate.json", "retention_audit.json", "config.json", "status.json")]
    validate_preflight(result, audit, gate, current, PROBE_HASHES["audit.json"])
    if (probe.diagnosis.original.process_identity(current["pid"]) is not None
            or config["mean_distillation"] != probe.stronger.SPECIFICATION
            or config["native_teacher_mean"] != probe.retention.MEAN_SPECIFICATION
            or config["anchor_retention"] != probe.retention.SPECIFICATION
            or retention["complete"] is not True or retention["eligibility_decisions"] != 28
            or retention["reference_grids"] != 26 or retention["reference_role_counts"] != {"general": 2, "v8": 24}
            or audit["teacher_replay"]["blend_tasks"] != 26 or audit["teacher_replay"]["component_grids"] != 52
            or audit["teacher_replay"]["observed_weights_refitted"] is not True):
        raise ValueError("Bind the complete stopped native-column probe and unchanged references")
    mean_directory = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
    identities[str(mean_directory / "audit.json")] = MEAN_AUDIT_SHA256
    probe.verify_hashes(identities)
    mean_audit = probe.read(mean_directory / "audit.json")
    if (mean_audit["complete"] is not True or mean_audit["tasks"] != 28 or mean_audit["columns"] != 3279
            or mean_audit["native_reweighted_tasks"] != 2 or mean_audit["unchanged_structured_tasks"] != 24
            or mean_audit["unchanged_rating_tasks"] != 2 or mean_audit["adverse_teacher_mean_tasks"] != 16
            or any(mean_audit[key] is not False for key in ("new_gate", "model_inference_performed",
                "full_composite_loss_replayed", "autograd_replayed", "cuda_initialized"))):
        raise ValueError("Preserve the complete descriptive mean-term audit and its adverse outcomes")
    mean_config = probe.read(mean_directory / "config.json")
    if probe.sha256(mean_directory / "config.json") != mean_audit["config_sha256"]:
        raise ValueError("Bind the original supplementary mean-audit registration")
    for mapping in (config["source_sha256"], config["input_sha256"], audit["artifact_sha256"], retention["artifact_sha256"],
            mean_audit["artifact_sha256"], mean_config["source_sha256"], mean_config["input_sha256"]):
        probe.verify_hashes(mapping)
    paths = [probe.OUTPUT / name for name in ("registration.json", "status.json")]
    paths.append(mean_directory / "config.json")
    return {**evidence, **identities, **config["input_sha256"], **mean_audit["artifact_sha256"],
        **{str(path): probe.sha256(path) for path in paths}}


STORAGE_EVIDENCE_SOURCE = replace_once(predecessor.STORAGE_EVIDENCE_SOURCE, "1652558524", "1562737394")
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, "exec"), globals())


def adapt_training(source):
    source = predecessor.adapt_training(source)
    return replace_once(source, 'or config["mean_distillation"] != stronger.SPECIFICATION',
        'or config["mean_distillation"] != stronger.SPECIFICATION\n'
        '            or config["native_teacher_mean"] != retention.MEAN_SPECIFICATION')


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    source = replace_once(source, 'config = {"policy": POLICY,', 'config = {"policy": POLICY, "native_teacher_mean": retention.MEAN_SPECIFICATION,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_native_column_trial.py", '
        'REPO / "tests/test_single_model_native_column_full_audit.py", '
        'REPO / "tests/test_single_model_native_column_pipeline.py", '
        'REPO / "tests/test_single_model_native_column_training.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    return predecessor.adapt_recipe(source)
