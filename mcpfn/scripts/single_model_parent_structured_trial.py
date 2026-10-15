"""Qualify one parent-structured student without changing the frozen protocol."""

import inspect

import single_model_native_column_mean_trial as predecessor
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
adapt_training = predecessor.adapt_training
adapt_recipe = predecessor.adapt_recipe
POLICY = (
    "A separately registered V49 trial initializes only from unchanged actual V29 step25/98 "
    "inherited updates, never V48 or a probe endpoint. After fully audited negative V48 closure, "
    "complete284-case initializer attribution and the independently audited fresh feasibility "
    "probe, change only the structured training retention reference from V8 to a separate frozen "
    "actual V29 network, uniformly across every family/component. Native retention stays general "
    "and ratings have no retention. The consumed V48 bank shows two introduced structured "
    "failures and two improved inherited general raw-RMSE failures; attribution is not causation "
    "or fresh qualification. Preserve the probe's one worsened individual rating RMSE diagnostic, "
    "25/26 increased retention penalties and10/28 increased teacher-mean terms. All30 grouped "
    "point guards improve but that does not establish held-out improvement. Keep native "
    "count-shrunk teacher-mean weighting, raw-residual mean20 Huber, all NLL/CDF/query/point "
    "objectives, observed-blend imputation and specialist rating teachers, priors, normalization, "
    ".5/.25/.25 masses and weight10 shrunk equal-column retention. Use fresh5e-6 AdamW, fixed-half "
    "displacement,.1 margin,84 raw gradients,30 grouped protections,2x norm cap and exact rollback. "
    "Attempt64 fresh batches without replacement, with boundaries0/10/25/50/64, all22 original "
    "V8/general development guards, strict minimax/earlier ties/ineligible0 and sole selection. "
    "Require284 genuinely fresh confirmation cases/all32 comparisons before every frozen real "
    "suite and complete review. Replay every original task, teacher/component and reference "
    "grid, up to128 general-native and1536 V29-structured grids. Do not claim full historical "
    "autograd or optimizer-trajectory replay. Inference uses one ordinary77,998,496-parameter "
    "network and16 frozen buffers, without routing or ensembling. Preserve18/6/50 GiB reserves, "
    "2432 MiB/16 MiB/40 GiB budgets,24 BLAS/4 Torch threads and.9 allocator cap. Scratch is "
    "machine-local, not external backup. Coordinators remain CUDA-uninitialized; GPU inference "
    "and CUDA-hidden saved-grid audits remain separate. No real-data selection, seed reuse, "
    "scientific retry, family exception, guard relaxation or aggregate-only success."
)
PROBE_HASHES = {
    "config.json": "0d1fd3a0ff91f275de89709b8be8538b8611777bc296675a8e56fe0c84ec0821",
    "result.json": "4a047ec9573148486585c86c8b3a09af2da82c5b4f200accadc9899766e714ce",
    "audit.json": "00e7c97f5314ef0a5b7a8107f3414fadd4bf0daa2a4879ad42b423047e2f3e93",
    "gate.json": "4ca020a6969dc29c2e1123632b37e385cf634fffdcb4297bce72c5dc225b9d64",
    "retention_audit.json": "a1b848cb5eac20ef4ebf9938af52a83e1be71ccf7e40cabc3e8cb4e1014aa3bf",
}
MEAN_AUDIT_SHA256 = "42d66e16d488a9b2aff4e74db3b22cab0540be4d7c63dae90b46f1715ee1ba15"
PROBE_EVIDENCE_SOURCE = inspect.getsource(predecessor.require_probe)
for old, new in (("require_completed_v47", "require_completed_v48"), ('{"general": 2, "v8": 24}', '{"general": 2, "v29": 24}'),
        ('mean_audit["columns"] != 3279', 'mean_audit["columns"] != 3505'),
        ('mean_audit["adverse_teacher_mean_tasks"] != 16', 'mean_audit["adverse_teacher_mean_tasks"] != 10')):
    PROBE_EVIDENCE_SOURCE = replace_once(PROBE_EVIDENCE_SOURCE, old, new)
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, "exec"), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (("1562737394", "1922740672"), ("'canonical_artifacts_checked'] != 3", "'canonical_artifacts_checked'] != 5"),
                 ("three-file preservation", "five-file preservation")):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, "exec"), globals())


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_parent_structured_trial.py", '
        'REPO / "tests/test_single_model_parent_structured_full_audit.py", '
        'REPO / "tests/test_single_model_parent_structured_pipeline.py", '
        'REPO / "tests/test_single_model_parent_structured_training.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')
