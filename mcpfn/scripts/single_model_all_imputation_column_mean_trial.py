"""Qualify one all-imputation column-mean student under the frozen protocol."""

import single_model_native_parent_trial as predecessor
import single_model_all_imputation_column_mean_storage as component_storage
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
adapt_training = predecessor.adapt_training
POLICY = (
    "A separately registered V52 trial initializes only from unchanged actual V29 step25/98 "
    "inherited updates, never V51 or a feasibility endpoint. After fully audited negative V51 "
    "closure and a complete fresh feasibility probe, extend native count-shrunk equal-column "
    "mean20 Huber weighting to every structured imputation task uniformly across all twelve "
    "families and four components. Keep native/rating callbacks, delta1, one unit-variance "
    "pseudocount,1e-6 floor, detached target moments and every original query. Native retention "
    "stays actual V29; structured retention stays the exact FP32 half-sum of V8/V29; ratings "
    "have no retention. Preserve both adverse individual probe metrics,24/26 increased retention "
    "penalties and7/28 worsened teacher-mean terms despite all30 grouped point improvements; "
    "feasibility is not held-out or benchmark improvement. Keep NLL/CDF/query/point terms, "
    "observed-blend imputation/specialist rating teachers, all mixed priors/.5/.25/.25 masses, "
    "weight10 retention, normalization and inference unchanged. Fresh5e-6 AdamW, fixed-half "
    "displacement,.1 inward margin,84 raw gradients,30 grouped protections,2x norm cap and "
    "exact rollback. Attempt64 fresh batches without replacement, retain boundaries0/10/25/50/64 "
    "and all22 original development guards, strict minimax/earlier ties/ineligible0 and sole "
    "selection. Require284 genuinely fresh confirmation cases and all32 original V8/general "
    "comparisons before every frozen real suite and full review. Independently replay every "
    "original task, teacher/component and retention reference, up to3200 retention components "
    "and1664 targets. One ordinary77,998,496-parameter network and16 frozen buffers at inference, "
    "without routing or ensembling. Preserve18/6/50 GiB reserves,2432 MiB/16 MiB checkpoint/root "
    "and complete48 GiB scratch allowances without compression/rejection assumptions. Keep24 "
    "BLAS/4 Torch threads and.9 allocator cap. Scratch is machine-local, not external backup. "
    "CUDA-hidden coordinators, GPU full-task auditors and separate CPU saved-grid audits keep "
    "original roles. No historical full-autograd/optimizer-trajectory claim, real-data selection, "
    "seed reuse, scientific retry, family exception, guard relaxation or aggregate-only success."
)
PROBE_HASHES = {
    "config.json": "669d4b633a362e5e577771ed48d7838253a3ced0ea7e496d76087bc7c93f6d11",
    "result.json": "3bbe5ab09090297e07335feb331ff18139cf13d811d80f359b5b9795be8e0323",
    "audit.json": "9920b8778fe5f3750e24d457706838ea8357a41bf725a467960386b85b76fbbc",
    "gate.json": "f6a6dc0090748d1b7182de85d04b827d14008d8d3eb7d90908e5a270bc392f0f",
    "retention_audit.json": "77ecbcdd237b8b34127052f0e1de259e6910163a650f94772929bf72d1e3b842",
    "OUTCOME.md": "7973c05618ec5a11e977ae317b587c4ab633385f37410ebbae9dd496d515eddd",
    "external_recheck.json": "7e02c8034dd0c60b6bd9dd65d3aed6b590af3283af7e4e7f566b11ba568a5834",
}
MEAN_AUDIT_SHA256 = "9496bb89f46c54d92fbfc14f65767a251497dcc4368a5e2013254be880708a14"
EXPECTED_MEAN_COLUMNS = 3353
EXPECTED_MEAN_ADVERSE = 7
PROBE_EVIDENCE_SOURCE = predecessor.PROBE_EVIDENCE_SOURCE
for old, new in (
    ("require_completed_v50", "require_completed_v51"),
    ('mean_audit["columns"] != 3369', 'mean_audit["columns"] != EXPECTED_MEAN_COLUMNS'),
    ('mean_audit["adverse_teacher_mean_tasks"] != 10', 'mean_audit["adverse_teacher_mean_tasks"] != EXPECTED_MEAN_ADVERSE'),
    ('mean_audit["unchanged_structured_tasks"] != 24', 'mean_audit["structured_reweighted_tasks"] != 24\n'
     '            or mean_audit["imputation_reweighted_tasks"] != 26'),
):
    PROBE_EVIDENCE_SOURCE = replace_once(PROBE_EVIDENCE_SOURCE, old, new)
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, "exec"), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (("1517751498", "1248288852"), ("'canonical_artifacts_checked'] != 8", "'canonical_artifacts_checked'] != 2"),
        ("eight-file preservation", "two-file preservation")):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, "exec"), globals())


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_all_imputation_column_mean_trial.py", '
        'REPO / "tests/test_single_model_all_imputation_column_mean_full_audit.py", '
        'REPO / "tests/test_single_model_all_imputation_column_mean_pipeline.py", '
        'REPO / "tests/test_single_model_all_imputation_column_mean_training.py", '
        'REPO / "tests/test_single_model_all_imputation_column_mean_storage.py", '
        'REPO / "tests/test_single_model_all_imputation_column_mean_probe_recheck.py", '
        'REPO / "scripts/recheck_single_model_all_imputation_column_mean_probe.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    return replace_once(source, "import single_model_native_parent_storage as diagnostics",
        "import single_model_all_imputation_column_mean_storage as diagnostics")
