"""Qualify one standardized-Huber student under the complete frozen protocol."""

import single_model_balanced_structured_trial as predecessor
import single_model_standardized_huber_storage as component_storage
import single_model_balanced_structured_components as components
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
adapt_training = predecessor.adapt_training
POLICY = (
    'A separately registered V54 trial initializes only from actual V29, never V53 or the '
    'feasibility endpoint. Bind V53 audited negative closure and both preserved workflow '
    'failures. After complete fresh feasibility/audits, change only all-imputation teacher-'
    'mean geometry to equal-column Huber of count-shrunk variance-standardized residuals, '
    'delta1/weight20, one unit-variance pseudocount/1e-6 floor/detached target moments. '
    'Keep every query, including singleton/constant columns, and balanced structured '
    'component assignments. Preserve all30 grouped probe improvements, both adverse '
    'polarization point metrics,16/26 adverse retention penalties and8/28 adverse mean terms. '
    'These are feasibility results, not generalization evidence. Native V29 retention, '
    'fixed FP32 V8/V29 structured retention, no rating retention, observed-blend imputation '
    'teachers/specialist rating teacher, NLL/CDF/query/point/rating objectives, .5/.25/.25 '
    'domain masses, normalization and inference stay unchanged. Fresh5e-6 AdamW, fixed-half '
    'displacement,.1 margin,84 raw gradients,30 grouped protections,2x norm cap/exact rollback. '
    'Attempt64 fresh batches without replacement; boundaries0/10/25/50/64; all22 original '
    'development guards, strict minimax/earlier ties/ineligible0 and sole selection. Require '
    '284 fresh confirmation cases/all32 comparisons before frozen MissBench/UCI/four-RecSys/'
    'matched-LLM suites and full480-comparison review including392 required/18 primary. '
    'Replay every original task/teacher/component/reference, up to3200 retention components/'
    '1664 targets. One ordinary77,998,496-parameter network,16 frozen buffers, no inference '
    'routing or ensembling. Preserve18/6/50 GiB reserves and2432 MiB/16 MiB/48 GiB checkpoint/'
    'root/scratch allowances,24 BLAS/4 Torch threads/.9 allocator cap. CUDA-hidden coordinators, '
    'GPU task replay/separate CPU saved-grid audits. Scratch is machine-local, not external '
    'backup. No full historical autograd claim, seed reuse, scientific retry, real-data '
    'selection, family exception, guard relaxation or aggregate-only success.'
)
PROBE_HASHES = {
    'config.json': '7bf3acc2e92274eda8e5254546913f49353f1700091292d78b4c0ed93a0641d8',
    'result.json': '0230984e9e069aa13742e4eac374c06a77cc7de8cdd78fb68fc37dba13544f38',
    'audit.json': '7aae4dda4aab5256dfa882bb0e4647d3245c0fde8bb23d784dd10c9d4cdb7842',
    'gate.json': 'f07471751849e83a8d5b7d4e7fa85e62b8524fa1d0c060eecdb728cb916a83b2',
    'retention_audit.json': '6e9d2fcc408542bd2ba7ee0dcaae0db84bc2ff7b44becc19e480079936ac2bdc',
    'OUTCOME.md': '8f1edec3f192b843c8af1aab4b0e2b1b75977c3b09ff5b7119cc98f34bda4432',
    'external_recheck.json': 'e498f61b88642938b5f332411d42aa198320188c5ae1652dec50cec3749b9baf',
}
MEAN_AUDIT_SHA256 = '82cb9584a65e25911111c5ebdd48b2a947561c971cac3274507566931ba0f04a'
EXPECTED_MEAN_COLUMNS = 4053
EXPECTED_MEAN_ADVERSE = 8
PROBE_EVIDENCE_SOURCE = replace_once(predecessor.PROBE_EVIDENCE_SOURCE, 'require_completed_v52', 'require_completed_v53')
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, 'exec'), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (('1348269191', '1872586390'), ("'canonical_artifacts_checked'] != 3", "'canonical_artifacts_checked'] != 2"),
        ('three-file preservation', 'two-file preservation')):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, 'exec'), globals())


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_standardized_huber_trial.py", '
        'REPO / "tests/test_single_model_standardized_huber_full_audit.py", '
        'REPO / "tests/test_single_model_standardized_huber_pipeline.py", '
        'REPO / "tests/test_single_model_standardized_huber_training.py", '
        'REPO / "tests/test_single_model_standardized_huber_storage.py", '
        'REPO / "tests/test_single_model_standardized_teacher_huber_probe_recheck.py", '
        'REPO / "tests/test_single_model_standardized_huber_capacity.py", '
        'REPO / "scripts/recheck_single_model_standardized_teacher_huber_probe.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    return replace_once(source, 'import single_model_balanced_structured_storage as diagnostics',
        'import single_model_standardized_huber_storage as diagnostics')
