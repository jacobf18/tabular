"""Qualify one component-balanced student under the complete frozen protocol."""

import single_model_all_imputation_column_mean_trial as predecessor
import single_model_balanced_structured_storage as component_storage
import single_model_balanced_structured_components as components
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
POLICY = (
    'A separately registered V53 trial initializes only from unchanged actual V29 step25/98 '
    'inherited updates, never V52 or a feasibility endpoint. After fully audited negative V52 '
    'closure and complete independently audited fresh feasibility, change only the prospective '
    'structured component schedule. Each batch has six tasks and25 percent of structured '
    'supervised mass per original component, across every original family, with all four '
    'components per family over two consecutive scheduled attempts. This is scheduled balance, '
    'not balanced admissions, accepted updates or gradient norms. Preserve native rehearsal, '
    'ratings, all28 task slots, requested geometry/rates and all other specifications; actual '
    'latent-versus-spectral geometry and masks may change. Keep all-imputation count-shrunk '
    'equal-column mean20 Huber, delta1, one unit-variance pseudocount,1e-6 floor, detached target '
    'moments and every query. Native retention remains actual V29; structured retention is '
    'the exact FP32 half-sum of V8/V29; ratings have none. The probe improves30 grouped point '
    'checks but worsens two of56 individual metrics,20/26 retention penalties and10/28 teacher-'
    'mean terms. Preserve these adverse results; feasibility is not held-out or real improvement. '
    'Keep NLL/CDF/query/point terms, observed-blend imputation/specialist rating teachers, '
    '.5/.25/.25 domain masses, weight10 retention, normalization and inference unchanged. '
    'Fresh5e-6 AdamW, fixed-half displacement,.1 inward margin,84 raw gradients,30 grouped '
    'protections,2x norm cap and exact rollback. Attempt64 fresh batches without replacement; '
    'retain boundaries0/10/25/50/64, all22 original development guards, strict minimax/earlier '
    'ties/ineligible0 and sole selection. Require284 genuinely fresh confirmation cases/all32 '
    'original comparisons before every frozen MissBench/UCI/four-RecSys/matched-LLM suite '
    'and full480-comparison review including392 required and18 primary comparisons. Replay '
    'every original task, teacher/component and retention reference, up to3200 components '
    'and1664 targets. One ordinary77,998,496-parameter network with16 frozen buffers, without '
    'routing or ensembling at inference. Preserve18/6/50 GiB reserves and2432 MiB/16 MiB/48 GiB '
    'checkpoint/root/scratch budgets, without compression/rejection assumptions. Keep24 BLAS/4 '
    'Torch threads and.9 allocator cap. Scratch is machine-local, not external backup. CUDA-'
    'hidden coordinators, GPU task replay and separate CPU saved-grid audits retain their roles. '
    'No historical full-autograd/optimizer-trajectory claim, real-data selection, seed reuse, '
    'scientific retry, family exception, guard relaxation or aggregate-only success.'
)
PROBE_HASHES = {
    'config.json': 'a5d077f51bffbaa44f1f577b19f61f0f7a53864b581d29e8993a3c84fcfa26ed',
    'result.json': '34155c950f482f7e7697e0247b3833524b3d759c27db2bbf388ad73134b8c8aa',
    'audit.json': '2e48409db5dc031221dec51b83a56f119024f944fab84e9ad17f2a122830581f',
    'gate.json': 'db670962fe6c6258fc2b78ba2448ad0948b147ed9c949685c7b423b7ed9b7303',
    'retention_audit.json': '733165692eeff8cea6a11ed1a3184a2440597a6f5dc3bb0fa25578e62d9a64c1',
    'OUTCOME.md': 'a5b68b00198f5d88872ff3fef40a65a737daf5e25c3bd94a3cce0d4269d4cb76',
    'external_recheck.json': '0844b4bc3885ae196d44217946b0d56be56fa5b4f752e18ac65197097b475216',
}
MEAN_AUDIT_SHA256 = '8c5996deec9522ed5a0ee6b612356899e5daed2d5788170b9606ed870f4b34aa'
EXPECTED_MEAN_COLUMNS = 4039
EXPECTED_MEAN_ADVERSE = 10
PROBE_EVIDENCE_SOURCE = replace_once(predecessor.PROBE_EVIDENCE_SOURCE, 'require_completed_v51', 'require_completed_v52')
PROBE_EVIDENCE_SOURCE = replace_once(PROBE_EVIDENCE_SOURCE,
    'or config["mean_distillation"] != probe.stronger.SPECIFICATION',
    'or config["structured_component_balance"] != components.SPECIFICATION\n'
    '            or config["batch"] != probe.batch()\n'
    '            or config["mean_distillation"] != probe.stronger.SPECIFICATION')
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, 'exec'), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (('1248288852', '1348269191'), ("'canonical_artifacts_checked'] != 2", "'canonical_artifacts_checked'] != 3"),
        ('two-file preservation', 'three-file preservation')):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, 'exec'), globals())


def adapt_training(source):
    source = predecessor.adapt_training(source)
    return replace_once(source, 'or config["native_teacher_mean"] != retention.MEAN_SPECIFICATION',
        'or config["native_teacher_mean"] != retention.MEAN_SPECIFICATION\n'
        '            or config["structured_component_balance"] != components.SPECIFICATION')


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "structured_component_balance": components.SPECIFICATION,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_balanced_structured_trial.py", '
        'REPO / "tests/test_single_model_balanced_structured_full_audit.py", '
        'REPO / "tests/test_single_model_balanced_structured_pipeline.py", '
        'REPO / "tests/test_single_model_balanced_structured_storage.py", '
        'REPO / "tests/test_single_model_balanced_structured_components.py", '
        'REPO / "tests/test_single_model_balanced_structured_components_probe_recheck.py", '
        'REPO / "tests/test_single_model_balanced_structured_capacity.py", '
        'REPO / "scripts/recheck_single_model_balanced_structured_components_probe.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    return replace_once(source, 'import single_model_all_imputation_column_mean_storage as diagnostics',
        'import single_model_balanced_structured_storage as diagnostics')
