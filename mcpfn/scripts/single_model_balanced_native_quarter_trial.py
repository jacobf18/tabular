"""Qualify one native-balanced quarter-step checkpoint under the complete frozen protocol."""

import single_model_standardized_huber_trial as predecessor
import single_model_balanced_native_quarter_storage as component_storage
import single_model_balanced_native_coverage as native
import single_model_balanced_structured_components as components
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
POLICY = (
    'A separately registered V55 trial initializes only from unchanged actual V29, never V54 '
    'or a feasibility endpoint. After complete negative V54/half-probe closure, accepted '
    'fresh quarter feasibility and verified byte-preserving capacity preparation, change '
    'the native component/orientation schedule and fixed projected contraction0.5 to0.25. '
    'Preserve the complete balanced structured schedule, requested shapes,28 task slots, '
    '.5/.25/.25 native/structured/rating masses and all other specifications. Native balance '
    'covers all32 component/requested-shape/orientation strata once per16 scheduled attempts; '
    'it does not assert balanced actual intrinsic shapes, admissions or accepted gradients. '
    'Keep standardized equal-column teacher Huber20, count-shrunk variance/1e-6 floor/delta1, '
    'every query and all NLL/CDF/point/rating terms. Observed-only blended imputation teachers, '
    'specialist rating teacher, native V29 retention, fixed FP32 half V8/V29 structured '
    'retention and no rating retention remain unchanged. Half retention weights are distinct '
    'from quarter optimizer displacement. Preserve30/30 grouped and56/56 individual probe '
    'point improvements,12/28 adverse mean terms and19/26 adverse retention terms; this '
    'is feasibility, not generalization. Fresh5e-6 AdamW/original accepted moments,.1 inward '
    'margin,84 raw gradients,30 grouped protections,2x norm cap and exact rollback. Attempt64 '
    'fresh batches without replacement, boundaries0/10/25/50/64, all22 original development '
    'guards, strict minimax/earlier ties/ineligible0 and sole checkpoint selection. Require '
    '284 fresh confirmation cases/all32 comparisons before frozen MissBench/UCI/four-RecSys/'
    'matched-LLM suites and the full480-comparison review including392 required/18 primary. '
    'Replay every original task/teacher/reference/component, up to3200 retention components/'
    '1664 targets; independently reconstruct the terminal actual quarter candidate on GPU, '
    'including after rejection, followed by CPU saved-grid audit. One ordinary77,998,496-'
    'parameter network,238 trainable tensors,16 frozen buffers and unchanged inference; '
    'no routing or ensemble. Preserve18/6/50 GiB reserves and2432 MiB/16 MiB/48 GiB training '
    'allowances,24 BLAS/4 Torch threads/.9 allocator cap. CUDA-hidden coordinators and CPU '
    'audits, GPU-visible original training/task replay. Scratch is machine-local, not external '
    'backup. No historical full-autograd claim, seed reuse, scientific retry, real-data '
    'selection, family exception, relaxed guard or aggregate-only success.'
)
PROBE_HASHES = {
    'config.json': '9b8418c5e0ee88fb64b9a163efa31594927fccc386419986cb0fd0e6f0c859e4',
    'result.json': '25ae2d701e016d3334954324c2cbd314e19163a0025aad8cd3576d9bddc63626',
    'audit.json': '41ba96e6f6de3abd2cbbae2bb6e4bfb2fe8c4aaee359f79f0cc3f28aa82f5f3d',
    'gate.json': 'cd4e020d3ee80cf294161bd11eb00d7d1f11e7b9cd972049b5bccdd8422b5296',
    'retention_audit.json': '6c5f938d00eda2252b82339872b003935b8fd600ba9ec4bf0b29cc789ca46163',
    'OUTCOME.md': '453b2db16b63c26ff4e908412b5e45c9b83df46ba5983b5157ff5a50eb7c1bd3',
    'external_recheck.json': 'af6847f754424aba0f108ae5f3b0599c11e88132d4153ab9c13b8241add9f6ee',
}
MEAN_AUDIT_SHA256 = 'f104c7952c3081c66b6c12af32918c316da5a05c761a4c50149c63025d856f12'
EXPECTED_MEAN_COLUMNS = 6870
EXPECTED_MEAN_ADVERSE = 12
PROBE_EVIDENCE_SOURCE = replace_once(predecessor.PROBE_EVIDENCE_SOURCE,
    'require_completed_v53', 'require_completed_native_probe')
PROBE_EVIDENCE_SOURCE = replace_once(PROBE_EVIDENCE_SOURCE,
    'or config["structured_component_balance"] != components.SPECIFICATION',
    'or config["native_coverage"] != native.SPECIFICATION\n'
    '            or config["native_probe_shape"] != probe.NATIVE_SHAPE\n'
    '            or config["displacement_fraction"] != .25\n'
    '            or config["structured_component_balance"] != components.SPECIFICATION')
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, 'exec'), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (('1872586390', '3744936228'), ("'canonical_artifacts_checked'] != 2", "'canonical_artifacts_checked'] != 12"),
        ('two-file preservation', 'twelve-file preservation')):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, 'exec'), globals())
original_storage_evidence = storage_evidence


def storage_evidence(capacity, receipt_sha256):
    evidence = original_storage_evidence(capacity, receipt_sha256)
    failure = capacity.copying.SCRATCH / 'balanced_native_quarter_capacity_relocation_v1_preserve.log'
    evidence[str(failure)] = '86c251f144bc5beefe953cb38e2cbcb6206af5d9ee508c3c289f089511d8fb3d'
    capacity.verify_hashes(evidence)
    return evidence


def adapt_training(source):
    source = predecessor.adapt_training(source)
    return replace_once(source, 'or config["structured_component_balance"] != components.SPECIFICATION',
        'or config["structured_component_balance"] != components.SPECIFICATION\n'
        '            or config["native_coverage"] != native.SPECIFICATION')


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "native_coverage": native.SPECIFICATION,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_balanced_native_quarter_trial.py", '
        'REPO / "tests/test_single_model_balanced_native_quarter_full_audit.py", '
        'REPO / "tests/test_single_model_balanced_native_quarter_pipeline.py", '
        'REPO / "tests/test_single_model_balanced_native_quarter_storage.py", '
        'REPO / "tests/test_single_model_balanced_native_quarter_capacity.py", '
        'REPO / "tests/test_single_model_balanced_native_quarter_probe_recheck.py", '
        'REPO / "tests/test_single_model_balanced_native_coverage.py", '
        'REPO / "scripts/recheck_single_model_balanced_native_quarter_probe.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    if source.count('tests/test_single_model_fixed_half_trial.py') != 2:
        raise ValueError('Bind both inherited optimizer-test registration expressions to quarter integration')
    source = source.replace('tests/test_single_model_fixed_half_trial.py', 'tests/test_single_model_fixed_quarter_trial.py')
    return replace_once(source, 'import single_model_standardized_huber_storage as diagnostics',
        'import single_model_balanced_native_quarter_storage as diagnostics')
