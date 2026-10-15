"""Qualify one native-parent student under the unchanged complete protocol."""

import single_model_dual_parent_trial as predecessor
import single_model_native_parent_storage as component_storage
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
adapt_training = predecessor.adapt_training
POLICY = (
    "A separately registered V51 trial initializes only from unchanged actual V29 step25/98 "
    "inherited updates, never V50 or a probe. After V50's complete audited negative closure "
    "and an independently audited fresh feasibility probe, change only native training "
    "retention from general to frozen actual V29. Structured retention remains the fixed "
    "FP32 half-sum of original V8 and V29; ratings have no retention. The probe improves30 "
    "grouped protections but worsens two of56 individual metrics,21/26 retention penalties "
    "and10/28 teacher-mean terms; preserve every adverse result. Keep native count-shrunk "
    "mean20 Huber, NLL/CDF/query/point terms, observed-blend imputation/specialist rating "
    "teachers, all priors/.5/.25/.25 masses, weight10 count-shrunk retention and original "
    "normalization. Verify all three frozen reference identities but perform no general "
    "retention inference. Use fresh5e-6 AdamW, fixed-half displacement,.1 margin,84 raw "
    "gradients,30 grouped protections,2x norm cap and exact rollback. Attempt64 fresh batches "
    "with no replacement; retain boundaries0/10/25/50/64, all22 original V8 development "
    "guards, strict minimax/earlier ties/ineligible0 and sole selection. Require284 genuinely "
    "fresh confirmation cases and all32 original V8/general comparisons before every frozen "
    "real suite/full review. Independently replay every original task, teacher/component "
    "and reference, up to3200 retention components and1664 final targets. Keep one ordinary "
    "77,998,496-parameter network and16 frozen buffers at inference without routing or "
    "ensembling. Preserve18/6/50 GiB reserves,2432 MiB/16 MiB checkpoint/root allowances "
    "and the complete48 GiB scratch allowance without compression or rejection assumptions. "
    "Keep24 BLAS/4 Torch threads and.9 allocator cap. Scratch is machine-local, not external "
    "backup. CUDA-hidden coordinators, GPU full-task auditors and separate CPU saved-grid "
    "audits keep original roles. No historical full-autograd or optimizer-trajectory replay "
    "claim, real-data selection, seed reuse, scientific retry, family exception, guard "
    "relaxation or aggregate-only success."
)
PROBE_HASHES = {
    "config.json": "fc04a4288d5dbbbf925ccb3dc6c1d81bdc7b23c6d528a949fb9370040d681cb3",
    "result.json": "fc7e82f923ce27f69cea9ffcf724b81569ea95b62f761adf6c3b6b9b56fee69c",
    "audit.json": "6783e75029c160f8d5e93c606787b3de9e347346f7978d1cd0c6e6f5c9ef68a2",
    "gate.json": "84cd4d13b77620a77c843e56981759bd42c8e43b1f73329096933a7d4eb3d6da",
    "retention_audit.json": "1b2a5fb65d38890ec75db5ce0c658f056f548a20a247acca4e41f917fd5280f8",
    "OUTCOME.md": "a8edcf0d35ecece7124f5ddf0f16e524948096a963804a3079c7c705b7411203",
    "external_recheck.json": "abb6a10d23bad5e0690ffa2083b5f352d18f2ae5e203abeefa4b543d6f4e318f",
}
MEAN_AUDIT_SHA256 = "89df1dde232472e5111d4a09da64b3ed03e796eb9949f2af6975e6c4093576a7"
PROBE_EVIDENCE_SOURCE = predecessor.PROBE_EVIDENCE_SOURCE
for old, new in (
    ("require_completed_v49", "require_completed_v50"),
    ('{"general": 2, "structured_mean": 24}', '{"v29": 2, "structured_mean": 24}'),
    ('{"general": 2, "v8": 24, "v29": 24}', '{"v8": 24, "v29": 26}'),
    ('mean_audit["columns"] != 3444', 'mean_audit["columns"] != 3369'),
    ('mean_audit["adverse_teacher_mean_tasks"] != 11', 'mean_audit["adverse_teacher_mean_tasks"] != 10'),
):
    PROBE_EVIDENCE_SOURCE = replace_once(PROBE_EVIDENCE_SOURCE, old, new)
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, "exec"), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (("1475237654", "1517751498"), ("'canonical_artifacts_checked'] != 4", "'canonical_artifacts_checked'] != 8"),
        ("four-file preservation", "eight-file preservation")):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, "exec"), globals())


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_native_parent_trial.py", '
        'REPO / "tests/test_single_model_native_parent_full_audit.py", '
        'REPO / "tests/test_single_model_native_parent_pipeline.py", '
        'REPO / "tests/test_single_model_native_parent_training.py", '
        'REPO / "tests/test_single_model_native_parent_storage.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    return replace_once(source, "import single_model_dual_parent_storage as diagnostics",
        "import single_model_native_parent_storage as diagnostics")
