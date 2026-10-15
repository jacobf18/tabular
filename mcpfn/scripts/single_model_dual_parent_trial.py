"""Qualify one fixed dual-parent student under the unchanged complete protocol."""

import single_model_parent_structured_trial as predecessor
import single_model_dual_parent_storage as component_storage
from single_model_registered_adaptation import replace_once


ARITHMETIC = predecessor.ARITHMETIC
reference_bytes = predecessor.reference_bytes
wrap_cli = predecessor.wrap_cli
adapt_training = predecessor.adapt_training
POLICY = (
    "A separately registered V50 trial initializes only from unchanged actual V29 step25/98 "
    "inherited updates, never V49 or a probe. After V49's complete audited negative closure, "
    "284-case actual-initializer attribution and independently audited fresh feasibility probe, "
    "change only structured training retention targets to the fixed FP32 half-sum of original "
    "V8 and V29 predictions, uniformly across every family/component. Native retention stays "
    "general; ratings have no retention. V49 inherited both MAR and skip-logic failures and "
    "worsened skip-logic; this attribution is not causal evidence or fresh qualification. The "
    "probe improves30 grouped protections but worsens two of56 individual metrics,22/26 "
    "retention penalties and11/28 teacher-mean terms; preserve every adverse result. Keep native "
    "count-shrunk mean20 Huber, NLL/CDF/query/point terms, observed-blend imputation/specialist "
    "rating teachers, all priors/.5/.25/.25 masses, weight10 count-shrunk retention and original "
    "normalization. Use fresh5e-6 AdamW, fixed-half displacement,.1 margin,84 raw gradients,30 "
    "grouped protections,2x norm cap and exact rollback. Attempt64 fresh batches with no "
    "replacement; retain boundaries0/10/25/50/64, all22 original V8 development guards, strict "
    "minimax/earlier ties/ineligible0 and sole selection. Require284 genuinely fresh confirmation "
    "cases and all32 original V8/general comparisons before every frozen real suite/full review. "
    "Independently replay every original task, teacher/component and reference, up to3200 "
    "retention components and1664 final targets. Keep one ordinary77,998,496-parameter network "
    "and16 frozen buffers at inference without routing or ensembling. Preserve18/6/50 GiB "
    "reserves and2432 MiB/16 MiB checkpoint/root allowances. Prospectively budget48 GiB scratch "
    "rather than40 GiB to preserve all additional component archives and independent replays, "
    "without compression or rejection assumptions. Keep24 BLAS/4 Torch threads and.9 allocator cap. "
    "Scratch is machine-local, not external backup. CUDA-hidden coordinators, GPU full-task "
    "auditors and separate CPU saved-grid audits keep original roles. No historical full-autograd "
    "or optimizer-trajectory replay claim, real-data selection, seed reuse, scientific retry, "
    "family exception, guard relaxation or aggregate-only success."
)
PROBE_HASHES = {
    "config.json": "040ee68892038461bc2c3f31492dec9ebdced9e03e0b285dcc98500dd282f9ac",
    "result.json": "bcb6c2014fdc248aba161aaba91febf2854046474359e21d8b6d97fa6ae1657d",
    "audit.json": "6284000dd723ce554209a8d92e82c352c0faec84824a5b50afec5ec34c6f2e23",
    "gate.json": "30bf0ebf43d3ac10f120af16583f252e5b6a06be606f8c7526cb75780459481e",
    "retention_audit.json": "c2b3971bf282e74b5fcff3b515d87c43d089c72cd6c641e2c586b2a3c7e260c5",
    "OUTCOME.md": "01dfa0a77541e50b13e9c2adedf688828440f361ebbc66e111a6581e6e81857f",
    "external_recheck.json": "27c039c979a0029da96a8010b9649d15425323674c76358715bc413a0ed92cb6",
}
MEAN_AUDIT_SHA256 = "0cf9f2a7a58820312e5b37d58421b689cd2a33dd18b1434e94e7ba6dd9d80b9f"
PROBE_EVIDENCE_SOURCE = predecessor.PROBE_EVIDENCE_SOURCE
for old, new in (
    ("require_completed_v48", "require_completed_v49"),
    ('{"general": 2, "v29": 24}', '{"general": 2, "structured_mean": 24}'),
    ('mean_audit["columns"] != 3505', 'mean_audit["columns"] != 3444'),
    ('mean_audit["adverse_teacher_mean_tasks"] != 10', 'mean_audit["adverse_teacher_mean_tasks"] != 11'),
    ('or retention["reference_grids"] != 26',
     'or retention["component_grids"] != 50\n'
     '            or retention["component_role_counts"] != {"general": 2, "v8": 24, "v29": 24}\n'
     '            or retention["reference_grids"] != 26'),
):
    PROBE_EVIDENCE_SOURCE = replace_once(PROBE_EVIDENCE_SOURCE, old, new)
exec(compile(PROBE_EVIDENCE_SOURCE, __file__, "exec"), globals())
STORAGE_EVIDENCE_SOURCE = predecessor.STORAGE_EVIDENCE_SOURCE
for old, new in (("1922740672", "1475237654"), ("'canonical_artifacts_checked'] != 5", "'canonical_artifacts_checked'] != 4"),
                 ("five-file preservation", "four-file preservation")):
    STORAGE_EVIDENCE_SOURCE = replace_once(STORAGE_EVIDENCE_SOURCE, old, new)
exec(compile(STORAGE_EVIDENCE_SOURCE, __file__, "exec"), globals())


def adapt_registration(source):
    source = predecessor.adapt_registration(source)
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_dual_parent_trial.py", '
        'REPO / "tests/test_single_model_dual_parent_full_audit.py", '
        'REPO / "tests/test_single_model_dual_parent_pipeline.py", '
        'REPO / "tests/test_single_model_dual_parent_training.py", '
        'REPO / "tests/test_single_model_dual_parent_storage.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = predecessor.adapt_recipe(source)
    source = replace_once(source, "import single_model_all_imputation_storage as diagnostics",
        "import single_model_dual_parent_storage as diagnostics")
    return replace_once(source,
        'STORAGE_ACCOUNTING["scratch"]["all_imputation_reference_allowance"] = all_trial.reference_bytes(previous_training.batches())',
        'STORAGE_ACCOUNTING["scratch"]["all_imputation_reference_allowance"] = all_trial.reference_bytes(previous_training.batches())\n'
        'STORAGE_ACCOUNTING["scratch"]["dual_parent_component_allowance"] = all_trial.component_storage.component_bytes(previous_training.batches())')
