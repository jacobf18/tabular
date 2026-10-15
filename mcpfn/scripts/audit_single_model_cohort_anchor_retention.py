"""Audit all V42 tasks, original teachers, cohort references and optimizer evidence."""

import audit_single_model_legacy_blend_teacher as previous_audit
from audit_single_model_legacy_blend_teacher import fixed_trial, admission_source
import single_model_cohort_anchor_retention_training as retention_training
import single_model_legacy_blend_full_replay as blend_replay
import single_model_cohort_anchor_trial as cohort_trial
import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery
from single_model_registered_adaptation import replace_once


SOURCE = previous_audit.SOURCE
for old, new in {
    "train_single_matrixformer_legacy_blend_teacher": "train_single_matrixformer_cohort_anchor_retention",
    "legacy_blend_teacher_v41_audit": "cohort_anchor_retention_v42_audit",
    "single_model_legacy_blend_teacher_audit_replay": "single_model_cohort_anchor_retention_audit_replay",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V42 cohort audit adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
REPLAY_ADAPTATIONS = [
    ('            model.eval().requires_grad_(False)',
        '            model.eval().requires_grad_(False)\n'
        '            references = retention_training.frozen_reference(model, retention_training.identity(config))\n'
        '            del model'),
    ('training, config, counts, model, criterion, active["backend"])',
        'training, config, counts, references, criterion, active["backend"])'),
]
integration = "\n".join(f"RUN_AUDIT_SOURCE = replace_once(RUN_AUDIT_SOURCE, {old!r}, {new!r})"
    for old, new in REPLAY_ADAPTATIONS)
SOURCE = replace_once(SOURCE, '\nif __name__ == "__main__":',
    '\n' + integration + '\nexec(compile(RUN_AUDIT_SOURCE, __file__, "exec"), globals())\n\nif __name__ == "__main__":')
SOURCE = cohort_trial.wrap_cli(SOURCE.replace("V41", "V42"))
exec(compile(SOURCE, __file__, "exec"), globals())
