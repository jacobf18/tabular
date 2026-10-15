"""Audit all V41 tasks, optimizer evidence, retention references and raw blend components."""

import audit_single_model_all_imputation_retention as previous_audit
from audit_single_model_all_imputation_retention import retention_training, fixed_trial, admission_source
import single_model_legacy_blend_full_replay as blend_replay
from single_model_registered_adaptation import replace_once


INTEGRATION_SOURCE = '''
retention_load_regenerated_task = load_regenerated_task
retention_regeneration_evidence = regeneration_evidence


def load_regenerated_task(path):
    payload = retention_load_regenerated_task(path)
    REPLAY.get()["legacy_blend"].check(path, payload)
    return payload


def regeneration_evidence(through_step):
    original = retention_regeneration_evidence(through_step)
    return {**original, **REPLAY.get()["legacy_blend"].finish()}


RUN_AUDIT_SOURCE = replace_once(RUN_AUDIT_SOURCE, '            token = REPLAY.set(active)',
    '            active["legacy_blend"] = blend_replay.ComponentReplay(training, config, counts, teachers, criterion)\\n'
    '            token = REPLAY.set(active)')
exec(compile(RUN_AUDIT_SOURCE, __file__, "exec"), globals())
'''

SOURCE = previous_audit.SOURCE
for old, new in {
    "train_single_matrixformer_all_imputation_retention": "train_single_matrixformer_legacy_blend_teacher",
    "all_imputation_retention_v40_audit": "legacy_blend_teacher_v41_audit",
    "single_model_all_imputation_retention_audit_replay": "single_model_legacy_blend_teacher_audit_replay",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V41 independent-audit adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = SOURCE.replace("V40", "V41")
SOURCE = replace_once(SOURCE, 'if __name__ == "__main__":', INTEGRATION_SOURCE + '\nif __name__ == "__main__":')
exec(compile(SOURCE, __file__, "exec"), globals())
