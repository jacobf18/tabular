"""Independently audit V37 and every additional frozen native-anchor reference grid."""

import audit_single_model_full_network_cap90_v2 as predecessor
import audit_single_model_legacy_strata as admission_source
import single_model_anchor_retention_training as retention_training
from single_model_registered_adaptation import replace_once


INTEGRATION_SOURCE = '''
original_load_regenerated_task = load_regenerated_task
original_regeneration_evidence = regeneration_evidence


def load_regenerated_task(path):
    payload = original_load_regenerated_task(path)
    active = REPLAY.get()
    active["anchor_retention"].check(path, payload)
    return payload


def regeneration_evidence(through_step):
    original = original_regeneration_evidence(through_step)
    return {**original, **REPLAY.get()["anchor_retention"].finish()}


RUN_AUDIT_SOURCE = admission_source.RUN_AUDIT_SOURCE
RUN_AUDIT_SOURCE = replace_once(RUN_AUDIT_SOURCE, '            del model, optimizer',
    '            del optimizer\\n'
    '            model.load_state_dict(torch.load(training.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"], strict=True)\\n'
    '            model.eval().requires_grad_(False)')
RUN_AUDIT_SOURCE = replace_once(RUN_AUDIT_SOURCE, '            token = REPLAY.set(active)',
    '            active["anchor_retention"] = retention_training.RetentionReplay(\\n'
    '                training, config, counts, model, criterion, active["backend"])\\n'
    '            token = REPLAY.set(active)')
exec(compile(RUN_AUDIT_SOURCE, __file__, "exec"), globals())

'''
SOURCE = predecessor.SOURCE
for old, new in {
    "import train_single_matrixformer_full_network_cap90_v2 as training": "import train_single_matrixformer_anchor_retention as training",
    "full_network_cap90_v36_audit": "anchor_retention_v37_audit",
    "single_model_full_network_cap90_v2_audit_replay": "single_model_anchor_retention_audit_replay",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V37 audit adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'if __name__ == "__main__":', INTEGRATION_SOURCE + '\nif __name__ == "__main__":')
SOURCE = SOURCE.replace("V36", "V37")
exec(compile(SOURCE, __file__, "exec"), globals())
