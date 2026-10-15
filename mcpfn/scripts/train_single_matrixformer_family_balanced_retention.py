"""Train V43 with family-balanced retention and unchanged supervised priors."""

import train_single_matrixformer_cohort_anchor_retention as previous_training
import single_model_fixed_half_trial as fixed_trial
import single_model_family_balanced_trial as all_trial
from single_model_registered_adaptation import replace_once


CAPACITY_RECEIPT_SHA256 = '43fc295406bb622ad06cc8c92ba021a045755c87f1be70c2eb80ff7dc16d64cb'
ADAPTATION_SOURCE = previous_training.ADAPTATION_SOURCE
for old, new in {
    'cohort_anchor_retention': 'family_balanced_retention',
    'relocate_single_model_cohort_capacity': 'relocate_single_model_family_balanced_capacity',
    '1061100901, 1062100901': '1091100901, 1092100901',
    '1071100901, 1072100901': '1101100901, 1102100901',
    'receipt["canonical_artifacts_checked"] != 7': 'receipt["canonical_artifacts_checked"] != 3',
    'seven-file preservation receipt': 'three-file preservation receipt',
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f'Missing explicit family-balanced full-trial adaptation: {old}')
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace('family_balanced_retention_v42', 'family_balanced_retention_v43')
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace('V42', 'V43')
BATCH_WRAPPER = '\nunvalidated_batches = batches\n\ndef batches():\n    result = unvalidated_batches()\n'\
    '    for batch in result:\n        retention.validate_task_mixture(batch["tasks"])\n    return result\n'
ADAPTATION_SOURCE = replace_once(ADAPTATION_SOURCE, 'exec(compile(SOURCE, __file__, "exec"), globals())',
    'SOURCE = replace_once(SOURCE, \'\\nif __name__ == "__main__":\', BATCH_WRAPPER + \'\\nif __name__ == "__main__":\')\n'
    'exec(compile(SOURCE, __file__, "exec"), globals())')
exec(compile(ADAPTATION_SOURCE, __file__, 'exec'), globals())
