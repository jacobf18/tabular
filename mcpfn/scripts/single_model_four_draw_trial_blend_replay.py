"""Preserve every teacher component for the complete four-draw training schedule."""

import single_model_balanced_native_long_blend_replay as template
from single_model_registered_adaptation import replace_assignment, replace_once


TRIAL_POLICY = (
    "Prospective full-trial teacher audit for128 four-draw56-task batches. Independently "
    "replay both frozen imputation teachers for every admitted native or structured task, "
    "refit the original observed-only blend and reproduce the complete cached target. Keep "
    "all52 imputation targets/104 teacher component grids and four specialist-rating tasks "
    "per admitted batch. Reopen every original source, component and target archive on CPU "
    "with the original arithmetic and full payload identities. Data-only rejected attempts "
    "must remain independently audited, not replaced or interpreted as inference. Retain "
    "all original qualification checks and no historical full-autograd claim. This module "
    "authorizes no training, seed reservation, validation, selection or inference ensemble."
)
SOURCE = replace_assignment(template.SOURCE, "POLICY", TRIAL_POLICY)
for before, after in (
    ('len(batches) != 256', 'len(batches) != 128'),
    ('Budget every registered256 attempt before training', 'Budget every registered128 attempt before training'),
    ('len(batch["tasks"]) != 28', 'len(batch["tasks"]) != 56'),
    ('Preserve every original28-task batch', 'Preserve every registered56-task batch'),
    ('len(selected) != 26', 'len(selected) != 52'),
    ('Budget all26 imputation draws', 'Budget all52 imputation draws'),
    ('step not in range(1, 257)', 'step not in range(1, 129)'),
    ('counts["training_tasks"] != 28 * len(steps)', 'counts["training_tasks"] != 56 * len(steps)'),
    ('list(range(28))', 'list(range(56))'),
    ('blend_tasks != 26 * len(counts["admitted_steps"])', 'blend_tasks != 52 * len(counts["admitted_steps"])'),
):
    SOURCE = replace_once(SOURCE, before, after)
exec(compile(SOURCE, __file__, "exec"), globals())
