"""Bind declared retention weights to full training and independent reference replay."""

from collections import Counter
import types

import single_model_cohort_anchor_retention_training as previous
import single_model_family_balanced_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = ('Full-trial integration of family-balanced additional retention. ' + cohort_retention.POLICY
    + ' Validate every complete registered supervised task mixture, including data-only rejections. '
    'Keep raw gradient scale1 distinct from the declared mixture weight. Bind each actual weighting '
    'decision, original task weight, full source/mask and V8/V29 reference to the training manifest '
    'and independent reference replay. Qualification checks every row against its registered task '
    'and metadata-only weighting rule. Preserve all original teachers and finite protections. '
    'No inference ensemble, altered supervision, omitted family or historical full-autograd claim. '
    'This helper alone authorizes no trial, seeds or qualification.')


for name in ('reference_identities', 'verify_references'):
    function = getattr(previous, name)
    if function.__closure__ is not None:
        raise ValueError('Do not retain predecessor reference closures')
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

original_identity = types.FunctionType(previous.cohort_identity.__code__, globals(),
    'original_identity', previous.cohort_identity.__defaults__)


def cohort_identity(config):
    identities = original_identity(config)
    for batch in config['batches']:
        retention.validate_task_mixture(batch['tasks'])
    return identities


SOURCE = previous.SOURCE.replace('single_model_cohort_anchor_retention as retention',
    'single_model_family_balanced_retention as retention')
SOURCE = replace_once(SOURCE,
    'or decision["reference_role"] != reference_role',
    'or decision["reference_role"] != reference_role\n'
    '                or decision["weighting"] != retention.retention_weighting(episode.metadata)\n'
    '                or decision["gradient_scale"] != 1.\n'
    '                or task["weight"] != decision["weighting"]["task_weight"]')
SOURCE = replace_once(SOURCE,
    '"reference_role": reference_role, "metadata": dict(episode.metadata)}',
    '"reference_role": reference_role, "metadata": dict(episode.metadata),\n'
    '                  "weighting": decision["weighting"], "gradient_scale": decision["gradient_scale"],\n'
    '                  "task_weight": task["weight"]}')
SOURCE = replace_once(SOURCE,
    'cached["reference_role"] != reference_role',
    'cached["reference_role"] != reference_role or cached["weighting"] != decision["weighting"]')
SOURCE = replace_once(SOURCE,
    '    if (report["complete"] is not True or report["policy"] != POLICY',
    '    registered_weights = {str(training.OUTPUT / "training_batches" / f"step_{batch[\'step\']:03d}"\n'
    '        / f"task_{task[\'index\']:02d}.pt"): task["weight"]\n'
    '        for batch in config["batches"] if batch["step"] in counts["admitted_steps"] for task in batch["tasks"]}\n'
    '    if (report["complete"] is not True or report["policy"] != POLICY')
SOURCE = replace_once(SOURCE,
    'or row["eligible"] is not (row["reference_role"] is not None) for row in rows)',
    'or row["eligible"] is not (row["reference_role"] is not None)\n'
    '                or row["weighting"] != retention.retention_weighting(row["metadata"])\n'
    '                or row["gradient_scale"] != 1.\n'
    '                or row["task_weight"] != row["weighting"]["task_weight"]\n'
    '                or row["task_weight"] != registered_weights[row["task"]] for row in rows)')
SOURCE = replace_assignment(SOURCE, 'POLICY', POLICY_TEXT)
exec(compile(SOURCE, __file__, 'exec'), globals())
