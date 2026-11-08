"""Preserve full reference replay for128 four-draw training attempts."""

from pathlib import Path

import single_model_standardized_huber_training as template
from single_model_registered_adaptation import replace_assignment, replace_once


TRIAL_POLICY = (
    "Prospective full-trial integration of four-draw standardized teacher Huber and unchanged "
    "per-task retention gradients. Replay every admitted56-task batch's52 reference targets "
    "and100 original components:48 V8 and52 V29. Preserve each component separately, exact "
    "full-grid inference, all eligibility decisions, payload and target identities and "
    "original query masks. Halve supervised and effective retention task weights together; "
    "do not alter individual objectives or teachers. No retention inference for ratings or "
    "data-only rejected batches. Keep all128 scheduled attempts and consume rejections "
    "without replacement. Preserve all held-out and real qualification requirements, "
    "single-network inference and the absence of historical full-autograd claims. This "
    "module authorizes no sampling, seed reservation, checkpoint or training execution."
)


def adapt_source(source):
    for before, after in (
        ('step not in range(1, 65)', 'step not in range(1, 129)'),
        ('Keep the original64 registered optimizer attempts', 'Keep all128 registered four-draw attempts'),
        ('len(manifest["records"]) != 28', 'len(manifest["records"]) != 56'),
        ('len(batch["tasks"]) != 28', 'len(batch["tasks"]) != 56'),
        ('Preserve all28 decisions', 'Preserve all56 decisions'),
        ('for index in range(28)', 'for index in range(56)'),
    ):
        source = replace_once(source, before, after)
    return source


SOURCE = Path(template.__file__).read_text()
if SOURCE.count("single_model_standardized_teacher_huber") != 2:
    raise ValueError("Bind both the retained callback and runtime to the four-draw objective")
SOURCE = SOURCE.replace("single_model_standardized_teacher_huber", "single_model_four_draw_retention")
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", TRIAL_POLICY)
SOURCE = replace_once(SOURCE, "exec(compile(RETENTION_TRAINING_SOURCE, __file__, 'exec'), globals())",
    "RETENTION_TRAINING_SOURCE = adapt_source(RETENTION_TRAINING_SOURCE)\n"
    "exec(compile(RETENTION_TRAINING_SOURCE, __file__, 'exec'), globals())")
exec(compile(SOURCE, __file__, "exec"), globals())


def expected_component_counts(admitted_batches):
    if type(admitted_batches) is not int or not 0 <= admitted_batches <= 128:
        raise ValueError("Require an exact admitted count within the registered128-attempt budget")
    return {role: count * admitted_batches for role, count in (("v8", 48), ("v29", 52)) if admitted_batches}
