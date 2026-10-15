"""Audit the full structured-teacher trial without changing any original development gate."""

import hashlib

import audit_single_model_paired_decoder as baseline
from single_model_registered_adaptation import replace_once


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_paired_decoder as training': 'import train_single_matrixformer_structured_teacher as training',
    'point_before.update(check_task(payload, task, logged))':
        'point_before.update(check_task(payload, task, logged))\n'
        '            training.preflight.check_routing(payload, task, config)',
    'The V19 initialization must reproduce the original V8 prediction grids': 'The V20 initialization must reproduce the original V8 prediction grids',
    '"training_episodes": 28 * through_step':
        '"training_episodes": 28 * through_step, "teacher_identity_checked_tasks": 28 * through_step,\n'
        '               "training_teacher_policy": config["teacher_policy"]',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-step", type=int, default=training.STEPS)
    arguments = parser.parse_args()
    audit(arguments.output.resolve(), arguments.through_step)
