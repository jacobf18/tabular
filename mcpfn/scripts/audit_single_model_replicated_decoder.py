"""Audit all doubled training draws while retaining the complete original V8 development gate."""

import audit_single_model_targeted_decoder as template
from single_model_registered_adaptation import install


install(template, globals(), {
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_replicated_decoder as training',
    'training.learning_rate(step, 100, 1e-5, 5)': 'training.constant_rate(step)',
    'Never replace the full hundred-attempt qualification audit with a partial trial': 'Never replace the full64-attempt qualification audit with a partial trial',
    'The registered first-fifty original learning rates changed': 'The registered constant2e-5 learning rate changed',
    'All152 V14 initialization grids are mandatory': 'All152 original V8 initialization grids are mandatory',
    'The V15 initialization must reproduce the V14 final prediction grids': 'The V18 initialization must reproduce the original V8 prediction grids',
    'saved["parent_training_updates"] != 95': 'saved["parent_training_updates"] != 0',
    'parent_report_path = training.INITIAL_RUN / "validation_50.json"': 'parent_report_path = training.INITIAL_RUN / "validation_0.json"',
    'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",)), (parent_report, ("student",))], []':
        'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",))], []',
    '152 * (4 + len(boundaries))': '152 * (3 + len(boundaries))',
    'len(manifest["tasks"]) != 14': 'len(manifest["tasks"]) != 28',
    'Every registered fourteen-task batch is mandatory': 'Every registered twenty-eight-task batch is mandatory',
    '"training_episodes": 14 * through_step': '"training_episodes": 28 * through_step',
    'all{14 * through_step} cached tasks': 'all{28 * through_step} cached tasks',
}, {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-step", type=int, default=training.STEPS)
    arguments = parser.parse_args()
    audit(arguments.output.resolve(), arguments.through_step)
