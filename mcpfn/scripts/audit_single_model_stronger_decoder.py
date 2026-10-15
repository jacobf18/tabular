"""Independently audit the declared stronger-rate trial without changing V8 retention."""

import audit_single_model_targeted_decoder as template
from single_model_registered_adaptation import install


install(template, globals(), {
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_stronger_decoder as training',
    'training.learning_rate(step, 100, 1e-5, 5)': 'training.constant_rate(step)',
    'Never replace the full hundred-attempt qualification audit with a partial trial': 'Never replace the full64-attempt qualification audit with a partial trial',
    'The registered first-fifty original learning rates changed': 'The registered constant2e-5 learning rate changed',
    'The V15 initialization': 'The V16 initialization',
}, {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-step", type=int, default=training.STEPS)
    arguments = parser.parse_args()
    audit(arguments.output.resolve(), arguments.through_step)
