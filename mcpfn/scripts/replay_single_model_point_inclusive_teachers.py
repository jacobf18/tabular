"""Independently replay all1792 frozen teacher targets for the point-inclusive trial."""

import replay_single_model_training_teachers as baseline
from single_model_registered_adaptation import install


install(baseline, globals(), {
    'import train_single_matrixformer_structured_teacher as training': 'import train_single_matrixformer_point_inclusive as training',
    'OUTPUT = training.ROOT / "structured_teacher_v20_teacher_replay"': 'OUTPUT = training.ROOT / "point_inclusive_v21_teacher_replay"',
}, {})


if __name__ == "__main__":
    main()
