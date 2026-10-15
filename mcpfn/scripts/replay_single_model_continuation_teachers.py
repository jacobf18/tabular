"""Independently replay all1792 original teacher grids for the V20-start continuation."""

import fcntl

import replay_single_model_training_teachers as baseline
from single_model_registered_adaptation import install


install(baseline, globals(), {
    'import train_single_matrixformer_structured_teacher as training':
        'import train_single_matrixformer_teacher_continuation as training',
    'OUTPUT = training.ROOT / "structured_teacher_v20_teacher_replay"':
        'OUTPUT = training.ROOT / "teacher_continuation_v22_teacher_replay"',
}, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("replay", "audit"), required=True)
    arguments = parser.parse_args()
    with (OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            {"replay": replay, "audit": audit}[arguments.stage](OUTPUT)
        except BaseException as error:
            status(OUTPUT, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
