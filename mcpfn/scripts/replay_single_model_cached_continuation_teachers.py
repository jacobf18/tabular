"""Replay all original V22 teacher targets for the separately reconstructed candidate."""

import fcntl
import hashlib

import replay_single_model_continuation_teachers as baseline
from single_model_registered_adaptation import replace_once


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_teacher_continuation as training':
        'import recover_single_model_cached_continuation as training',
    'OUTPUT = training.ROOT / "teacher_continuation_v22_teacher_replay"':
        'OUTPUT = training.ROOT / "teacher_continuation_v22_cached_recovery_v1_teacher_replay"',
    '        path = training.OUTPUT / "training_batches" / f\'step_{batch["step"]:03d}\' / "manifest.json"':
        '        path = training.original_manifest(batch["step"])',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


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
