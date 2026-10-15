"""Audit reconstructed V22 evidence without altering its original science or failed run."""

import hashlib

import audit_single_model_teacher_continuation as baseline
import single_model_composite_gradient_storage as storage
import single_model_shared_tensor_storage as shared_storage
from single_model_registered_adaptation import replace_once


check_gradient_alias = storage.check_alias
ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_teacher_continuation as training':
        'import recover_single_model_cached_continuation as training',
    '    config = training.verify_registration(run)':
        '    recovery_hashes = training.audit_recovery()\n    config = training.verify_registration(run)',
    '        manifest_path = run / "training_batches" / f"step_{step:03d}" / "manifest.json"':
        '        manifest_path = training.original_manifest(step)',
    '    accepted, previous_rate = 0, config["base"]["peak_lr"]':
        '    hashes.update(recovery_hashes)\n    accepted, previous_rate = 0, config["base"]["peak_lr"]',
    '"autograd_replayed": False, "full_optimizer_trajectory_replayed": False,':
        '"autograd_replayed": False, "full_optimizer_trajectory_replayed": False,\n'
        '               "cached_reconstruction": training.read(run / "recovery_result.json"),',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.AUDIT)
    arguments = parser.parse_args()
    audit(arguments.output.resolve(), training.STEPS)
