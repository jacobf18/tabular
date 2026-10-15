"""Audit every point-inclusive attempt and the full shared-storage decoder endpoint."""

import hashlib

import audit_single_model_structured_teacher as baseline
import single_model_paired_protection as paired
import single_model_shared_tensor_storage as shared_storage
from single_model_registered_adaptation import replace_once


def check_gradient_alias(payload):
    gradients = payload["gradients"]
    saved = payload["snapshot"]["task_gradients"]
    if (gradients.data_ptr() != saved.data_ptr() or gradients.shape != saved.shape
            or gradients.stride() != saved.stride() or gradients.dtype != saved.dtype
            or not torch.equal(gradients, saved)):
        raise ValueError("Retain every original gradient in one exact shared tensor")


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_structured_teacher as training': 'import train_single_matrixformer_point_inclusive as training',
    'import single_model_paired_protection as checks': 'import single_model_point_inclusive_proposal as checks',
    'checks.validate_layout(entries, batch["tasks"], point_before)': 'paired.validate_layout(entries, batch["tasks"], point_before)',
    'The registered constant2e-5 learning rate changed': 'The registered constant1e-5 learning rate changed',
    'The V20 initialization must reproduce the original V8 prediction grids': 'The V21 initialization must reproduce the original V8 prediction grids',
    'payload = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)':
        'payload = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)\n'
        '        hashes.update(shared_storage.audit(payload, raw_path))\n'
        '        check_gradient_alias(payload)',
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
