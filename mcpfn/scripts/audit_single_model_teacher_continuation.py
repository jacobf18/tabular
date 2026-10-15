"""Audit the V20-start continuation against original V8 retention and exact raw evidence."""

import hashlib

import audit_single_model_structured_teacher as baseline
import audit_single_model_point_inclusive_training as shared_audit
import single_model_shared_tensor_storage as shared_storage
from single_model_registered_adaptation import replace_once


check_gradient_alias = shared_audit.check_gradient_alias
ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_structured_teacher as training':
        'import train_single_matrixformer_teacher_continuation as training',
    'All152 original V8 initialization grids are mandatory': 'All152 V20 initialization grids are mandatory',
    'The V20 initialization must reproduce the original V8 prediction grids':
        'The V22 initialization must reproduce the V20 boundary64 prediction grids',
    'saved["parent_training_updates"] != 0': 'saved["parent_training_updates"] != 56',
    'parent_report_path = training.INITIAL_RUN / "validation_0.json"':
        'parent_report_path = training.INITIAL_RUN / "validation_64.json"',
    'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",))], []':
        'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",)), (parent_report, ("student",))], []',
    '152 * (3 + len(boundaries))': '152 * (4 + len(boundaries))',
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
