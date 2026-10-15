"""Independently audit V27 mixed-prior training, original references and teacher identities."""

from contextvars import ContextVar
import hashlib
import inspect
import sys
import time

import audit_single_model_top_two_blocks as template
from audit_single_model_top_two_blocks import compact_cache, compact_template, compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_general_imputation_teacher_audit_replay", default=None)
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_top_two_blocks as training':
        'import train_single_matrixformer_general_imputation_teacher as training',
    'saved["parent_training_updates"] != 0': 'saved["parent_training_updates"] != training.PARENT_UPDATES',
    'parent_report_path = training.INITIAL_RUN / "validation_0.json"':
        'parent_report_path = training.INITIAL_RUN / "validation_64.json"',
    'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",))], []':
        'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",)), (parent_report, ("student",))], []',
    '152 * (3 + len(boundaries))': '152 * (4 + len(boundaries))',
    'The V26 initialization must reproduce the original V8 prediction grids':
        'The V27 initialization must reproduce the audited V26 step64 prediction grids',
    '# Final-two-block plus decoder training audit': '# General-imputation teacher training audit',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("load_regenerated_task", "regeneration_evidence", "run_audit"):
    exec(compile(inspect.getsource(getattr(compact_template, name)), __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "general_imputation_teacher_v27_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
