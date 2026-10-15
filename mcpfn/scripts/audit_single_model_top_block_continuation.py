"""Audit every new V25 task, original retention guard and last-block optimizer coordinate."""

from contextvars import ContextVar
import hashlib
import inspect
import sys
import time

import audit_single_model_top_block_training as template
import single_model_compact_training_cache as compact_cache
import single_model_shared_tensor_storage as compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_top_block_continuation_audit", default=None)
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_top_block as training':
        'import train_single_matrixformer_top_block_continuation as training',
    'saved["parent_training_updates"] != 0': 'saved["parent_training_updates"] != training.PARENT_UPDATES',
    'parent_report_path = training.INITIAL_RUN / "validation_0.json"':
        'parent_report_path = training.INITIAL_RUN / "validation_64.json"',
    'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",))], []':
        'fields, reports = [(reference, ("general", "teacher")), (initial, ("student",)), (parent_report, ("student",))], []',
    '152 * (3 + len(boundaries))': '152 * (4 + len(boundaries))',
    'Never replace the full64-attempt qualification audit with a partial trial':
        'Never replace the full96-attempt qualification audit with a partial trial',
    'The V24 initialization must reproduce the original V8 prediction grids':
        'The V25 initialization must reproduce the audited V24 step64 prediction grids',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("load_regenerated_task", "regeneration_evidence", "run_audit"):
    exec(compile(inspect.getsource(getattr(template.compact_template, name)), __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "top_block_continuation_v25_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
