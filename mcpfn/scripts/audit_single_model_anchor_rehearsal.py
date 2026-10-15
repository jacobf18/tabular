"""Independently audit V28 anchor-stratified training and all original references."""

from contextvars import ContextVar
import hashlib
import inspect
import sys
import time

import audit_single_model_general_imputation_teacher as template
from audit_single_model_general_imputation_teacher import compact_cache, compact_template, compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_anchor_rehearsal_audit_replay", default=None)
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_general_imputation_teacher as training':
        'import train_single_matrixformer_anchor_rehearsal as training',
    'parent_report_path = training.INITIAL_RUN / "validation_64.json"':
        'parent_report_path = training.INITIAL_RUN / "validation_10.json"',
    'The V27 initialization must reproduce the audited V26 step64 prediction grids':
        'The V28 initialization must reproduce the audited V27 step10 prediction grids',
    '# General-imputation teacher training audit': '# Anchor-rehearsal training audit',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("load_regenerated_task", "regeneration_evidence", "run_audit"):
    exec(compile(inspect.getsource(getattr(compact_template, name)), __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "anchor_rehearsal_v28_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
