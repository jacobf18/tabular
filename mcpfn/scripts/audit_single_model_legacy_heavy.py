"""Independently audit every V30 admission and the complete legacy-heavy continuation."""

from contextvars import ContextVar
import hashlib
import sys
import time
import types

import audit_single_model_legacy_strata as template
from audit_single_model_legacy_strata import admission, attempts, compact_cache, compact_template, compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_legacy_heavy_audit_replay", default=None)
for name in ("admission_inventory", "replay_admission", "check_empty_optimizer_endpoint", "audit_final_endpoint",
             "load_regenerated_task", "regeneration_evidence", "run_audit"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain helper closures into the previous independent audit")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_legacy_strata as training': 'import train_single_matrixformer_legacy_heavy as training',
    'parent_report_path = training.INITIAL_RUN / "validation_10.json"':
        'parent_report_path = training.INITIAL_RUN / "validation_25.json"',
    'The V29 initialization must reproduce the audited V27 step10 prediction grids':
        'The V30 initialization must reproduce the audited V29 step25 prediction grids',
    '# Admission-aware legacy-stratum training audit': '# Legacy-heavy mixed-prior training audit',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "legacy_heavy_v30_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
