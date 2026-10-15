"""Independently audit V31 admission, task payloads and the full point-inclusive endpoint."""

from contextvars import ContextVar
import hashlib
import sys
import time
import types

import audit_single_model_legacy_heavy as template
from audit_single_model_legacy_heavy import admission, attempts, compact_cache, compact_template, compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_legacy_point_audit_replay", default=None)
for name in ("admission_inventory", "replay_admission", "check_empty_optimizer_endpoint", "audit_final_endpoint",
             "load_regenerated_task", "regeneration_evidence", "run_audit"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding independent audit closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

ADAPTED_SOURCE = replace_once(template.ADAPTED_SOURCE,
    'import train_single_matrixformer_legacy_heavy as training', 'import train_single_matrixformer_legacy_point as training')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE,
    'import single_model_legacy_stratum_protection as checks', 'import single_model_legacy_point_proposal as checks')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, "The V30 initialization", "The V31 initialization")
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "legacy_point_v31_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
