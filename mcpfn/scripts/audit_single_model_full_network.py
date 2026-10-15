"""Independently audit every V34 admission, task and full-network optimizer coordinate."""

from contextvars import ContextVar
import hashlib
import sys
import time
import types

import audit_single_model_top_three_blocks as template
from audit_single_model_top_three_blocks import admission, attempts, compact_cache, compact_template, compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_full_network_audit_replay", default=None)
for name in ("admission_inventory", "replay_admission", "check_empty_optimizer_endpoint", "audit_final_endpoint",
             "load_regenerated_task", "regeneration_evidence", "run_audit"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding independent audit closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_top_three_blocks as training': 'import train_single_matrixformer_full_network as training',
    'len(selected) != 90 or sum(row[2] for row in selected) != 36036497':
        'len(selected) != 238 or sum(row[2] for row in selected) != 77998496',
    'frozen["frozen_tensors"] != 164': 'frozen["frozen_tensors"] != 16',
    'saved.get("top_three_block_training") is not True': 'saved.get("full_network_training") is not True',
    '"top_three_block_training": True': '"full_network_training": True',
    "The V33 initialization": "The V34 initialization",
    "# Final-three-block plus decoder training audit": "# Full-network training audit",
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "full_network_v34_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
