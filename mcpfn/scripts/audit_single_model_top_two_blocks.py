"""Independently audit every V26 compact task, original guard and two-block coordinate."""

from contextvars import ContextVar
import hashlib
import inspect
import sys
import time

import audit_single_model_top_block_training as template
from audit_single_model_top_block_training import compact_cache, compact_template, compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_top_two_blocks_audit_replay", default=None)
ADAPTED_SOURCE = template.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_top_block as training': 'import train_single_matrixformer_top_two_blocks as training',
    'len(selected) != 34 or sum(row[2] for row in selected) != 19496843':
        'len(selected) != 62 or sum(row[2] for row in selected) != 27766670',
    'frozen["frozen_tensors"] != 220': 'frozen["frozen_tensors"] != 192',
    'or saved.get("top_block_training") is not True':
        'or saved.get("top_block_training") is not True or saved.get("top_two_block_training") is not True',
    '"top_block_training": True': '"top_block_training": True, "top_two_block_training": True',
    'The V24 initialization must reproduce': 'The V26 initialization must reproduce',
    '# Last-block plus decoder training audit': '# Final-two-block plus decoder training audit',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("load_regenerated_task", "regeneration_evidence", "run_audit"):
    exec(compile(inspect.getsource(getattr(compact_template, name)), __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "top_two_blocks_v26_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
