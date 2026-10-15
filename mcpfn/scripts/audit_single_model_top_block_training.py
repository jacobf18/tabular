"""Audit all V24 compact tasks, original guards and last-block optimizer coordinates."""

from contextvars import ContextVar
import hashlib
import inspect
import sys
import time

import audit_single_model_structured_teacher as baseline
import audit_single_model_compact_continuation as compact_template
import single_model_compact_training_cache as compact_cache
import single_model_shared_tensor_storage as compressed_endpoint
from single_model_registered_adaptation import replace_once


REPLAY = ContextVar("single_model_top_block_audit_replay", default=None)

ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'import train_single_matrixformer_structured_teacher as training': 'import train_single_matrixformer_top_block as training',
    'row[0].startswith("decoder.")': 'row[0].startswith(training.PREFIXES)',
    'len(selected) != 6 or sum(row[2] for row in selected) != 11227016':
        'len(selected) != 34 or sum(row[2] for row in selected) != 19496843',
    'frozen = decoder.frozen_state_equal(parent, saved["model"], {row[0] for row in selected})':
        'frozen = training.preflight.frozen_inventory(parent, saved["model"], {row[0] for row in selected})',
    'or saved["independent_confirmation_passed"] or saved["preflight_only"] or not saved["decoder_only_training"]':
        'or saved["independent_confirmation_passed"] or saved["preflight_only"] or saved["decoder_only_training"]\n'
        '            or saved.get("top_block_training") is not True',
    '    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():\n'
    '        raise ValueError("Training audit must be CPU only")':
        '    if REPLAY.get() is None:\n        raise ValueError("Require independent complete compact-task regeneration")',
    '            payload = torch.load(path, map_location="cpu", weights_only=False)':
        '            payload = load_regenerated_task(path)',
    'frozen["frozen_tensors"] != 248': 'frozen["frozen_tensors"] != 220',
    'not frozen["backbone_and_buffers_bitwise_unchanged"]':
        'not frozen["unselected_parameters_and_all_buffers_bitwise_unchanged"]',
    '        endpoint_result = checks.check_endpoint(': '        hashes.update(training.audit_endpoint(payload, raw_path))\n        endpoint_result = checks.check_endpoint(',
    '    verify_hashes(hashes)\n    training.verify_registration(run)':
        '    hashes.update(regeneration_evidence(through_step))\n    verify_hashes(hashes)\n    training.verify_registration(run)',
    '"decoder_only_training": True': '"decoder_only_training": False, "top_block_training": True',
    '"cuda_initialized": False, "autograd_replayed": False':
        '"cuda_initialized": torch.cuda.is_initialized(), "autograd_replayed": False,\n'
        '               "independently_regenerated_training_tasks": len(REPLAY.get()["seen"]),\n'
        '               "independently_regenerated_rating_contexts": REPLAY.get()["rating_contexts"],\n'
        '               "exact_complete_task_and_teacher_digest_checks": True',
    '# Decoder-only finite-guarded training audit': '# Last-block plus decoder training audit',
    'The V20 initialization must reproduce the original V8 prediction grids':
        'The V24 initialization must reproduce the original V8 prediction grids',
    'The registered constant2e-5 learning rate changed': 'The registered constant1e-5 learning rate changed',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
if ADAPTED_SOURCE.count('"changed_decoder_tensors"') != 3:
    raise ValueError("Adapt every boundary/selected-state change inventory")
ADAPTED_SOURCE = ADAPTED_SOURCE.replace('"changed_decoder_tensors"', '"changed_trainable_tensors"')
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())

for name in ("load_regenerated_task", "regeneration_evidence", "run_audit"):
    exec(compile(inspect.getsource(getattr(compact_template, name)), __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "top_block_v24_audit")
    arguments = parser.parse_args()
    run_audit(arguments.output.resolve())
