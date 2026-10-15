"""Preserve three completed nonselectable diagnostic checkpoints before further training."""

import inspect
from pathlib import Path

import torch

import relocate_single_model_completed_preflight_checkpoints as template
from relocate_single_model_completed_preflight_checkpoints import copying, readers, sha256, verify_hashes, write_json
from single_model_registered_adaptation import replace_once


ROOT = copying.ROOT
OUTPUT = ROOT / "anchor_preflight_checkpoint_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/anchor_preflight_checkpoints_v1"
OWNERS = {
    "anchor_rehearsal_preflight_v1": (0, "0b5591ffb858a30365860ca83ad49fb782eac7235b2b2c21c2df7e5e8e836669"),
    "anchor_small_step_preflight_v1": (1, "cb3635d9baa90a21f7f94e2afec96228964ab2c477e9c334359debac5c51c9c0"),
    "legacy_strata_preflight_v1": (1, "c8f9ad4b5fcafaac3842367f5bae277861150a8c8799b1fa25c1fab42c52c345"),
}
POLICY = ("Storage-only relocation of exactly three completed and independently audited, nonselectable "
    "preflight checkpoints. Preserve the first anchor probe's rejected update and the small-step/legacy-stratum "
    "probes' accepted updates without promotion. Never move selected models, original teachers, archives, "
    "active outputs or V28's failed partial checkpoint. Require all owners and readers stopped; bind original "
    "audit/result/configuration hashes, copy/fsync/hash/mmap each file before atomic canonical symlink "
    "publication, then independently recheck all three. Keep18/6/50 GiB checkpoint/root/scratch reserves. "
    "The exact diagnostic bytes remain readable at their canonical paths, but their backing storage becomes "
    "UUID-bound machine-local scratch, not provider-persistent storage or an external backup. No training, "
    "evaluation, qualification, source change, altered evidence or retry.")

checkpoint_identity = template.checkpoint_identity
for name in ("require_idle", "inventory"):
    exec(compile(inspect.getsource(getattr(template, name)), __file__, "exec"), globals())
exec(compile(inspect.getsource(template.template.status), __file__, "exec"), globals())
MAIN_SOURCE = replace_once(template.MAIN_SOURCE,
    'Path(__file__).resolve().parents[1] / "tests/test_single_model_preflight_checkpoint_relocation.py",',
    'Path(template.template.__file__).resolve(), '
    'Path(__file__).resolve().parents[1] / "tests/test_single_model_anchor_checkpoint_relocation.py",')
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())
os, shutil, REPO = template.os, template.shutil, template.REPO


if __name__ == "__main__":
    main()
