"""Probe all-imputation retention once after V39's complete negative qualification."""

from pathlib import Path

import preflight_single_model_fixed_half_shrunk_retention as previous_probe
import run_single_model_fixed_half_shrunk_retention_pipeline as completed
import single_model_all_imputation_retention as all_retention
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = ("One fresh28-task full-network feasibility probe after all64 V39 attempts,59 accepted "
    "updates, complete original/fixed-half/reference audits and56-grid endpoint replay, with no eligible "
    "selection, evaluated confirmation or real campaign. Initialize only from unchanged V29 step25, "
    "never V39 or a diagnostic endpoint. " + all_retention.POLICY
    + " This separate preregistration authorizes exactly one fresh synthetic probe, not a full training "
    "trial. Preserve238 trainable tensors/77,998,496 parameters,16 frozen buffers,28 tasks,84 raw gradients, "
    "30 original point protections,.5/.25/.25 prior masses, original teachers/losses/normalization, fresh "
    "5e-6 AdamW,.1 uncontracted projection margin, fixed-half displacement,2x norm bound and exact rollback. "
    "Keep the.9 allocator cap and18/6/50 GiB reserves. Allow512 MiB/16 MiB/32 GiB writes, with128 MiB "
    "additional checkpoint metadata/reference allowance for the broader full-grid coverage. "
    "Independently replay both original teachers and every added full reference grid, including all "
    "spectral tasks; audit every eligibility decision and original query. This is a scope-only loss "
    "experiment, not a teacher replacement, loss rescaling, variance-estimator change or relaxed guard. "
    "Scratch is machine-local, not provider-persistent backup. No held-out evaluation, reselection, "
    "real data, consumed-seed reuse, inference ensemble, scientific retry or probe-checkpoint warm start. "
    "Feasibility does not establish qualification or benchmark improvement.")


def require_exact_predecessor():
    paths = {
        completed.training.OUTPUT / "config.json": "8c8be9d67504d26b346ae22179f983b4a1f4b07af69212aa8067d40dd9674f4b",
        completed.OUTPUT / "result.json": "2db49b6c9031d1b3987414433b3673d63ea694d003c1488a50c3b752c9fd4050",
        completed.AUDIT / "audit.json": "998a5fbc4077f419f27f8b4e3a559a0ceb862617aad52ab6515271ce7678a949",
        completed.REPLAY / "audit.json": "73df940b10b156304825820c34edd1496d65e75b93247aeb34d6679931cdad65",
        Path("/mnt/matrixformer_scratch/fixed_half_shrunk_retention_v39/anchor_retention_replay/summary.json"):
            "b32925df2aaacdb84403a5d3197e584cc15d01ff0904171dc4b7500976a448c0",
    }
    expected = {str(path): digest for path, digest in paths.items()}
    previous_probe.verify_hashes(expected)
    return expected


SOURCE = previous_probe.SOURCE
for old, new in {
    "import run_single_model_column_anchor_retention_full_pipeline as latest":
        "import run_single_model_fixed_half_shrunk_retention_pipeline as latest",
    "import single_model_shrunk_column_anchor_retention as retention":
        "import single_model_all_imputation_retention as retention",
    "fixed_half_shrunk_retention_preflight_v1": "all_imputation_retention_preflight_v1",
    "961100901, 962100901": "991100901, 992100901",
    "require_v38_predecessor": "require_v39_predecessor",
    "tests/test_single_model_shrunk_column_anchor_retention.py": "tests/test_single_model_all_imputation_retention.py",
    "tests/test_single_model_shrunk_column_anchor_retention_preflight.py":
        "tests/test_single_model_all_imputation_retention_preflight.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit all-imputation feasibility adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE,
    '    return {**require_v39_predecessor(), **require_displacement_panel()}',
    '    return {**require_v39_predecessor(), **require_exact_predecessor()}')
SOURCE = replace_once(SOURCE, '"prediction_grids": 1368, "full_network_training": True,',
    '"prediction_grids": 1368, "full_network_training": True, "fixed_half_training": True, "displacement_fraction": .5,')
SOURCE = replace_once(SOURCE, 'RESERVES, WRITE_BUDGETS = dict(template.RESERVES), dict(template.WRITE_BUDGETS)',
    'RESERVES, WRITE_BUDGETS = dict(template.RESERVES), dict(template.WRITE_BUDGETS)\n'
    'WRITE_BUDGETS["checkpoint"] += 128 * 1024**2')
SOURCE = replace_once(SOURCE, 'STORAGE_ACCOUNTING = dict(template.STORAGE_ACCOUNTING)',
    'STORAGE_ACCOUNTING = dict(template.STORAGE_ACCOUNTING)\n'
    'STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] += 128 * 1024**2')
SOURCE = SOURCE.replace("V38", "V39")
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
exec(compile(SOURCE, __file__, "exec"), globals())
