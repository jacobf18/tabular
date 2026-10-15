"""Probe equal-column normalized native-anchor retention after the complete negative V37 trial."""

import inspect

import preflight_single_model_anchor_retention as predecessor
import single_model_column_anchor_retention as column_retention
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = ("One fresh28-task full-network optimizer feasibility probe after all64 V37 attempts,60 accepted "
    "updates, complete original and added-reference audits and56-grid endpoint replay, without any eligible "
    "selection, evaluated confirmation or real campaign. Initialize only from the same V29 step25, not V37 "
    "or a diagnostic endpoint. " + column_retention.POLICY
    + " Preserve238 trainable tensors/77,998,496 parameters,16 frozen buffers, all28 fresh tasks and84 raw "
    "gradients,30 true-target point protections,.5/.25/.25 prior masses, half-composite/half-point proposal, "
    "fresh5e-6 AdamW,.1 margin,2x norm bound and exact rollback. Keep the .9 allocator-only base change. "
    "Independently replay both original teachers and every full added reference grid, then independently "
    "score the new penalty column by column on CPU. Preserve18/6/50 GiB reserves and384 MiB/16 MiB/32 GiB "
    "stage allowances. Scratch is machine-local, not provider-persistent backup. Initial zero residual and "
    "float32 streaming cancellation are implementation questions, not evidence of regularization efficacy. "
    "No new validation, selection, confirmation, real evaluation, inference ensemble, consumed-seed reuse, "
    "scientific retry, gate change or mutation of registered predecessors. The diagnostic model must never "
    "initialize a subsequent scientific trial.")

SOURCE = inspect.getsource(predecessor)
for old, new in {
    "import run_single_model_full_network_cap90_v2_pipeline as latest":
        "import run_single_model_anchor_retention_full_pipeline as latest",
    "import single_model_anchor_retention as retention":
        "import single_model_column_anchor_retention as retention",
    "anchor_retention_preflight_v1": "column_anchor_retention_preflight_v1",
    "871100901, 872100901": "901100901, 902100901",
    '"accepted_updates": 61': '"accepted_updates": 60',
    'tests/test_single_model_anchor_retention.py': 'tests/test_single_model_column_anchor_retention.py',
    'tests/test_single_model_anchor_retention_preflight.py': 'tests/test_single_model_column_anchor_retention_preflight.py',
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit column-retention probe adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
SOURCE = replace_once(SOURCE,
    'row[role] = retention.SPECIFICATION["weight"] * float(difference.square().mean())',
    'row[role] = retention.independent_penalty(episode.target, episode.mask, prediction, payload["prediction"])')
SOURCE = SOURCE.replace("V36", "V37")
exec(compile(SOURCE, __file__, "exec"), globals())
