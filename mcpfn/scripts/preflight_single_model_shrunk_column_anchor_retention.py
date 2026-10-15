"""Probe count-shrunk retention only after the completely audited negative V38 trial."""

import preflight_single_model_column_anchor_retention as predecessor
import single_model_shrunk_column_anchor_retention as shrunk_retention
from single_model_registered_adaptation import replace_assignment


POLICY_TEXT = ("One fresh28-task full-network optimizer feasibility probe only after all64 V38 attempts,59 "
    "accepted updates, complete original and added-reference audits and56-grid endpoint replay, with no "
    "eligible selection, consumed confirmation or real campaign. Initialize only from the unchanged V29 "
    "step25, never V38 or a diagnostic endpoint. " + shrunk_retention.POLICY
    + " This separate preregistration, not the standalone loss module, authorizes exactly one bounded "
    "synthetic feasibility probe after its predecessor is complete and stopped. Preserve238 trainable "
    "tensors/77,998,496 parameters,16 frozen buffers, all28 fresh tasks and84 raw gradients,30 true-target "
    "point protections,.5/.25/.25 prior masses, half-composite/half-point proposal, fresh5e-6 AdamW,.1 margin, "
    "2x norm bound and exact rollback. Keep the .9 allocator-only base adjustment. Independently replay "
    "both original teachers and every eligible full added reference grid, then independently score the "
    "count-shrunk penalty column by column on CPU. Preserve18/6/50 GiB reserves and384 MiB/16 MiB/32 GiB "
    "stage allowances. Scratch is machine-local, not provider-persistent backup. No new validation, "
    "selection, confirmation, real evaluation, inference ensemble, consumed-seed reuse, scientific retry, "
    "gate change or mutation of registered predecessors. The diagnostic model must never initialize a "
    "subsequent scientific trial. Feasibility is not evidence of improved qualification or benchmark results.")

SOURCE = predecessor.SOURCE
for old, new in {
    "import run_single_model_anchor_retention_full_pipeline as latest":
        "import run_single_model_column_anchor_retention_full_pipeline as latest",
    "import single_model_column_anchor_retention as retention":
        "import single_model_shrunk_column_anchor_retention as retention",
    "column_anchor_retention_preflight_v1": "shrunk_column_anchor_retention_preflight_v1",
    "901100901, 902100901": "931100901, 932100901",
    '"accepted_updates": 60': '"accepted_updates": 59',
    "tests/test_single_model_column_anchor_retention.py": "tests/test_single_model_shrunk_column_anchor_retention.py",
    "tests/test_single_model_column_anchor_retention_preflight.py":
        "tests/test_single_model_shrunk_column_anchor_retention_preflight.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit count-shrunk probe adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
SOURCE = SOURCE.replace("V37", "V38")
exec(compile(SOURCE, __file__, "exec"), globals())
