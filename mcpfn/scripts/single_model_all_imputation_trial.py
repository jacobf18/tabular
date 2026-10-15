"""Bind prospective storage and complete-scope retention to a fresh V40 trial."""

from math import prod

import single_model_all_imputation_retention as retention
from single_model_registered_adaptation import replace_once


POLICY = ("V40 is a fresh scope-only retention trial after the accepted and independently audited "
    "all-imputation probe. " + retention.POLICY
    + " This separate full-trial registration authorizes64 fresh attempts initialized only from V29, "
    "not either probe or V39. Retain the unchanged priors, teachers, losses, fixed-half rule,22 development "
    "guards,32 confirmation comparisons and frozen real benchmark campaign. Extend only training-time "
    "V29-reference coverage. Prospectively allow2432 MiB checkpoint/16 MiB root/40 GiB scratch, with256 MiB "
    "checkpoint metadata and explicit additional complete-grid storage. Synthetic results do not establish "
    "benchmark success; all individual regressions remain reportable.")


def reference_storage(batches):
    areas = []
    for batch in batches:
        for task in batch["tasks"]:
            specification = task["specification"]
            if specification["domain"] == "imputation":
                shape = specification.get("structured", specification)["shape"]
                if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
                    raise ValueError("Require complete fixed positive matrix dimensions before budgeting")
                areas.append(prod(shape))
    if len(batches) != 64 or len(areas) != 1664:
        raise ValueError("Account for every original imputation task in all64 fresh attempts")
    return {"tasks": len(areas), "cells": sum(areas), "archived_bytes": 9 * sum(areas),
        "replay_bytes": 4 * sum(areas), "atomic_reference_bytes": 9 * max(areas),
        "serialization_and_manifest_allowance": 64 * 1024**2}


def reference_bytes(batches):
    accounting = reference_storage(batches)
    return sum(value for key, value in accounting.items() if key not in {"tasks", "cells"})


def adapt_recipe(source):
    source = replace_once(source, "import single_model_full_network_storage as diagnostics",
        "import single_model_all_imputation_storage as diagnostics")
    source = replace_once(source, '"metadata_and_audit_allowance": 128 * 1024**2',
        '"metadata_and_audit_allowance": 256 * 1024**2')
    source = replace_once(source, 'TRIAL_POLICY = (',
        'STORAGE_ACCOUNTING["scratch"]["all_imputation_reference_allowance"] = all_trial.reference_bytes(previous_training.batches())\n'
        'TRIAL_POLICY = (')
    source = replace_once(source, 'TRIAL_POLICY = ANCHOR_POLICY', 'TRIAL_POLICY = ANCHOR_POLICY = all_trial.POLICY')
    return replace_once(source, 'REGISTER_SOURCE = fixed_trial.adapt_registration(REGISTER_SOURCE)',
        'REGISTER_SOURCE = fixed_trial.adapt_registration(REGISTER_SOURCE)\n'
        'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, \'    sources.add(REPO / "tests/test_single_model_fixed_half_trial.py")\',\n'
        '    \'    sources.update({REPO / "tests/test_single_model_fixed_half_trial.py", REPO / "tests/test_single_model_all_imputation_trial.py", REPO / "tests/test_single_model_all_imputation_storage.py"})\')')
