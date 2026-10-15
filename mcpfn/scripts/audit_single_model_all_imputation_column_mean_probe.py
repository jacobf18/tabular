"""Independently reconstruct all26 column-weighted imputation mean terms."""

from collections import Counter
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_native_parent_mean_probe as blueprint
import audit_single_model_native_column_mean_probe as original_math
import preflight_single_model_all_imputation_column_mean as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
CONFIG_SHA256 = "669d4b633a362e5e577771ed48d7838253a3ced0ea7e496d76087bc7c93f6d11"
POLICY = (
    "CPU-only supplementary mean-term audit for the one registered all-imputation column-mean "
    "feasibility probe. Require its stopped complete original proposal, teacher-component and "
    "retention-component audits. Independently reconstruct count-shrunk equal-column mean20 "
    "Huber on all26 imputation tasks: both native and all24 structured tasks across every family. "
    "Keep the original uniform mean20 term on both ratings. Preserve all original columns, "
    "including empty-query, singleton and constant columns, both endpoints, task weights and "
    "every adverse/improved contribution. Check the exact registered compiled mean callback. "
    "No new inference, scientific seed, training, selection, gate, full-composite replay or "
    "historical autograd claim; objective arithmetic is not benchmark qualification."
)
huber = blueprint.huber
describe = types.FunctionType(blueprint.describe.__code__, globals(), "describe", blueprint.describe.__defaults__)
describe.__kwdefaults__ = blueprint.describe.__kwdefaults__


def imputation_task(registered, metadata):
    probe.retention.reference_role(metadata)
    cohort = "recsys" if metadata["domain"] == "recsys" else metadata["family"] if metadata.get("structured", False) else "legacy"
    if registered["cohort"] != cohort:
        raise ValueError("The registered cohort must agree with the complete synthetic task identity")
    return metadata["domain"] == "imputation"


AUDIT_SOURCE = replace_once(blueprint.AUDIT_SOURCE,
    '        native = native_task(registered, metadata)',
    '        column_weighted = imputation_task(registered, metadata)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE, 'predictions, native=native)', 'predictions, native=column_weighted)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE, '"native_reweighting": native}', '"native_reweighting": column_weighted}')
for old, new in {
    'native_reweighting': 'imputation_reweighting',
    'native_teacher_mean_tasks': 'imputation_teacher_mean_tasks',
    '"native_reweighted_tasks": 2': '"imputation_reweighted_tasks": 26, "native_reweighted_tasks": 2',
    '"unchanged_structured_tasks": 24': '"structured_reweighted_tasks": 24',
    '{True: 2, False: 26}': '{True: 26, False: 2}',
    'Reweight exactly both native tasks, retaining all24 structured and both ratings': 'Reweight all26 imputation tasks, preserving both uniform rating terms',
    'test_single_model_native_parent_mean_probe_audit.py': 'test_single_model_all_imputation_column_mean_probe_audit.py',
}.items():
    if old not in AUDIT_SOURCE:
        raise ValueError(f"Missing explicit all-imputation mean-audit adaptation: {old}")
    AUDIT_SOURCE = AUDIT_SOURCE.replace(old, new)
exec(compile(AUDIT_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
