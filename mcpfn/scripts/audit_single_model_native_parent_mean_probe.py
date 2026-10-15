"""Independently reconstruct all mean20 terms in the native-parent probe."""

from collections import Counter
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_dual_parent_mean_probe as blueprint
import audit_single_model_native_column_mean_probe as original_math
import preflight_single_model_native_parent_retention as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
CONFIG_SHA256 = "fc04a4288d5dbbbf925ccb3dc6c1d81bdc7b23c6d528a949fb9370040d681cb3"
POLICY = original_math.POLICY + (
    " Apply the unchanged independent saved-grid calculation to the registered native-parent "
    "probe. Native retention uses frozen actual V29, while all structured targets remain the "
    "exact FP32 equal-weight V8/V29 mean. These reference choices do not alter mean20, the "
    "native count-shrunk coefficient formula or the original feasibility gate. Require the "
    "stopped original audit to verify all50 retention components (24 V8 and26 V29) and26 "
    "combined targets before this supplementary mean-only audit. Preserve every column, "
    "task, endpoint and adverse contribution, with no new scientific data or selection."
)
huber = blueprint.huber
describe = types.FunctionType(blueprint.describe.__code__, globals(), "describe", blueprint.describe.__defaults__)
describe.__kwdefaults__ = blueprint.describe.__kwdefaults__


def native_task(registered, metadata):
    probe.retention.reference_role(metadata)
    cohort = "recsys" if metadata["domain"] == "recsys" else metadata["family"] if metadata.get("structured", False) else "legacy"
    if registered["cohort"] != cohort:
        raise ValueError("The original registered cohort must agree with the complete synthetic task metadata")
    return cohort == "legacy"


AUDIT_SOURCE = replace_once(blueprint.AUDIT_SOURCE,
    'or probe.read(probe.OUTPUT / "retention_audit.json")["component_grids"] != 50',
    'or probe.read(probe.OUTPUT / "retention_audit.json")["component_grids"] != 50\n'
    '            or probe.read(probe.OUTPUT / "retention_audit.json")["component_role_counts"] != {"v8": 24, "v29": 26}')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    'Path(blueprint.__file__).resolve(), Path(blueprint.blueprint.__file__).resolve(),',
    'Path(blueprint.__file__).resolve(), Path(blueprint.blueprint.__file__).resolve(), Path(original_math.__file__).resolve(),')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    'REPO / "tests/test_single_model_dual_parent_mean_probe_audit.py",',
    'REPO / "tests/test_single_model_native_parent_mean_probe_audit.py",\n'
    '        REPO / "tests/test_single_model_dual_parent_mean_probe_audit.py",')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '        reference_role = probe.retention.reference_role(metadata)\n        native = reference_role == "general"',
    '        native = native_task(registered, metadata)')
exec(compile(AUDIT_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
