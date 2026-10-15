"""Independently reconstruct unchanged mean20 terms for the dual-parent probe."""

from collections import Counter
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_parent_structured_mean_probe as blueprint
import preflight_single_model_dual_parent_retention as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
CONFIG_SHA256 = "040ee68892038461bc2c3f31492dec9ebdced9e03e0b285dcc98500dd282f9ac"
POLICY = blueprint.blueprint.POLICY + (
    " Apply the same independent saved-grid calculation to the separately registered fixed "
    "V8/V29 mean-structured retention probe. Both native references remain general. All24 "
    "structured targets use the registered fixed FP32 mean of both frozen components; neither "
    "target construction changes the mean20 formula, coefficient calculation or original "
    "feasibility gate. The original independent probe audit must verify all50 reference "
    "components and26 combined grids before this supplementary mean-only audit runs."
)
huber = blueprint.huber
describe = types.FunctionType(blueprint.describe.__code__, globals(), "describe", blueprint.describe.__defaults__)
describe.__kwdefaults__ = blueprint.describe.__kwdefaults__
AUDIT_SOURCE = replace_once(blueprint.AUDIT_SOURCE,
    'or not probe.read(probe.OUTPUT / "retention_audit.json")["complete"]',
    'or not probe.read(probe.OUTPUT / "retention_audit.json")["complete"]\n'
    '            or probe.read(probe.OUTPUT / "retention_audit.json")["component_grids"] != 50\n'
    '            or probe.read(probe.OUTPUT / "retention_audit.json")["reference_grids"] != 26')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    'Path(blueprint.__file__).resolve(), REPO / "tests/test_single_model_parent_structured_mean_probe_audit.py",',
    'Path(blueprint.__file__).resolve(), Path(blueprint.blueprint.__file__).resolve(),\n'
    '        REPO / "tests/test_single_model_dual_parent_mean_probe_audit.py",\n'
    '        REPO / "tests/test_single_model_parent_structured_mean_probe_audit.py",')
exec(compile(AUDIT_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
