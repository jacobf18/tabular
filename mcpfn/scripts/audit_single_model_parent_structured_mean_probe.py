"""Reconstruct unchanged mean20 terms from the complete parent-structured probe."""

from collections import Counter
import inspect
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_native_column_mean_probe as blueprint
import preflight_single_model_parent_structured_retention as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
CONFIG_SHA256 = "0d1fd3a0ff91f275de89709b8be8538b8611777bc296675a8e56fe0c84ec0821"
POLICY = blueprint.POLICY + (
    " Apply the identical independent saved-grid calculation to the separately registered "
    "parent-structured retention probe. Its two native references remain general and all24 "
    "structured references are frozen actual V29; neither reference substitution changes the "
    "mean20 formula, coefficient computation, scope of this audit or original feasibility gate."
)
huber = blueprint.huber
describe = types.FunctionType(blueprint.describe.__code__, globals(), "describe", blueprint.describe.__defaults__)
describe.__kwdefaults__ = blueprint.describe.__kwdefaults__
AUDIT_SOURCE = inspect.getsource(blueprint.main)
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    'config = probe.read(probe.OUTPUT / "config.json")', 'config = probe.verify(probe.OUTPUT)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    'or not probe.read(probe.OUTPUT / "retention_audit.json")["complete"]',
    'or not probe.read(probe.OUTPUT / "retention_audit.json")["complete"]\n'
    '            or probe.diagnosis.original.process_identity(probe.read(probe.OUTPUT / "status.json")["pid"]) is not None')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    'Path(__file__).resolve(), REPO / "tests/test_single_model_native_mean_probe_audit.py"',
    'Path(__file__).resolve(), Path(blueprint.__file__).resolve(), '
    'REPO / "tests/test_single_model_parent_structured_mean_probe_audit.py", '
    'REPO / "tests/test_single_model_native_mean_probe_audit.py"')
exec(compile(AUDIT_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
