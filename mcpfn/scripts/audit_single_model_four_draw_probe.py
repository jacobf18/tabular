"""Independently reconstruct every four-draw probe teacher-mean contribution."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_standardized_teacher_huber_probe as blueprint
import audit_single_model_native_column_mean_probe as original_math
import preflight_single_model_four_draw as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")
CONFIG_SHA256 = None
POLICY = blueprint.POLICY.replace("all26 imputation tasks", "all52 imputation tasks").replace(
    "both original uniform mean20 rating terms", "all four original uniform mean20 rating terms") + (
    " Apply the unchanged independent arithmetic to all56 four-draw tasks, every physical "
    "column and both original endpoints; native and structured task weights are halved, not "
    "their raw objectives. This supplementary audit authorizes no model promotion or full trial."
)
huber = blueprint.huber
for name in ("describe", "imputation_task"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Independent mean arithmetic must not retain an old-probe closure")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
AUDIT_SOURCE = blueprint.AUDIT_SOURCE
for old, new in (("len(tasks) != 28", "len(tasks) != 56"),
        ('{"v8": 24, "v29": 26}', '{"v8": 48, "v29": 52}'),
        ('["reference_grids"] != 26', '["reference_grids"] != 52'),
        ('len(config["batch"]["tasks"]) != 28', 'len(config["batch"]["tasks"]) != 56'),
        ("len(rows) != 28", "len(rows) != 56"), ("list(range(28))", "list(range(56))"),
        ("{True: 26, False: 2}", "{True: 52, False: 4}"),
        ("Reweight all26 imputation tasks, preserving both uniform rating terms",
            "Reweight all52 imputation tasks, preserving all four uniform rating terms"),
        ('"tasks": 28', '"tasks": 56'), ('"imputation_reweighted_tasks": 26', '"imputation_reweighted_tasks": 52'),
        ('"native_reweighted_tasks": 2', '"native_reweighted_tasks": 4'),
        ('"structured_reweighted_tasks": 24', '"structured_reweighted_tasks": 48'),
        ('"unchanged_rating_tasks": 2', '"unchanged_rating_tasks": 4'),
        ("test_single_model_standardized_teacher_huber_probe_audit.py", "test_single_model_four_draw_probe_audit.py")):
    AUDIT_SOURCE = replace_once(AUDIT_SOURCE, old, new)
exec(compile(AUDIT_SOURCE, __file__, "exec"), globals())
_audit = main
audit = types.FunctionType(blueprint.audit.__code__, globals(), "audit", blueprint.audit.__defaults__)
main = types.FunctionType(blueprint.main.__code__, globals(), "main", blueprint.main.__defaults__)


if __name__ == "__main__":
    main()
