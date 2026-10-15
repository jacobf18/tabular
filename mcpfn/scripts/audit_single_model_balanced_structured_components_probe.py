"""Reconstruct all saved teacher-mean terms for the balanced-prior probe."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import shutil
import types

import numpy as np
import torch

import audit_single_model_all_imputation_column_mean_probe as blueprint
import audit_single_model_native_column_mean_probe as original_math
import preflight_single_model_balanced_structured_components as probe
from analyze_single_model_development_columns import save_csv
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + '_mean_audit')
CONFIG_SHA256 = None
POLICY = blueprint.POLICY + (
    ' Apply the same independent calculation to the separately registered balanced-structured-'
    'components probe. Bind its original configuration SHA through a required explicit CLI '
    'argument, checked against the original registration before any output. Preserve complete '
    'positive and adverse contributions whether the optimizer proposal is accepted or rejected. '
    'This supplementary mean-only audit does not authorize a full trial or alter any original gate.'
)
huber = blueprint.huber
for name in ('describe', 'imputation_task'):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Mean-term audit callbacks must remain closure-free')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
AUDIT_SOURCE = replace_once(blueprint.AUDIT_SOURCE,
    'test_single_model_all_imputation_column_mean_probe_audit.py',
    'test_single_model_balanced_structured_components_probe_audit.py')
exec(compile(AUDIT_SOURCE, __file__, 'exec'), globals())
_audit = main


def audit(config_sha256):
    if (not isinstance(config_sha256, str) or len(config_sha256) != 64
            or any(character not in '0123456789abcdef' for character in config_sha256)):
        raise ValueError('Require the exact original configuration SHA256')
    namespace = {**globals(), 'CONFIG_SHA256': config_sha256}
    implementation = types.FunctionType(_audit.__code__, namespace, 'audit', _audit.__defaults__)
    return implementation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-sha256', required=True)
    arguments = parser.parse_args()
    audit(arguments.config_sha256)


if __name__ == '__main__':
    main()
