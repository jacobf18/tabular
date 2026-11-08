"""Read-only terminal review preserving the original fresh quarter-step outcome."""

import argparse
from pathlib import Path
import types

import recheck_single_model_balanced_native_coverage_probe as blueprint
from single_model_registered_adaptation import replace_once


SOURCE = blueprint.SOURCE
for old, new in {
    'balanced_native_coverage_preflight_v1': 'balanced_native_quarter_preflight_v1',
    'preflight_single_model_balanced_native_coverage': 'preflight_single_model_balanced_native_quarter',
    'audit_single_model_balanced_native_coverage_probe.py': 'audit_single_model_balanced_native_quarter_probe.py',
    'test_single_model_balanced_native_coverage_probe_recheck.py': 'test_single_model_balanced_native_quarter_probe_recheck.py',
    'Balanced native-coverage probe: terminal review': 'Balanced native quarter-step probe: terminal review',
    'full-size native contexts; standardized': 'full-size native contexts, fixed quarter optimizer displacement; standardized',
    "['fixed_half_step']": "['fixed_quarter_step']",
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit quarter-review adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, '    record = result[\'record\']',
    '    require(config["displacement_fraction"] == probe.paired.FRACTION == .25\n'
    '        and audit["raw_endpoint"]["fixed_quarter_step"]["fraction"] == .25,\n'
    '        "Keep the registered quarter displacement and explicit audit schema")\n'
    '    record = result[\'record\']')
exec(compile(SOURCE, __file__, 'exec'), globals())
_review = main
review = types.FunctionType(blueprint.review.__code__, globals(), 'review', blueprint.review.__defaults__)
main = types.FunctionType(blueprint.main.__code__, globals(), 'main', blueprint.main.__defaults__)


if __name__ == '__main__':
    main()
