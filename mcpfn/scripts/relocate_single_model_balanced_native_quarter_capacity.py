"""Prospective byte-preserving capacity preparation after complete quarter-probe closure."""

from pathlib import Path

import relocate_single_model_balanced_native_capacity as native_blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


SOURCE = Path(native_blueprint.__file__).read_text()
for old, new in {
    'import preflight_single_model_balanced_native_coverage as probe':
        'import preflight_single_model_balanced_native_quarter as probe',
    'balanced_native_capacity_relocation_v1': 'balanced_native_quarter_capacity_relocation_v1',
    'balanced_native_capacity_v1': 'balanced_native_quarter_capacity_v1',
    'test_single_model_balanced_native_capacity.py': 'test_single_model_balanced_native_quarter_capacity.py',
    'probe.closed.original.process_identity': 'probe.probe_blueprint.closed.original.process_identity',
    "'balanced_native_coverage_preflight_v1': 'complete'":
        "'balanced_native_coverage_preflight_v1': 'complete', 'balanced_native_quarter_preflight_v1': 'complete'",
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit quarter-capacity adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, 'PROBE_CONFIG_SHA256',
    '9b8418c5e0ee88fb64b9a163efa31594927fccc386419986cb0fd0e6f0c859e4')
SOURCE = replace_assignment(SOURCE, 'POLICY', native_blueprint.POLICY.replace(
    'full-context native-coverage probe closes accepted', 'fresh full-context quarter-step probe closes accepted')
    + ' Preserve the original native half-step rejection unchanged. Bind the explicit quarter '
    'optimizer audit with fraction0.25 and all238 changed trainable tensors,16 unchanged buffers. '
    'A pending run or accepted result without completed audits does not authorize storage changes.')
SOURCE = replace_once(SOURCE, "or audit['teacher_replay']['component_grids'] != 52 or audit['cuda_initialized'] is not False",
    "or audit['teacher_replay']['component_grids'] != 52 or audit['cuda_initialized'] is not False\n"
    "            or audit['raw_endpoint']['fixed_quarter_step']['fraction'] != .25\n"
    "            or audit['raw_endpoint']['fixed_quarter_step']['parameters'] != 77998496\n"
    "            or audit['raw_endpoint']['fixed_quarter_step']['actual_saved_displacements_checked'] is not True\n"
    "            or audit['raw_endpoint']['accepted'] is not True\n"
    "            or audit['raw_endpoint']['actual_final_parameters_checked'] is not True\n"
    "            or len(audit['frozen']['changed_trainable_tensors']) != 238\n"
    "            or audit['frozen']['frozen_tensors'] != 16\n"
    "            or audit['frozen']['unselected_parameters_and_all_buffers_bitwise_unchanged'] is not True")
exec(compile(SOURCE, __file__, 'exec'), globals())
