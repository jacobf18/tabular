"""Fresh full-context quarter-displacement probe after complete half-step rejection."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import analyze_single_model_native_probe_linearization as native_diagnosis
import preflight_single_model_balanced_native_coverage as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_balanced_native_coverage as native
import single_model_balanced_structured_components as components
import single_model_standardized_teacher_huber as cohort_retention
import single_model_stronger_mean_distillation as stronger
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
diagnosis = probe_blueprint.diagnosis
NATIVE_SHAPE = list(probe_blueprint.NATIVE_SHAPE)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
PREVIOUS = probe_blueprint.OUTPUT
MEAN = PREVIOUS.with_name(PREVIOUS.name + '_mean_audit')
DECOMPOSITION = native_diagnosis.OUTPUT
CLOSURE_HASHES = {
    str(PREVIOUS / 'config.json'): '1b3b5fae260b205702287b5fa858b199531b82d5fee138823f6b393e24e9416b',
    str(PREVIOUS / 'result.json'): 'fdac95db8b02da8e0dcb964c45b7b0431cfde6a566ffba76c2dc7f4f769e1e60',
    str(PREVIOUS / 'audit.json'): 'a08dc0cde6e96d0a04e59e026d9da560e9163898341cc59a9ddb125c4b8ce18b',
    str(PREVIOUS / 'gate.json'): '5ab00d5dc954c3cb6e6533a59a46d69e2505b390136e2ad27edaf0753445b1ea',
    str(PREVIOUS / 'retention_audit.json'): 'c28250885fa34f5360914e1586c0cd41c6a39779a2b60cdde44ee7905895eb07',
    str(PREVIOUS / 'external_recheck.json'): '39a8561b044399e13769939335d1bb8601e22e238d4c03acf086e4e6c158fca9',
    str(PREVIOUS / 'OUTCOME.md'): 'ad83ce70bdb6e54dd6ac62939be1448e00773b9d33eba511748b18897599a999',
    str(MEAN / 'audit.json'): '1a5161fe19b9b46f458f2ac8e0d13352f2039ca2ce06dff2f72ffe2c5a9a314a',
    str(DECOMPOSITION / 'config.json'): '72fce5d54628e5e1e22696c59284575acbf6c8d47a10cbe11a5de26de42cd02d',
    str(DECOMPOSITION / 'summary.json'): 'bf4c786ec435a9e846f1438f6bc3f0a1781bd7346af880451bbf04cc279e2c1a',
    str(DECOMPOSITION / 'point_changes.csv'): 'fcd43369100ebae7ca0dc77c6c1a1516dcb65b8bda7fb084eb46c9f2c6b18a20',
    str(DECOMPOSITION / 'grouped_changes.csv'): 'b38676bb0f6d7e9349de369d44e18fb8898e80b7993c0fba484ebf0ef906d0e1',
    str(DECOMPOSITION / 'OUTCOME.md'): '63e38fdb7d8e49c30a17ab34dfc8c6a3e9cfc71add201ef55fbb66b4f344e85c',
}
POLICY_TEXT = (
    'One separately registered fresh28-task full-context feasibility update after the original '
    'balanced-native half-step probe closes rejected with exact rollback and its complete '
    'saved-step decomposition closes. Initialize unchanged actual V29, never a probe endpoint. '
    'Change only the projected displacement contraction from0.5 to0.25 and fresh data/backward '
    'streams. Never reevaluate a fraction on consumed contexts or retry that rejected update. '
    'Preserve the balanced native component/orientation pair0/false and4/true, both requested '
    '2000x1000 shapes (actual spectral2000x1000 and1000x2000), all24 balanced structured '
    'specifications, both rating tasks,28 slots and.5/.25/.25 masses. Retain standardized '
    'teacher-Huber20, every NLL/CDF/query/point/rating term, observed-only blended imputation '
    'teachers/specialist rating teacher, native V29 retention, fixed-half V8/V29 structured '
    'retention and no rating retention. Fixed-half retention weights are NOT the new quarter '
    'optimizer displacement. Keep fresh5e-6 AdamW/original accepted moments,.1 margin,84 raw '
    'gradients,30 grouped protections,2x norm cap and exact rollback. One77,998,496-parameter '
    'network,238 trainable tensors,16 frozen buffers and unchanged inference; no ensemble. '
    'Replay52 teacher components,50 retention components/26 targets and all28 tasks/56 '
    'original endpoint grids; independently audit every actual fresh physical column, never '
    'assume intrinsic structured dimensions repeat. Use the explicit fixed_quarter_step audit '
    'schema. Preserve the old negative gate and its unperformed parameter-autograd/independent '
    'rejected-neural-endpoint replay limitations. No full-trial authorization or qualification '
    'claim; any later trial must retain all22 development guards,32 fresh confirmation '
    'comparisons and the complete frozen480-comparison real review. Keep24 BLAS/4 Torch '
    'threads,.9 allocator cap,18/6/50 GiB reserves and576 MiB/16 MiB/32 GiB checkpoint/root/'
    'scratch allowances. Smaller steps can slow learning; no fraction search, data replacement, '
    'case/family exception, reference swap or relaxed guard.'
)


def validate_previous(reports):
    native_diagnosis.validate_reports(*(reports[name] for name in (
        'status', 'result', 'gate', 'audit', 'retention', 'review')))
    audit, mean, summary = (reports[name] for name in ('audit', 'mean', 'decomposition'))
    if (audit['raw_endpoint']['fixed_half_step']['fraction'] != .5
            or audit['raw_endpoint']['fixed_half_step']['parameters'] != 77998496
            or audit['raw_endpoint']['rejected_candidate_requires_gpu_reconstruction'] is not True
            or audit['raw_endpoint']['finite_point_scores_require_neural_replay'] is not True
            or audit['autograd_replayed'] is not False or audit['frozen']['changed_trainable_tensors'] != []
            or mean['complete'] is not True or mean['tasks'] != 28 or mean['columns'] != 6875
            or mean['imputation_reweighted_tasks'] != 26 or mean['unchanged_rating_tasks'] != 2
            or mean['adverse_teacher_mean_tasks'] != 11 or mean['cuda_initialized'] is not False
            or summary['complete'] is not True or summary['tasks'] != 28 or summary['point_slots'] != 56
            or summary['grouped_guards'] != 30 or summary['original_accepted'] is not False
            or summary['original_accepted_updates'] != 0 or len(summary['adverse_points']) != 3
            or len(summary['adverse_groups']) != 1
            or any(summary[key] is not False for key in ('model_inference_performed', 'cuda_initialized',
                'parameter_autograd_replayed', 'new_scientific_seed', 'training_performed',
                'checkpoint_selection_performed', 'confirmation_repeated', 'real_data_used', 'guard_changed',
                'full_trial_authorized', 'goal_achieved'))):
        raise ValueError('Preserve the fully audited original half-step rejection and complete limited-scope diagnosis')


def require_completed_native_probe():
    evidence = dict(CLOSURE_HASHES)
    verify_hashes(evidence)
    native_diagnosis.require_closed()
    reports = {name: read(PREVIOUS / filename) for name, filename in (
        ('status', 'status.json'), ('result', 'result.json'), ('gate', 'gate.json'), ('audit', 'audit.json'),
        ('retention', 'retention_audit.json'), ('review', 'external_recheck.json'))}
    reports.update(mean=read(MEAN / 'audit.json'), decomposition=read(DECOMPOSITION / 'summary.json'))
    validate_previous(reports)
    for directory in (PREVIOUS, DECOMPOSITION):
        path = directory / 'status.json'
        state = read(path)
        if (state['state'] != 'complete' or type(state.get('pid')) is not int or state['pid'] <= 0
                or probe_blueprint.closed.original.process_identity(state['pid']) is not None):
            raise ValueError('Both original scientific and diagnostic owners must be stopped')
        evidence[str(path)] = sha256(path)
    for directory in (PREVIOUS, MEAN, DECOMPOSITION):
        path = directory / 'config.json'
        evidence[str(path)] = sha256(path)
        config = read(path)
        for key in ('input_sha256', 'source_sha256'):
            for source, digest in config.get(key, {}).items():
                if source in evidence and evidence[source] != digest:
                    raise ValueError('Never reconcile changed original evidence or registered sources')
                evidence[source] = digest
    verify_hashes(evidence)
    return evidence


for name in ('reference_identities', 'validate_saved_components', 'validate_replay_archive'):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Reference behavior must remain closure-free')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
RETENTION_REPLAY_SOURCE = probe_blueprint.RETENTION_REPLAY_SOURCE
exec(compile(RETENTION_REPLAY_SOURCE, __file__, 'exec'), globals())

QUARTER_PROBE_SOURCE = probe_blueprint.NATIVE_PROBE_SOURCE
for old, new in {
    'balanced_native_coverage_preflight_v1': 'balanced_native_quarter_preflight_v1',
    '1441100901, 1442100901': '1451100901, 1452100901',
    'require_completed_v54': 'require_completed_native_probe',
    'single_model_fixed_half_protection as paired': 'single_model_fixed_quarter_protection as paired',
}.items():
    if old not in QUARTER_PROBE_SOURCE:
        raise ValueError(f'Missing explicit quarter-probe adaptation: {old}')
    QUARTER_PROBE_SOURCE = QUARTER_PROBE_SOURCE.replace(old, new)
QUARTER_PROBE_SOURCE = replace_assignment(QUARTER_PROBE_SOURCE, 'POLICY_TEXT', POLICY_TEXT)
exec(compile(QUARTER_PROBE_SOURCE, __file__, 'exec'), globals())
WRITE_BUDGETS['checkpoint'] += 64 * 1024**2
component_unbalanced_batch = batch
batch = types.FunctionType(probe_blueprint.batch.__code__, globals(), 'batch', probe_blueprint.batch.__defaults__)

REGISTER_SOURCE = replace_once(probe_blueprint.REGISTER_SOURCE, '    sources.update(Path(module.__file__).resolve()',
    '    sources.update({REPO / "tests/test_single_model_fixed_quarter_protection.py", '
    'REPO / "tests/test_single_model_balanced_native_quarter_preflight.py", '
    'REPO / "scripts/audit_single_model_balanced_native_quarter_probe.py", '
    'REPO / "tests/test_single_model_balanced_native_quarter_probe_audit.py", '
    'REPO / "scripts/recheck_single_model_balanced_native_quarter_probe.py", '
    'REPO / "tests/test_single_model_balanced_native_quarter_probe_recheck.py"})\n'
    '    sources.update(Path(module.__file__).resolve()')
exec(compile(REGISTER_SOURCE, __file__, 'exec'), globals())
original_verify = verify
verify = types.FunctionType(probe_blueprint.verify.__code__, globals(), 'verify', probe_blueprint.verify.__defaults__)
main = types.FunctionType(probe_blueprint.main.__code__, globals(), 'main', probe_blueprint.main.__defaults__)


if __name__ == '__main__':
    main()
