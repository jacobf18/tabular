"""Probe balanced native coverage at full native context size after V54 closes negative."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import diagnose_single_model_standardized_huber_initializer as closed
import preflight_single_model_standardized_teacher_huber as probe_blueprint
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
NATIVE_SHAPE = [2000, 1000]
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE + 2 * (2000 * 1000 - 256 * 128) * 8
DIAGNOSIS = ROOT / 'standardized_teacher_huber_v54_initializer_diagnosis_v1'
COLUMNS = ROOT / 'standardized_teacher_huber_v54_confirmation_columns_v1'
CLOSURE_HASHES = {
    str(closed.CLOSURE): closed.CLOSURE_SHA256,
    str(closed.TRAINING_AUDIT): closed.TRAINING_AUDIT_SHA256,
    str(closed.REPLAY / 'audit.json'): closed.REPLAY_AUDIT_SHA256,
    str(closed.REPLAY / 'gate.json'): closed.REPLAY_GATE_SHA256,
    str(closed.original.CONFIRMATION / 'audit.json'): closed.CONFIRMATION_AUDIT_SHA256,
    str(closed.original.CONFIRMATION / 'gate.json'): closed.CONFIRMATION_GATE_SHA256,
    str(DIAGNOSIS / 'config.json'): '49771e19f566b852416a4ab749595bf33a7097dd8eb574f3c27b73c993f0a2ed',
    str(DIAGNOSIS / 'audit.json'): '29bf6700ca0d4058679839dd8919d83e7575555c5b233075115f9ee0b469f6d8',
    str(DIAGNOSIS / 'RESULTS.md'): '7d3ef6094e0940bb7940b6fdf564dd99aa2ee116a433ce1635811d70231dd2d2',
    str(COLUMNS / 'config.json'): '68388650e204293b802544132ddc3e8f3fa5b66484509b18f08d8a7a69aa451d',
    str(COLUMNS / 'summary.json'): '81c8f13b4bbb336c759f0543004cfa9fd95c357bf875ccd750ee66a171bd320a',
    str(COLUMNS / 'OUTCOME.md'): '8267615ca2625f857b71e1910cf562564003c8ce7538a97e5dcfe40bfd9c8949',
    str(COLUMNS / 'columns.csv'): '27302e35121f83e302ccc14c0c36978e9657b1d84166af045d43ed3727b3a21e',
    str(COLUMNS / 'cases.csv'): '618fa786bebd666f60db9ce14f1e679e5ab9faf859740219e248408892a04f2a',
    str(COLUMNS / 'contributions.csv'): '02c2e0c78a6f7b76f84f1dc4d19b753fcf5027f01c13219fc5a5d5f1acc34376',
    str(COLUMNS / 'ratings.csv'): 'cb7a824ee41a8f3f0580ed879985bd00100f6287b26a1fca86be77a924cc517c',
}
POLICY_TEXT = (
    'One separately registered fresh28-task feasibility update after V54 complete negative '
    'closure: selected step50,57 accepted updates,7/32 failed fresh confirmation comparisons. '
    'Initialize only unchanged actual V29, never V54 or a feasibility endpoint. '
    + native.POLICY + ' ' + cohort_retention.POLICY
    + ' This separate registration authorizes one update and its original independent audits '
    'only. Use the first scheduled balanced component/orientation pair, but both native requested '
    'shapes are the full-trial maximum2000x1000 rather than the historical small first-batch '
    '256x128. Thus exercise2000x1000 and1000x2000 spectral contexts without extra tasks, seed '
    'selection or mask repair. This resource stress test is not the full64-attempt schedule, '
    'coverage proof or held-out qualification. Preserve all structured/rating specifications '
    'except fresh seeds,28 task slots/.5/.25/.25 masses,77,998,496 parameters/238 trainable '
    'tensors/16 frozen buffers,84 raw gradients,30 grouped protections,fresh5e-6 AdamW,.1 margin, '
    'fixed-half displacement,2x norm cap and exact rollback. Replay52 imputation-teacher '
    'components,50 retention components (24 V8/26 V29),26 targets and all28 tasks. Independently '
    'reconstruct26 standardized teacher-mean terms and two unchanged rating terms on every '
    'column and endpoint. Preserve24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves '
    'and16 MiB root/32 GiB scratch budgets; increase checkpoint budget from512 to576 MiB for '
    'the two full native component archives. No retry, full-trial authorization, probe warm '
    'start or qualification claim. Preserve all22 development guards,32 fresh confirmation '
    'comparisons and complete frozen480-comparison review for any later registered trial.'
)


def validate_diagnoses(attribution, columns):
    if (attribution['complete'] is not True or attribution['prediction_grids'] != 284
            or len(attribution['comparisons']) != 32
            or sum(row['candidate_failed'] for row in attribution['comparisons']) != 7
            or attribution['post_hoc_consumed_bank'] is not True
            or attribution['goal_success_not_assessed'] is not True
            or any(attribution[key] is not False for key in ('cuda_initialized', 'checkpoint_selection_performed',
                'confirmation_repeated', 'training_performed', 'real_benchmarks_evaluated'))):
        raise ValueError('Preserve the complete negative consumed-bank V29 attribution')
    if (columns['complete'] is not True or columns['cases'] != 284 or columns['imputation_cases'] != 252
            or columns['rating_cases'] != 32 or len(set(columns['roles'])) != 5
            or columns['queried_columns'] != 35227 or columns['common_eligible_columns'] != 35152
            or columns['unique_imputation_metric_checks'] != 5040 or columns['attribution_comparisons'] != 32
            or columns['rating_case_metric_rows'] != 64
            or any(columns[key] is not False for key in ('cuda_initialized', 'model_inference_performed',
                'training_performed', 'checkpoint_selection_performed', 'confirmation_repeated',
                'gate_changed', 'real_data_used'))):
        raise ValueError('Preserve all five-role column diagnostics without qualification')


def column_worker_running():
    script = 'analyze_single_model_standardized_huber_confirmation_columns.py'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(argument).name == script for argument in arguments):
            return True
    return False


def require_terminal(directory, state):
    if directory == COLUMNS:
        expected = {'state': 'complete', 'summary_sha256': CLOSURE_HASHES[str(COLUMNS / 'summary.json')]}
        if state != expected or column_worker_running():
            raise ValueError('Require the exact PID-free completed column-analysis receipt and no live worker')
        return
    expected = 'training_complete' if directory == closed.training.OUTPUT else 'complete'
    if (state['state'] != expected or type(state.get('pid')) is not int or state['pid'] <= 0
            or closed.original.process_identity(state['pid']) is not None):
        raise ValueError('Every original V54 scientific or diagnostic owner must be terminal')


def require_completed_v54():
    closed.require_closed_failure()
    evidence = dict(CLOSURE_HASHES)
    verify_hashes(evidence)
    validate_diagnoses(read(DIAGNOSIS / 'audit.json'), read(COLUMNS / 'summary.json'))
    directories = (closed.training.OUTPUT, closed.original.OUTPUT, closed.original.CONFIRMATION,
        closed.REPLAY, ROOT / 'standardized_teacher_huber_v54_audit_task_replay', DIAGNOSIS, COLUMNS)
    for directory in directories:
        path = directory / 'status.json'
        state = read(path)
        require_terminal(directory, state)
        evidence[str(path)] = sha256(path)
        config_path = directory / 'config.json'
        if not config_path.is_file():
            continue
        evidence[str(config_path)] = sha256(config_path)
        config = read(config_path)
        for key in ('input_sha256', 'source_sha256'):
            for source, digest in config.get(key, {}).items():
                if source in evidence and evidence[source] != digest:
                    raise ValueError('Never reconcile changed predecessor identities')
                evidence[source] = digest
    for source, digest in read(DIAGNOSIS / 'audit.json')['artifact_sha256'].items():
        if source in evidence and evidence[source] != digest:
            raise ValueError('Never reconcile changed attribution evidence')
        evidence[source] = digest
    verify_hashes(evidence)
    return evidence


for name in ('reference_identities', 'validate_saved_components', 'validate_replay_archive'):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Preserve closure-free reference behavior')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
RETENTION_REPLAY_SOURCE = probe_blueprint.RETENTION_REPLAY_SOURCE
exec(compile(RETENTION_REPLAY_SOURCE, __file__, 'exec'), globals())

NATIVE_PROBE_SOURCE = probe_blueprint.STANDARDIZED_PROBE_SOURCE
for old, new in {
    'standardized_teacher_huber_preflight_v1': 'balanced_native_coverage_preflight_v1',
    '1411100901, 1412100901': '1441100901, 1442100901',
    'require_completed_v53': 'require_completed_v54',
}.items():
    if old not in NATIVE_PROBE_SOURCE:
        raise ValueError(f'Missing explicit native-coverage probe adaptation: {old}')
    NATIVE_PROBE_SOURCE = NATIVE_PROBE_SOURCE.replace(old, new)
NATIVE_PROBE_SOURCE = replace_assignment(NATIVE_PROBE_SOURCE, 'POLICY_TEXT', POLICY_TEXT)
exec(compile(NATIVE_PROBE_SOURCE, __file__, 'exec'), globals())
WRITE_BUDGETS['checkpoint'] += 64 * 1024**2
component_unbalanced_batch = batch


def batch():
    original = component_unbalanced_batch()
    for task in original['tasks']:
        if task['cohort'] == 'legacy':
            task['specification']['shape'] = list(NATIVE_SHAPE)
    result = native.balance(original)
    retention.validate_task_mixture(result['tasks'])
    if blend_replay.storage_accounting(result)['reserved_checkpoint_bytes'] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError('Budget the complete full-size native teacher components')
    return result


REGISTER_SOURCE = probe_blueprint.REGISTER_SOURCE
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, '    sources.update(Path(module.__file__).resolve()',
    '    sources.update({REPO / "tests/test_single_model_balanced_native_coverage.py", '
    'REPO / "tests/test_single_model_balanced_native_coverage_preflight.py", '
    'REPO / "scripts/audit_single_model_balanced_native_coverage_probe.py", '
    'REPO / "tests/test_single_model_balanced_native_coverage_probe_audit.py", '
    'REPO / "scripts/recheck_single_model_balanced_native_coverage_probe.py", '
    'REPO / "tests/test_single_model_balanced_native_coverage_probe_recheck.py", '
    'REPO / "scripts/recheck_single_model_standardized_teacher_huber_probe.py", '
    'REPO / "tests/test_single_model_standardized_teacher_huber_probe_recheck.py"})\n'
    '    sources.update(Path(module.__file__).resolve()')
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'config = {"policy": POLICY,',
    'config = {"policy": POLICY, "native_coverage": native.SPECIFICATION, "native_probe_shape": NATIVE_SHAPE,')
exec(compile(REGISTER_SOURCE, __file__, 'exec'), globals())
original_verify = verify


def verify(output):
    config = original_verify(output)
    if (config['native_coverage'] != native.SPECIFICATION or config['native_probe_shape'] != NATIVE_SHAPE
            or config['structured_component_balance'] != components.SPECIFICATION):
        raise ValueError('Preserve distinct native coverage, resource stress shape and structured balance')
    return config


main = types.FunctionType(probe_blueprint.main.__code__, globals(), 'main', probe_blueprint.main.__defaults__)


if __name__ == '__main__':
    main()
