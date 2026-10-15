"""Probe standardized teacher residuals after V53's preserved negative closure."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import preflight_single_model_balanced_structured_components as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_standardized_teacher_huber as cohort_retention
import single_model_balanced_structured_components as components
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
diagnosis = probe_blueprint.diagnosis
ANALYSIS = ROOT / 'standardized_teacher_huber_endpoint_analysis_v1'
CLOSURE_DOCUMENTS = {
    'training': ROOT / 'balanced_structured_components_v53_audit/audit.json',
    'endpoint': ROOT / 'balanced_structured_components_v53_endpoint_replay/audit.json',
    'closure': ROOT / 'balanced_structured_components_v53_closure_audit/result.json',
    'gate': ROOT / 'balanced_structured_components_v53_confirmation/gate.json',
    'retention': Path('/mnt/matrixformer_scratch/balanced_structured_components_v53/anchor_retention_replay/summary.json'),
    'blend': Path('/mnt/matrixformer_scratch/balanced_structured_components_v53/all_imputation_blend_teacher_components/summary.json'),
    'analysis': ANALYSIS / 'receipt.json',
}
CLOSURE_HASHES = {
    str(CLOSURE_DOCUMENTS['training']): '32519e3d1144c1cfc5cb69963c45b122820d684956726aa13ef2949cbbab8919',
    str(CLOSURE_DOCUMENTS['endpoint']): '72fbbd56c21a7d2316f9cf91c5e8390ecacf61355f88577e2bf621082a1546f7',
    str(CLOSURE_DOCUMENTS['closure']): 'fdaf6782d651054e181d7e48546bcb5b6f2e1be8de4ab3d5519c1dcdfa776167',
    str(CLOSURE_DOCUMENTS['gate']): '3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94',
    str(CLOSURE_DOCUMENTS['retention']): 'e4662d452c5970d14d506d3511247891258258b85fb1677caea57823a7e86083',
    str(CLOSURE_DOCUMENTS['blend']): 'eceeaae65e4abf7d9d87e1393f60d2a7341133044c29edae0f9c8e9082696f99',
    str(CLOSURE_DOCUMENTS['analysis']): '80b66c5903ac745b09774b4c99d165508a5b9dc8375ae2505e78cf79eb9d87c5',
    str(ANALYSIS / 'config.json'): '3f1beaf1379a8eb53c30456572645262314c66a2958d508eeee02a630bf67e9c',
    str(ROOT / 'balanced_structured_components_v53_pipeline/status.json'): 'f46ec75e9ca7b6c73b8f12a0af99aef11770b18fbbdbb22aae5b2b73b0ec8d2d',
    str(ROOT / 'balanced_structured_components_v53_capacity_recovery/status.json'): '6831cae0f074892a37c18cf7480279a921e7a2d701a0d2f7a505d15a22e3f0c5',
}
POLICY_TEXT = (
    'One separately registered fresh28-task feasibility update after V53 complete negative '
    'scientific closure, preserving its original capacity failure and later empty-lock '
    'bookkeeping failure as failures. Bind the successful additive read-only closure. '
    'Initialize only unchanged actual V29, never V53 or a feasibility endpoint. '
    + cohort_retention.POLICY
    + ' This registration authorizes one fresh feasibility update and its full original audits '
    'only. Preserve balanced structured component assignments, native/rating specifications '
    'except fresh seeds, all28 task slots/.5/.25/.25 masses,77,998,496 parameters/238 trainable '
    'tensors/16 frozen buffers,84 raw gradients,30 grouped protections,fresh5e-6 AdamW,.1 margin, '
    'fixed-half displacement,2x norm cap and exact rollback. Independently replay52 original '
    'imputation-teacher components,50 retention components (24 V8/26 V29),26 targets and every '
    'task, including both specialist rating contexts. Independently reconstruct26 standardized '
    'teacher-mean terms and two unchanged rating terms on every original column and endpoint. '
    'Keep24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB '
    'probe budgets. No retry, mask repair, probe warm start, full-trial authorization or '
    'qualification claim. Preserve all22 development guards,32 fresh confirmation comparisons '
    'and the complete frozen MissBench/UCI/four-RecSys/matched-LLM campaign/full480-comparison '
    'review for any later separately registered trial.'
)


def validate_closed_v53(reports):
    training_report, endpoint, closure, gate = (reports[name] for name in ('training', 'endpoint', 'closure', 'gate'))
    retention_report, blend, analysis = (reports[name] for name in ('retention', 'blend', 'analysis'))
    if (training_report['complete'] is not True or training_report['complete_training_audit'] is not True
            or training_report['through_step'] != 64 or training_report['accepted_updates'] != 59
            or training_report['best']['step'] != 0 or training_report['independently_regenerated_training_tasks'] != 1792
            or training_report['independently_regenerated_rating_contexts'] != 128
            or training_report['independently_regenerated_structured_draws'] != 1536
            or training_report['admission_attempts']['data_rejected_steps'] != []
            or training_report['autograd_replayed'] is not False
            or training_report['full_optimizer_trajectory_replayed'] is not False
            or training_report['confirmation_evaluated'] is not False
            or endpoint['complete'] is not True or endpoint['endpoint_grids'] != 56
            or endpoint['selected_grids'] != 0 or endpoint['cuda_initialized'] is not False
            or retention_report['complete'] is not True or retention_report['admitted_batches'] != 64
            or retention_report['reference_grids'] != 1664 or retention_report['component_grids'] != 3200
            or retention_report['component_role_counts'] != {'v8': 1536, 'v29': 1664}
            or blend['complete'] is not True or blend['training_tasks_checked'] != 1792
            or blend['blend_tasks'] != 1664 or blend['raw_component_grids'] != 3328
            or blend['query_truth_used_for_routing'] is not False):
        raise ValueError('Preserve every original V53 task, teacher, retention component and audit limitation')
    if (closure['complete'] is not True or closure['unqualified'] is not True or closure['gate'] != gate
            or closure['campaign'] is not None or closure['review'] is not None
            or closure['confirmation_cases_generated'] != 0 or closure['scientific_stages_repeated'] is not False
            or closure['normal_empty_lock_verified'] is not True
            or gate['passed'] is not False or gate['selected_step'] != 0
            or gate['confirmation_evaluated'] is not False or gate['real_benchmark_success'] is not False
            or closure['original_failed_pipeline'] != str(ROOT / 'balanced_structured_components_v53_pipeline')
            or closure['failed_capacity_recovery'] != str(ROOT / 'balanced_structured_components_v53_capacity_recovery')):
        raise ValueError('Require V53 additive negative closure without relabeling either preserved failure')
    if (analysis['complete'] is not True or analysis['task_role_rows'] != 56
            or analysis['imputation_rows'] != 52 or analysis['rating_rows'] != 4
            or analysis['retention_coefficients_bitwise_unchanged'] is not True
            or analysis['rating_terms_and_coordinate_gradients_bitwise_unchanged'] is not True
            or any(analysis[key] is not False for key in ('model_inference_performed', 'new_scientific_seeds',
                'training_performed', 'checkpoint_selection_performed', 'confirmation_evaluated', 'real_data_used', 'cuda_initialized'))):
        raise ValueError('Keep the complete saved-grid diagnosis without treating it as an efficacy result')


def require_completed_v53():
    evidence = dict(CLOSURE_HASHES)
    verify_hashes(evidence)
    reports = {name: read(path) for name, path in CLOSURE_DOCUMENTS.items()}
    validate_closed_v53(reports)
    states = {'balanced_structured_components_v53': 'training_complete',
        'balanced_structured_components_v53_audit_task_replay': 'complete',
        'balanced_structured_components_v53_endpoint_replay': 'complete',
        'balanced_structured_components_v53_confirmation': 'complete',
        'balanced_structured_components_v53_pipeline': 'failed',
        'balanced_structured_components_v53_capacity_recovery': 'failed',
        'balanced_structured_components_v53_closure_audit': 'complete'}
    for name, expected in states.items():
        path = ROOT / name / 'status.json'
        state = read(path)
        if state['state'] != expected or diagnosis.original.process_identity(state['pid']) is not None:
            raise ValueError('Every exact original V53 owner must be terminal, with both failures preserved')
        evidence[str(path)] = sha256(path)
    if any((ROOT / 'balanced_structured_components_v53' / name).exists()
            or (ROOT / 'balanced_structured_components_v53' / name).is_symlink()
            for name in ('selected_checkpoint.pth', 'best_validation_checkpoint.pth')):
        raise ValueError('The negative V53 trial must not publish a selected checkpoint')
    configurations = [ROOT / name / 'config.json' for name in ('balanced_structured_components_v53',
        'balanced_structured_components_v53_pipeline', 'balanced_structured_components_v53_capacity_recovery',
        'balanced_structured_components_v53_closure_audit', 'v53_root_archive_preservation_v1')]
    configurations.append(ANALYSIS / 'config.json')
    mappings = [report.get('artifact_sha256', {}) for report in reports.values()]
    mappings.append(reports['analysis']['output_sha256'])
    for path in configurations:
        config = read(path)
        evidence[str(path)] = sha256(path)
        mappings.extend(config.get(key, {}) for key in ('input_sha256', 'source_sha256', 'protected_v53_sources'))
    for mapping in mappings:
        for path, digest in mapping.items():
            if path in evidence and evidence[path] != digest:
                raise ValueError('Never reconcile changed predecessor evidence or registered source identities')
            evidence[path] = digest
    verify_hashes(evidence)
    return evidence


for name in ('reference_identities', 'validate_saved_components', 'validate_replay_archive'):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Probe callbacks must bind unchanged reference behavior without closures')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
RETENTION_REPLAY_SOURCE = probe_blueprint.RETENTION_REPLAY_SOURCE
exec(compile(RETENTION_REPLAY_SOURCE, __file__, 'exec'), globals())

STANDARDIZED_PROBE_SOURCE = probe_blueprint.BALANCED_PROBE_SOURCE
for old, new in {
    'single_model_all_imputation_column_mean as retention': 'single_model_standardized_teacher_huber as retention',
    'balanced_structured_components_preflight_v1': 'standardized_teacher_huber_preflight_v1',
    '1381100901, 1382100901': '1411100901, 1412100901',
    'require_completed_v52': 'require_completed_v53',
}.items():
    if old not in STANDARDIZED_PROBE_SOURCE:
        raise ValueError(f'Missing explicit standardized-Huber probe adaptation: {old}')
    STANDARDIZED_PROBE_SOURCE = STANDARDIZED_PROBE_SOURCE.replace(old, new)
STANDARDIZED_PROBE_SOURCE = replace_assignment(STANDARDIZED_PROBE_SOURCE, 'POLICY_TEXT', POLICY_TEXT)
exec(compile(STANDARDIZED_PROBE_SOURCE, __file__, 'exec'), globals())
component_unbalanced_batch = batch


def batch():
    result = components.balance(component_unbalanced_batch())
    retention.validate_task_mixture(result['tasks'])
    if blend_replay.storage_accounting(result)['reserved_checkpoint_bytes'] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError('Preserve full teacher-component replay and prospective capacity')
    return result


REGISTER_SOURCE = replace_once(REGISTER_SOURCE, '    sources.update(Path(module.__file__).resolve()',
    '    sources.update({REPO / "tests/test_single_model_standardized_teacher_huber.py", '
    'REPO / "tests/test_single_model_standardized_teacher_huber_analysis.py", '
    'REPO / "scripts/analyze_single_model_standardized_teacher_huber.py", '
    'REPO / "tests/test_single_model_standardized_teacher_huber_preflight.py", '
    'REPO / "scripts/audit_single_model_standardized_teacher_huber_probe.py", '
    'REPO / "tests/test_single_model_standardized_teacher_huber_probe_audit.py"})\n'
    '    sources.update(Path(module.__file__).resolve()')
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'config = {"policy": POLICY,',
    'config = {"policy": POLICY, "structured_component_balance": components.SPECIFICATION,')
exec(compile(REGISTER_SOURCE, __file__, 'exec'), globals())
main = types.FunctionType(probe_blueprint.main.__code__, globals(), 'main', probe_blueprint.main.__defaults__)


if __name__ == '__main__':
    main()
