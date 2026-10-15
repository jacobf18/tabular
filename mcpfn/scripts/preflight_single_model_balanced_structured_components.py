"""Probe balanced structured components after V52's complete negative closure."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import preflight_single_model_all_imputation_column_mean as probe_blueprint
import single_model_all_imputation_blend_replay as blend_replay
import single_model_all_imputation_column_mean as cohort_retention
import single_model_balanced_structured_components as components
import single_model_stronger_mean_distillation as stronger
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


ROOT = probe_blueprint.ROOT
ARITHMETIC = dict(probe_blueprint.ARITHMETIC)
BLEND_COMPONENT_ALLOWANCE = probe_blueprint.BLEND_COMPONENT_ALLOWANCE
diagnosis = probe_blueprint.diagnosis
COVERAGE = Path('/mnt/matrixformer_scratch/all_imputation_column_mean_v52_prior_coverage_v1')
CLOSURE_HASHES = {
    'all_imputation_column_mean_v52_audit/audit.json': 'b528e66571bd1ec9f85441203d38592895173e6dc461eb5a4902f6412161af5a',
    'all_imputation_column_mean_v52_endpoint_replay/audit.json': 'df9795201b06699987c363bd556b995735ec5bc469fd1e5d0f1c6e0c79f2df66',
    'all_imputation_column_mean_v52_pipeline/result.json': '91eaa6d602dd48200d6e4a26e9f7273357d7f150841b7d80033824e4c188b91b',
    'all_imputation_column_mean_v52_confirmation/gate.json': '3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94',
    'all_imputation_column_mean_v52_terminal_review_v1/closure.json': 'a36628310712139a311f72e20fb9098b432af5e976ed45dc29306f2834ea8ee6',
    'all_imputation_column_mean_v52_terminal_review_v1/OUTCOME.md': 'd25e7896df4a59eb2305b9b31eb2eae1c2b4a1d38063d19178d496c1b9a3ceaf',
    str(COVERAGE / 'receipt.json'): 'c64574a6e8dec48ad609c82affd458fc780b60efb7e1d2c38717ce31f41d84f7',
    str(COVERAGE / 'OUTCOME.md'): '41ebe98f536d2f814cae9bc69df4c8d75b12d2da1e3dccf3ecdf81360f832e2b',
}
CLOSURE_DOCUMENTS = {
    'training': ROOT / 'all_imputation_column_mean_v52_audit/audit.json',
    'endpoint': ROOT / 'all_imputation_column_mean_v52_endpoint_replay/audit.json',
    'pipeline': ROOT / 'all_imputation_column_mean_v52_pipeline/result.json',
    'gate': ROOT / 'all_imputation_column_mean_v52_confirmation/gate.json',
    'external': ROOT / 'all_imputation_column_mean_v52_terminal_review_v1/closure.json',
    'retention': Path('/mnt/matrixformer_scratch/all_imputation_column_mean_v52/anchor_retention_replay/summary.json'),
    'blend': Path('/mnt/matrixformer_scratch/all_imputation_column_mean_v52/all_imputation_blend_teacher_components/summary.json'),
}
POLICY_TEXT = (
    'One separately registered fresh28-task feasibility update after V52 completes its original '
    'full audit and negative no-selection qualification. Initialize only from unchanged actual '
    'V29, never V52 or a diagnostic endpoint. ' + components.POLICY
    + ' This registration authorizes one feasibility update and its independent audits only. '
    'Keep the existing all-imputation count-shrunk equal-column mean20 Huber objective, native '
    'V29 retention, fixed-half V8/V29 structured retention and no rating retention. Preserve '
    '77,998,496 parameters/238 trainable tensors,16 frozen buffers,84 raw gradients,30 grouped '
    'protections,fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm cap and exact rollback. '
    'Replay52 original imputation-teacher components,50 retention components (24 V8/26 V29) and '
    '26 complete targets. Independently reconstruct all26 column-weighted teacher-mean terms '
    'and both unchanged rating terms. General remains verified but supplies no retention target. '
    'Preserve24 BLAS/4 Torch threads,.9 allocator cap,18/6/50 GiB reserves and512 MiB/16 MiB/32 GiB '
    'probe budgets, with current capacity admission and honest actual geometry. No retry, mask '
    'repair, full-trial authorization, probe warm start, confirmation, consumed-bank selection '
    'or benchmark-success claim. Keep all22 development guards,32 fresh confirmation comparisons '
    'and every frozen real suite/full review for any later separately registered trial.'
)


def validate_closed_v52(reports):
    training_report, endpoint, pipeline, gate = (reports[name] for name in ('training', 'endpoint', 'pipeline', 'gate'))
    external, retention_report, blend = (reports[name] for name in ('external', 'retention', 'blend'))
    if (training_report['complete'] is not True or training_report['complete_training_audit'] is not True
            or training_report['through_step'] != 64 or training_report['accepted_updates'] != 57
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
        raise ValueError('Preserve every original V52 task, component, outcome and audit limitation')
    if (pipeline['complete'] is not True or pipeline['gate'] != gate
            or pipeline['campaign'] is not None or pipeline['review'] is not None
            or gate['passed'] is not False or gate['confirmation_evaluated'] is not False
            or gate['selected_step'] != 0 or gate['real_benchmark_success'] is not False
            or external['complete'] is not True or external['outcome'] != 'negative_no_eligible_checkpoint'
            or external['training_attempts'] != 64 or external['accepted_updates'] != 57
            or external['rejected_attempts'] != [1, 12, 45, 51, 52, 54, 60]
            or external['original_workers_terminal'] is not True
            or any(external[name] is not False for name in ('goal_achieved', 'confirmation_evaluated',
                'candidate_real_benchmarks_run', 'model_inference_performed_by_closure', 'gate_changed'))):
        raise ValueError('Require V52 exact stopped negative closure without confirmation or real data')


def require_completed_v52():
    evidence = {str(ROOT / name): digest for name, digest in CLOSURE_HASHES.items()}
    verify_hashes(evidence)
    reports = {name: read(path) for name, path in CLOSURE_DOCUMENTS.items()}
    validate_closed_v52(reports)
    for name in ('all_imputation_column_mean_v52', 'all_imputation_column_mean_v52_audit_task_replay',
            'all_imputation_column_mean_v52_endpoint_replay', 'all_imputation_column_mean_v52_confirmation',
            'all_imputation_column_mean_v52_pipeline'):
        path = ROOT / name / 'status.json'
        state = read(path)
        if (state['state'] not in {'complete', 'training_complete'}
                or diagnosis.original.process_identity(state['pid']) is not None):
            raise ValueError('Stop every original V52 worker before a new feasibility experiment')
        evidence[str(path)] = sha256(path)
    if any((ROOT / 'all_imputation_column_mean_v52' / name).exists()
            or (ROOT / 'all_imputation_column_mean_v52' / name).is_symlink()
            for name in ('selected_checkpoint.pth', 'best_validation_checkpoint.pth')):
        raise ValueError('Negative V52 closure must not publish a selected or qualified-best alias')
    analysis = read(COVERAGE / 'receipt.json')
    if (analysis['complete'] is not True or analysis['attempts'] != 64
            or analysis['accepted_attempts'] != 57 or analysis['tasks'] != 1792
            or analysis['structured_training_fractions'] != {'anchor_latent_mcar': .6,
                'low_rank_mcar': .15, 'slow_tail_mcar': .1, 'steep_tail_mcar': .15}
            or any(analysis[key] is not False for key in ('new_model_inference', 'new_seed', 'training_modified',
                'causal_effect_claimed', 'checkpoint_selected', 'gate_changed', 'full_training_audit_claimed',
                'real_benchmark_evaluated', 'goal_achieved'))):
        raise ValueError('Preserve complete consumed-prior coverage without an efficacy claim')
    for mapping in [report.get('artifact_sha256', {}) for report in reports.values()] + [
            reports['external']['bound_sha256'], reports['external']['registered_source_sha256'],
            analysis['input_sha256'], analysis['registered_source_sha256'], analysis['output_sha256']]:
        for path, digest in mapping.items():
            if path in evidence and evidence[path] != digest:
                raise ValueError('Every predecessor path must retain one immutable identity')
            evidence[path] = digest
    evidence.update({str(path): sha256(path) for path in CLOSURE_DOCUMENTS.values()})
    verify_hashes(evidence)
    return evidence


for name in ('reference_identities', 'validate_saved_components', 'validate_replay_archive'):
    function = getattr(probe_blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Probe helpers must preserve closure-free reference behavior')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
RETENTION_REPLAY_SOURCE = probe_blueprint.RETENTION_REPLAY_SOURCE
exec(compile(RETENTION_REPLAY_SOURCE, __file__, 'exec'), globals())

BALANCED_PROBE_SOURCE = probe_blueprint.ALL_IMPUTATION_PROBE_SOURCE
for old, new in {
    'all_imputation_column_mean_preflight_v1': 'balanced_structured_components_preflight_v1',
    '1351100901, 1352100901': '1381100901, 1382100901',
    'require_completed_v51': 'require_completed_v52',
}.items():
    if old not in BALANCED_PROBE_SOURCE:
        raise ValueError(f'Missing explicit balanced-prior probe adaptation: {old}')
    BALANCED_PROBE_SOURCE = BALANCED_PROBE_SOURCE.replace(old, new)
BALANCED_PROBE_SOURCE = replace_assignment(BALANCED_PROBE_SOURCE, 'POLICY_TEXT', POLICY_TEXT)
exec(compile(BALANCED_PROBE_SOURCE, __file__, 'exec'), globals())
component_unbalanced_batch = batch


def batch():
    result = components.balance(component_unbalanced_batch())
    retention.validate_task_mixture(result['tasks'])
    if blend_replay.storage_accounting(result)['reserved_checkpoint_bytes'] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError('Preserve complete teacher-component replay and prospective storage accounting')
    return result


REGISTER_SOURCE = replace_once(REGISTER_SOURCE, '    sources.update(Path(module.__file__).resolve()',
    '    sources.update({REPO / "tests/test_single_model_balanced_structured_components.py", '
    'REPO / "tests/test_single_model_balanced_structured_components_preflight.py", '
    'REPO / "scripts/audit_single_model_balanced_structured_components_probe.py", '
    'REPO / "tests/test_single_model_balanced_structured_components_probe_audit.py"})\n'
    '    sources.update(Path(module.__file__).resolve()')
REGISTER_SOURCE = replace_once(REGISTER_SOURCE, 'config = {"policy": POLICY,',
    'config = {"policy": POLICY, "structured_component_balance": components.SPECIFICATION,')
exec(compile(REGISTER_SOURCE, __file__, 'exec'), globals())
main = types.FunctionType(probe_blueprint.main.__code__, globals(), 'main', probe_blueprint.main.__defaults__)


if __name__ == '__main__':
    main()
