"""One prospective fixed-strength teacher-mean probe after complete V45 closure."""

import ast
import fcntl
from pathlib import Path
import types

import preflight_single_model_all_imputation_blend_teacher as ALL_BLEND_BLUEPRINT
from preflight_single_model_all_imputation_blend_teacher import diagnosis
import preflight_single_model_uniform_v8_retention as previous_probe
import single_model_all_imputation_blend_replay as blend_replay
import single_model_stronger_mean_distillation as stronger
import single_model_uniform_v8_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once


ARITHMETIC = dict(ALL_BLEND_BLUEPRINT.ARITHMETIC)
ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005')
CLOSED_REPORTS = {
    'all_imputation_blend_v45/config.json': 'b672f96797c02b4ca31988d142c3caa32fdc514702fe59c84c1de72df41d9f28',
    'all_imputation_blend_v45_audit/audit.json': 'de53f9b1158651670717092bbf45ab602605ccb5f34be1dc4a96d6b5dfcc5ff1',
    'all_imputation_blend_v45_endpoint_replay/audit.json': '3f04623e33f24b2771c2f808a9a74c3d654fe37ffaf6b6cf15fdc177aa6f21ad',
    'all_imputation_blend_v45_confirmation/gate.json': '40399f7013bc1de4bc7aa31c8e81802f3dc7820d2e3ad1280e431468541765c2',
    'all_imputation_blend_v45_confirmation/audit.json': '858d3fad20e00c3e4a67ada58a81d4f52341814ac1530085bb092d003d835fea',
    'all_imputation_blend_v45_pipeline/result.json': 'ce80ca428026c44e1cd771d0780d0f73ac4dcd51e626a464e17f693b84d2969b',
    'all_imputation_blend_v45_benchmarks/campaign_audit.json': 'e636f9624ce199555aa36a3f73aa02e959f1a43787077e87565cf3f87d56bc5e',
    'all_imputation_blend_v45_review/result.json': 'bf6fc0db1d3b6abea4ce46c17c6beffb04811dd66f7398605e19e91ce625b8a9',
}
SELECTED_SHA256 = 'dc429579b5504ab09411793cfb5b60394e23f356df813363e050544b3f1acb76'
ALIGNMENT = Path('/mnt/matrixformer_scratch/single_model_goal_20261005_owned/all_imputation_blend_v45_teacher_alignment_v1')
ALIGNMENT_SHA256 = '7ed699378d6b3f68495735a1b9082642961072774555a946f6a5f1f86e99a299'
BLEND_COMPONENT_ALLOWANCE = ALL_BLEND_BLUEPRINT.BLEND_COMPONENT_ALLOWANCE
POLICY_TEXT = (
    'One separately registered fresh28-task feasibility update after V45 complete negative real '
    'review and complete saved-output alignment diagnosis on all152 development and284 consumed '
    'confirmation cases. Initialize from the unchanged V29 checkpoint, never V45 or another probe '
    'endpoint. Change only teacher-mean Huber coefficient2 to20 in both domains, as prospectively '
    'fixed in the standalone objective. ' + stronger.POLICY +
    ' Preserve all original teacher identities and observed-only weights, priors, .5/.25/.25 task '
    'masses, probability scales, query/point losses, uniform V8 weight10 count-shrunk equal-column '
    'retention,238 trainable tensors/77,998,496 parameters,16 buffers,84 gradients and30 grouped '
    'guards. Retain fresh5e-6 AdamW, .1 projection margin, fixed-half displacement,2x norm cap and '
    'exact rejection rollback. Independently replay all52 imputation teacher component grids and '
    '26 V8 references. Keep24 BLAS/4 Torch threads, .9 allocator cap,18/6/50 GiB reserves and '
    '512 MiB/16 MiB/32 GiB probe budgets. This separate registration authorizes one feasibility '
    'update and its original independent audits, not a full trial, seed retry, changed evaluation '
    'or benchmark-based selection. All22 development and32 fresh confirmation requirements remain.'
)


BLUEPRINT_TEXT = Path(ALL_BLEND_BLUEPRINT.__file__).read_text()
BLUEPRINT_AST = ast.parse(BLUEPRINT_TEXT)
CLOSURE_SOURCES = []
for function_name in ('validate_completed', 'require_completed_v44'):
    node = next(node for node in BLUEPRINT_AST.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    source = ast.get_source_segment(BLUEPRINT_TEXT, node)
    source = source.replace('uniform_v8_retention_v44', 'all_imputation_blend_v45')
    source = source.replace('audit_single_model_uniform_v8_retention.py', 'audit_single_model_all_imputation_blend.py')
    source = source.replace('require_completed_v44', 'require_completed_v45').replace('V44', 'V45')
    if function_name == 'validate_completed':
        for old, new in [('"accepted_updates": 57', '"accepted_updates": 56'),
                         ('"training_episodes": 1792', '"training_episodes": 1764'),
                         ('"independently_regenerated_training_tasks": 1792', '"independently_regenerated_training_tasks": 1764'),
                         ('"independently_regenerated_rating_contexts": 128', '"independently_regenerated_rating_contexts": 126'),
                         ('["data_rejected_steps"] != []', '["data_rejected_steps"] != [2]'),
                         ('["optimizer_attempts"] != 64', '["optimizer_attempts"] != 63'),
                         ('["required_regressions"] != 241', '["required_regressions"] != 219')]:
            source = replace_once(source, old, new)
    CLOSURE_SOURCES.append(source)
    exec(compile(source, __file__, 'exec'), globals())
BASE_REQUIRE_CLOSURE = require_completed_v45


def require_completed_v45():
    identities = BASE_REQUIRE_CLOSURE()
    if read(ROOT / 'all_imputation_blend_v45_audit/audit.json')['training_teacher_policy'] != {
            'legacy': 'observed_blend', 'structured': 'observed_blend', 'recsys': 'specialist'}:
        raise ValueError('Preserve the completed uniform observed-blend teacher trial')
    verify_hashes({str(ALIGNMENT / 'result.json'): ALIGNMENT_SHA256})
    result = read(ALIGNMENT / 'result.json')
    if (result['complete'] is not True or result['development_cases'] != 152
            or result['consumed_confirmation_cases'] != 284 or result['raw_imputation_rmse_checks'] != 1688
            or result['case_role_split_rows'] != 3792 or any(result[key] is not False for key in (
                'cuda_initialized', 'model_inference_performed', 'real_data_used', 'training_performed',
                'selection_performed', 'gate_changed'))):
        raise ValueError('Bind complete read-only synthetic teacher alignment, not a narrowed diagnosis')
    diagnostic_inputs = {str(ALIGNMENT / 'result.json'): ALIGNMENT_SHA256,
        str(ALIGNMENT / 'config.json'): result['config_sha256'],
        str(ALIGNMENT / 'cases.csv'): result['cases_sha256'],
        str(ALIGNMENT / 'summary.csv'): result['summary_sha256']}
    verify_hashes(diagnostic_inputs)
    config = read(ALIGNMENT / 'config.json')
    verify_hashes(config['source_sha256'])
    verify_hashes(config['input_sha256'])
    identities.update(diagnostic_inputs)
    return identities


for function_name in ('reference_identities', 'replay_cohort_targets'):
    function = getattr(ALL_BLEND_BLUEPRINT, function_name)
    if function.__closure__ is not None:
        raise ValueError('Do not retain predecessor runtime closures')
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[function_name] = cloned

SOURCE = ALL_BLEND_BLUEPRINT.SOURCE
for old, new in {
        'all_imputation_blend_preflight_v1': 'stronger_mean_distillation_preflight_v1',
        '1141100901, 1142100901': '1171100901, 1172100901',
        'require_completed_v44': 'require_completed_v45'}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit stronger-mean probe adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, 'POLICY_TEXT', POLICY_TEXT)
registration_patches = [
    ('config = {"policy": POLICY,', 'config = {"policy": POLICY, "mean_distillation": stronger.SPECIFICATION,'),
    ('    sources.update(Path(module.__file__).resolve()',
     '    sources.update({REPO / "tests/test_single_model_stronger_mean_distillation.py", '
     'REPO / "tests/test_single_model_stronger_mean_preflight.py", '
     'REPO / "tests/test_single_model_teacher_alignment.py"})\n'
     '    sources.update(Path(module.__file__).resolve()'),
]
registration_source = '\n'.join(f'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, {old!r}, {new!r})'
    for old, new in registration_patches)
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    registration_source + '\nexec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
runtime_patches = [
    ('or config["teacher_policy"] != teacher_policy.POLICY',
     'or config["teacher_policy"] != teacher_policy.POLICY\n'
     '            or config["mean_distillation"] != stronger.SPECIFICATION'),
    ('    with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):',
     '    with stronger.training_objective(), teacher_policy.runtime(config["base"], output, config["teacher_identities"]):'),
]
runtime_source = '\n'.join(f'ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, {old!r}, {new!r})'
    for old, new in runtime_patches)
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)',
    runtime_source + '\nADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)')
exec(compile(SOURCE, __file__, 'exec'), globals())
unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result['tasks'])
    if blend_replay.storage_accounting(result)['reserved_checkpoint_bytes'] != BLEND_COMPONENT_ALLOWANCE:
        raise ValueError('Keep full original teacher component replay coverage and capacity')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('register', 'run', 'replay', 'audit'), required=True)
    stage = parser.parse_args().stage
    with diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if stage == 'register':
            register(OUTPUT)
            return
        with (OUTPUT / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {'run': run, 'replay': replay_targets, 'audit': audit}[stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, 'failed', error=repr(error), failed_stage=stage)
                raise


if __name__ == '__main__':
    main()
