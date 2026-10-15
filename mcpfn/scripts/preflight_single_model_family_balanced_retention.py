"""Probe equal-family retention only after V42's audited negative closure."""

from collections import Counter
import fcntl
from pathlib import Path
import types

import preflight_single_model_cohort_anchor_retention as previous_probe
from preflight_single_model_cohort_anchor_retention import diagnosis, blend_replay, replace_function
import single_model_family_balanced_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once


ARITHMETIC = {'blas_threads': 24, 'torch_threads': 4}
ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005')
CLOSED_REPORTS = {
    'cohort_anchor_retention_v42/config.json':
        'fec67da9d9de7a0127d9daa77b9ef68745d7a2feae4748407f3832d21b939b22',
    'cohort_anchor_retention_v42_audit/audit.json':
        '30aaeb3f4397b59bab54157232555032784c26c2603702ef4745de26e2c74eeb',
    'cohort_anchor_retention_v42_endpoint_replay/audit.json':
        '54f6b00b203f2245f2aa4e4a77177f97fe8dc1f71c55af528bb03f621b206b6a',
    'cohort_anchor_retention_v42_endpoint_replay/gate.json':
        '8880e1475b058bcbddf01455805022af1e0ec818b84af1fd062fcb17436cbf0e',
    'cohort_anchor_retention_v42_pipeline/result.json':
        '82a3da1d556d1ebd380b72642dd8254800c80ea4466da72cd99be7bc7920d1ca',
    'cohort_anchor_retention_v42_endpoint_diagnosis_v1/result.json':
        '826087d52f583b83e06d728e1c51642d41c2935fbb5dc7ea24306990e838f87f',
}
POLICY_TEXT = ('One fresh28-task full-network feasibility probe after the complete independently '
    'audited negative V42 trial and CPU-only saved-endpoint diagnosis. ' + cohort_retention.POLICY
    + ' Keep V29 step25/98 inherited updates as initializer, not the failed V42 endpoint or any probe. '
    'Keep238 tensors/77,998,496 trainable parameters,16 frozen buffers,84 raw gradients,30 protections, '
    'fresh5e-6 AdamW,.1 margin,fixed-half displacement,2x norm bound and exact rollback. Raw gradient '
    'capture passes gradient_scale1, which is not a supervised task weight: derive the fixed retention '
    'multiplier from declared cohort weights and independently bind them to every registered task. '
    'Archive all weighting decisions alongside complete original sources/masks and frozen references. '
    'Replay every original teacher, both legacy blend components and both retention references. '
    'Pin24 BLAS/4 Torch threads in every stage. Preserve.9 allocator cap,18/6/50 GiB reserves and '
    '512 MiB/16 MiB/32 GiB write allowances. No changed sampling, supervision, reference selection, '
    'scientific retry, reused seed, relaxed guard or inference ensemble. Acceptance proves feasibility '
    'only; this registration does not authorize full training, confirmation or a real campaign.')


def validate_completed(training_audit, replay, gate, pipeline, diagnostic):
    expected = {'complete': True, 'complete_training_audit': True, 'through_step': 64,
        'attempted_batches': 64, 'accepted_updates': 58, 'training_episodes': 1792,
        'prediction_grids': 1368, 'full_network_training': True, 'fixed_half_training': True,
        'displacement_fraction': .5, 'independently_regenerated_training_tasks': 1792,
        'independently_regenerated_rating_contexts': 128, 'independently_regenerated_structured_draws': 1536,
        'confirmation_evaluated': False, 'full_optimizer_trajectory_replayed': False, 'autograd_replayed': False}
    if (any(training_audit.get(key) != value for key, value in expected.items())
            or training_audit['best']['step'] != 0
            or training_audit['admission_attempts']['data_rejected_steps'] != []
            or training_audit['admission_attempts']['optimizer_attempts'] != 64
            or training_audit['raw_endpoint']['adam']['gradient_vectors'] != 84
            or training_audit['raw_endpoint']['optimizer_attempt_step'] != 64
            or not replay['complete'] or replay['endpoint_grids'] != 56 or replay['selected_grids'] != 0
            or gate['endpoint_reproduced'] is not True or gate['selected_validation_passed'] is not False
            or gate['selected_step'] != 0 or pipeline['complete'] is not True
            or pipeline['gate']['passed'] is not False or pipeline['gate']['confirmation_evaluated'] is not False
            or pipeline['campaign'] is not None or pipeline['review'] is not None
            or not diagnostic['complete'] or diagnostic['endpoint_step'] != 64
            or len(diagnostic['tasks']) != 28
            or sum(row['reference_role'] is not None for row in diagnostic['tasks']) != 26
            or any(diagnostic[key] is not False for key in ('cuda_initialized', 'training_performed',
                'model_inference_performed', 'checkpoint_selected'))):
        raise ValueError('Require V42 complete negative closure and full CPU endpoint diagnosis')


def require_completed_v42():
    identities = {str(ROOT / name): digest for name, digest in CLOSED_REPORTS.items()}
    verify_hashes(identities)
    config, training_audit, replay, gate, pipeline, diagnostic = [read(ROOT / name) for name in CLOSED_REPORTS]
    validate_completed(training_audit, replay, gate, pipeline, diagnostic)
    for name in ('cohort_anchor_retention_v42', 'cohort_anchor_retention_v42_endpoint_replay',
            'cohort_anchor_retention_v42_confirmation', 'cohort_anchor_retention_v42_pipeline'):
        path = ROOT / name / 'status.json'
        state = read(path)
        if (state['state'] not in ('complete', 'training_complete')
                or diagnosis.original.process_identity(state['pid']) is not None):
            raise ValueError('Every predecessor producer must be stopped')
        identities[str(path)] = sha256(path)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = path.read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if any(argument.rsplit(b'/', 1)[-1] == b'audit_single_model_cohort_anchor_retention.py'
                for argument in arguments):
            raise ValueError('Require the original independent auditor to have stopped')
    verify_hashes(config['source_sha256'])
    for report in (training_audit, replay, diagnostic):
        verify_hashes(report['artifact_sha256'])
    verify_hashes(diagnostic['source_sha256'])
    return identities


for name in ('reference_identities', 'replay_cohort_targets'):
    function = getattr(previous_probe, name)
    if function.__closure__ is not None:
        raise ValueError('Do not retain a predecessor reference or runtime closure')
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

SOURCE = previous_probe.SOURCE
for old, new in {
    'single_model_cohort_anchor_retention as retention': 'single_model_family_balanced_retention as retention',
    'cohort_anchor_retention_preflight_v1': 'family_balanced_retention_preflight_v1',
    '1051100901, 1052100901': '1081100901, 1082100901',
    'require_completed_v41': 'require_completed_v42',
    'test_single_model_cohort_anchor_retention.py': 'test_single_model_family_balanced_retention.py',
    'test_single_model_cohort_anchor_retention_preflight.py': 'test_single_model_family_balanced_retention_preflight.py',
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing family-balanced probe adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE,
    'or record["reference_role"] != retention.reference_role(episode.metadata)',
    'or record["reference_role"] != retention.reference_role(episode.metadata)\n'
    '                or record["weighting"] != retention.retention_weighting(episode.metadata)\n'
    '                or record["gradient_scale"] != 1.\n'
    '                or task["weight"] != record["weighting"]["task_weight"]')
SOURCE = replace_once(SOURCE,
    'payload["reference_role"] != record["reference_role"]',
    'payload["reference_role"] != record["reference_role"]\n'
    '                    or payload["weighting"] != record["weighting"]')
SOURCE = replace_once(SOURCE,
    'row[role] = retention.independent_penalty(episode.target, episode.mask, prediction, payload["prediction"])',
    'row[role] = record["weighting"]["multiplier"] * retention.independent_penalty(\n'
    '                episode.target, episode.mask, prediction, payload["prediction"])\n'
    '            row[role + "_mixture_weighted"] = task["weight"] * row[role]')
SOURCE = replace_once(SOURCE,
    '"reference_role": record["reference_role"]}',
    '"reference_role": record["reference_role"], "weighting": record["weighting"]}')
SOURCE = replace_once(SOURCE, '    seen, records = set(), []',
    '    retention.validate_task_mixture(registered)\n    seen, records = set(), []')
SOURCE = replace_assignment(SOURCE, 'POLICY_TEXT', POLICY_TEXT)
exec(compile(SOURCE, __file__, 'exec'), globals())

unvalidated_batch = batch


def batch():
    result = unvalidated_batch()
    retention.validate_task_mixture(result['tasks'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('register', 'run', 'replay', 'audit'), required=True)
    stage = parser.parse_args().stage
    with diagnosis.recovery.previous.original_arithmetic():
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
