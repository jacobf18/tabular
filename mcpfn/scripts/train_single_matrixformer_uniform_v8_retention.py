"""Train one V44 student with uniform V8 retention and unchanged mixed priors."""

import ast
import inspect

import train_single_matrixformer_family_balanced_retention as previous_training
import single_model_fixed_half_trial as fixed_trial
import single_model_uniform_v8_trial as all_trial
from single_model_registered_adaptation import replace_assignment


CAPACITY_RECEIPT_SHA256 = '97b3d82c1ee9419d201b3f0b19db2c414599c11291aad958f75acdf42ae2fa8d'


def retention_evidence():
    if CAPACITY_RECEIPT_SHA256 is None:
        raise ValueError('Do not register V44 before binding its completed storage-preservation receipt')
    path = capacity.OUTPUT / 'external_recheck.json'
    configuration = capacity.OUTPUT / 'config.json'
    receipt = capacity.read(path)
    if (capacity.sha256(path) != CAPACITY_RECEIPT_SHA256 or receipt['complete'] is not True
            or receipt['canonical_artifacts_checked'] != 3
            or receipt['config_sha256'] != capacity.sha256(configuration)
            or receipt['all_original_serialized_bytes_and_paths_preserved'] is not True
            or receipt['selected_checkpoints_unchanged'] is not True
            or receipt['provider_persistent_backup'] is not False
            or receipt['bytes_relocated'] != 1652558524
            or receipt['training_admission_bytes'] != capacity.TRAINING_ADMISSION_BYTES
            or receipt['checkpoint_available_bytes'] < receipt['training_admission_bytes']):
        raise ValueError('Bind the exact complete independent three-file preservation receipt')
    configuration_record = capacity.read(configuration)
    if (configuration_record['policy'] != capacity.POLICY
            or capacity.read(capacity.OUTPUT / 'registration.json')['config_sha256'] != capacity.sha256(configuration)
            or capacity.read(capacity.OUTPUT / 'status.json')['state'] != 'complete'):
        raise ValueError('Bind the registered completed preservation, not a replacement receipt')
    capacity.verify_hashes(receipt['artifact_sha256'])
    capacity.verify_hashes(configuration_record['input_sha256'])
    capacity.verify_hashes(configuration_record['source_sha256'])
    capacity.probe.verify(capacity.probe.OUTPUT)
    capacity.probe.retention_records(capacity.probe.OUTPUT)
    evidence = capacity.probe.require_completed_v43()
    return {**evidence, **configuration_record['input_sha256'], str(path): capacity.sha256(path),
            str(configuration): capacity.sha256(configuration)}


RETENTION_EVIDENCE_SOURCE = inspect.getsource(retention_evidence)
ADAPTATION_SOURCE = previous_training.ADAPTATION_SOURCE
for old, new in {
    'family_balanced_retention': 'uniform_v8_retention',
    'relocate_single_model_family_balanced_capacity': 'relocate_single_model_uniform_v8_capacity',
    '1091100901, 1092100901': '1121100901, 1122100901',
    '1101100901, 1102100901': '1131100901, 1132100901',
}.items():
    if old not in ADAPTATION_SOURCE:
        raise ValueError(f'Missing explicit uniform-V8 full-trial adaptation: {old}')
    ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace(old, new)
ADAPTATION_SOURCE = ADAPTATION_SOURCE.replace('uniform_v8_retention_v43', 'uniform_v8_retention_v44').replace('V43', 'V44')
node = next(node for node in ast.parse(ADAPTATION_SOURCE).body
    if isinstance(node, ast.FunctionDef) and node.name == 'retention_evidence')
lines = ADAPTATION_SOURCE.splitlines(keepends=True)
ADAPTATION_SOURCE = ''.join(lines[:node.lineno - 1]) + RETENTION_EVIDENCE_SOURCE + ''.join(lines[node.end_lineno:])
ADAPTATION_SOURCE = replace_assignment(ADAPTATION_SOURCE, 'ANCHOR_POLICY',
    all_trial.POLICY)
BATCH_WRAPPER = previous_training.BATCH_WRAPPER
exec(compile(ADAPTATION_SOURCE, __file__, 'exec'), globals())
