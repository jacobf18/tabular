"""Qualify one stronger-mean student using the preserved additive probe recovery."""

from pathlib import Path

import single_model_all_imputation_blend_trial as blueprint
import single_model_stronger_mean_distillation as stronger
import recover_single_model_stronger_mean_audit as recovery
from single_model_registered_adaptation import replace_once


ARITHMETIC = blueprint.ARITHMETIC
reference_bytes = blueprint.reference_bytes
wrap_cli = blueprint.wrap_cli
RECOVERY_SHA256 = '4e3d75aa81aef0d488eee631f4fedff72d98658ae407f5e66a316ccb86303df3'
POLICY = (
    'A separately registered V46 trial initializes only from unchanged V29 step25/98 inherited updates, '
    'never V45 or the nonpromotable probe. After the complete negative V45 review, saved-output synthetic '
    'diagnosis, accepted stronger-mean update, complete additive audit recovery and five-file exact-byte '
    'storage preservation, change only teacher-mean Huber weight2 to20 uniformly for imputation and ratings. '
    'Preserve the original failed probe status and missing-Counter error; bind the separately completed '
    'recovery, original base audit/gate and26 V8 reference checks. Preserve adverse19/26 retention penalty '
    'increases (.0625555844474841 to .06445195670130643), two worse individual guard slots and structured '
    'teacher-MSE increase despite aggregate teacher alignment improvement. Feasibility is not accuracy. '
    'Keep all priors/.5/.25/.25 masses, observed-only normalization, observed-blend imputation/specialist '
    'rating targets, NLL/CDF/query/point losses, uniform V8 weight10 count-shrunk equal-family retention, '
    'fresh5e-6 AdamW, fixed-half displacement,.1 margin,84 raw gradients,30 grouped guards,2x norm bound '
    'and exact rollback. Attempt64 fresh batches without replacement. Keep boundaries0/10/25/50/64, '
    'all22 development guards, strict minimax/earlier ties/ineligible0, sole selection and284 fresh cases '
    'with32 original confirmation comparisons, followed by every frozen real suite and full review. '
    'Replay all original teacher/component/reference grids; preserve16 frozen buffers and one ordinary '
    '77,998,496-parameter inference network. Budget41,224,310,912 scratch bytes within40 GiB and original '
    '2432 MiB/16 MiB checkpoint/root allowances,18/6/50 GiB reserves,24 BLAS/4 Torch threads and.9 allocator '
    'cap. Scratch is machine-local, not external backup. No ensemble, weight sweep, source mutation, '
    'real-data selection, seed reuse, scientific retry, guard relaxation or aggregate-only success claim.')


def require_probe(probe, validate_preflight):
    probe.verify(probe.OUTPUT)
    evidence = probe.require_completed_v45()
    recovered = probe.read(recovery.OUTPUT / 'recovery_result.json')
    if (probe.sha256(recovery.OUTPUT / 'recovery_result.json') != RECOVERY_SHA256
            or any(recovered[key] is not True for key in ('complete', 'accepted', 'original_failure_preserved',
                'original_base_audit_reused', 'original_inventory_unchanged'))
            or any(recovered[key] is not False for key in ('training_repeated', 'model_or_reference_inference_repeated',
                'new_seeds_consumed', 'selection_or_benchmarks_evaluated', 'cuda_initialized', 'gate_changed'))):
        raise ValueError('Require the exact completed additive recovery, not a repaired original status')
    original = probe.read(probe.OUTPUT / 'status.json')
    current = probe.read(recovery.OUTPUT / 'status.json')
    if (original['state'] != 'failed' or original['failed_stage'] != 'audit'
            or original['error'] != 'NameError("name \'Counter\' is not defined")'
            or current['original_failed_status_preserved'] is not True
            or any(Path(f'/proc/{record["pid"]}').exists() for record in (original, current))):
        raise ValueError('Preserve both stopped statuses, including the exact original failure')
    identities = {str(recovery.OUTPUT / name): digest for name, digest in (
        ('recovery_result.json', RECOVERY_SHA256), ('recovery_config.json', recovered['recovery_config_sha256']),
        ('audit.json', recovered['base_audit_sha256']), ('gate.json', recovered['base_gate_sha256']),
        ('retention_audit.json', recovered['retention_audit_sha256']))}
    identities.update({str(probe.OUTPUT / name): recovered[key] for name, key in
        (('audit.json', 'base_audit_sha256'), ('gate.json', 'base_gate_sha256'))})
    probe.verify_hashes(identities)
    configuration = probe.read(recovery.OUTPUT / 'recovery_config.json')
    probe.verify_hashes(configuration['source_sha256'])
    probe.verify_hashes(configuration['input_sha256'])
    result = probe.read(probe.OUTPUT / 'result.json')
    audit, gate, retention = [probe.read(recovery.OUTPUT / name) for name in
        ('audit.json', 'gate.json', 'retention_audit.json')]
    validate_preflight(result, audit, gate, current, identities[str(recovery.OUTPUT / 'audit.json')])
    if (retention['complete'] is not True or retention['eligibility_decisions'] != 28
            or retention['reference_grids'] != 26 or retention['reference_role_counts'] != {'v8': 26}
            or probe.read(probe.OUTPUT / 'config.json')['mean_distillation'] != stronger.SPECIFICATION
            or audit['teacher_replay']['blend_tasks'] != 26 or audit['teacher_replay']['component_grids'] != 52
            or audit['teacher_replay']['observed_weights_refitted'] is not True):
        raise ValueError('Keep complete recovered retention, original components and the exact stronger objective')
    probe.verify_hashes(audit['artifact_sha256'])
    probe.verify_hashes(retention['artifact_sha256'])
    paths = [probe.OUTPUT / name for name in ('config.json', 'registration.json', 'result.json', 'status.json')]
    paths.append(recovery.OUTPUT / 'status.json')
    return {**evidence, **configuration['input_sha256'], **identities,
            **{str(path): probe.sha256(path) for path in paths}}


def storage_evidence(capacity, receipt_sha256):
    if receipt_sha256 is None:
        raise ValueError('Bind the completed independent storage-preservation receipt before registration')
    path = capacity.OUTPUT / 'external_recheck.json'
    receipt, configuration = capacity.read(path), capacity.read(capacity.OUTPUT / 'config.json')
    if (capacity.sha256(path) != receipt_sha256 or receipt['complete'] is not True
            or receipt['canonical_artifacts_checked'] != 5 or receipt['bytes_relocated'] != 2138934894
            or receipt['config_sha256'] != capacity.sha256(capacity.OUTPUT / 'config.json')
            or receipt['all_original_serialized_bytes_and_paths_preserved'] is not True
            or receipt['selected_checkpoints_unchanged'] is not True
            or receipt['provider_persistent_backup'] is not False
            or receipt['training_admission_bytes'] != capacity.TRAINING_ADMISSION_BYTES
            or receipt['checkpoint_available_bytes'] < receipt['training_admission_bytes']
            or configuration['policy'] != capacity.POLICY
            or capacity.read(capacity.OUTPUT / 'registration.json')['config_sha256'] != receipt['config_sha256']
            or capacity.read(capacity.OUTPUT / 'status.json')['state'] != 'complete'):
        raise ValueError('Require the exact completed five-file preservation and original reserves')
    for mapping in (receipt['artifact_sha256'], configuration['input_sha256'], configuration['source_sha256']):
        capacity.verify_hashes(mapping)
    return {**configuration['input_sha256'], str(path): receipt_sha256,
            str(capacity.OUTPUT / 'config.json'): receipt['config_sha256']}


def adapt_training(source):
    source = blueprint.adapt_training(source)
    source = replace_once(source, 'or config["arithmetic"] != all_trial.ARITHMETIC',
        'or config["arithmetic"] != all_trial.ARITHMETIC\n'
        '            or config["mean_distillation"] != stronger.SPECIFICATION')
    return replace_once(source, 'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]),',
        'with stronger.training_objective(), teacher_policy.runtime(config["base"], output, config["teacher_identities"]),')


def adapt_registration(source):
    source = blueprint.adapt_registration(source)
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "mean_distillation": stronger.SPECIFICATION,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_stronger_mean_trial.py", '
        'REPO / "tests/test_single_model_stronger_mean_full_audit.py", '
        'REPO / "tests/test_single_model_stronger_mean_pipeline.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = blueprint.adapt_recipe(source)
    return replace_once(source, '\nif __name__ == "__main__":',
        '\ndef require_evidence():\n'
        '    return {**all_trial.require_probe(preflight, validate_preflight),\n'
        '            **memory_evidence.require_evidence(), **failed_entry_evidence(), **retention_evidence()}\n'
        '\nif __name__ == "__main__":')
