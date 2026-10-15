"""Complete only the missing Counter-dependent audit tail of the original stronger-mean probe."""

import builtins
from collections import Counter
import dis
import inspect
import os
from pathlib import Path
import shutil
import sys
import types

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import preflight_single_model_stronger_mean_distillation as probe


ORIGINAL = probe.OUTPUT
OUTPUT = ORIGINAL.with_name('stronger_mean_distillation_preflight_v1_audit_recovery_v1')
CONFIG_SHA256 = '5d74709639aa1279cda3a495a179fea0910f2573edf8098e14c4ebae3a56ba4a'
BASE_AUDIT_SHA256 = 'cc8f1b2999caf50c97ff793e6808dc3b0945c3c7a38216f75021b144c8600d80'
BASE_GATE_SHA256 = 'e266309f731daa63683fdc3eb397880dfe06076d7046cdd46751e78fe362c65f'
AUDIT_FILES = {'audit.json', 'retention_audit.json', 'gate.json', 'status.json'}
POLICY = (
    'Preserve the original stronger-mean probe, completed accepted update, independent GPU teacher/'
    'reference replay, completed base CPU audit/gate and exact subsequent Counter NameError. Add only '
    'collections.Counter to an isolated audit namespace. Run the unchanged retention-audit function '
    'code with strict path/write routing to this separate directory. Its base-audit call reuses the '
    'already completed, hash-bound original audit/gate byte-for-byte after verifying every bound input; '
    'it does not rerun completed admission, optimizer, prediction or teacher-component checks. '
    'Only the missing retention report/status tail is completed. Preserve all original artifact '
    'bytes, paths, failed status, source hashes, arithmetic and acceptance predicates. No training, '
    'optimizer update, model/reference inference, seed, held-out qualification or benchmark is '
    'repeated. A completed feasibility audit does not establish accuracy or authorize a full trial.'
)


class AuditView(os.PathLike):
    def __init__(self, original, destination):
        self.original, self.destination = Path(original), Path(destination)

    def __truediv__(self, name):
        name = Path(name)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Keep audit paths inside their declared owners')
        return (self.destination if str(name) in AUDIT_FILES else self.original) / name

    def __fspath__(self):
        return str(self.original)

    def __str__(self):
        return str(self.original)


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Recovery requires hidden and uninitialized CUDA')


def missing_globals(function, namespace):
    pending, loaded = [function.__code__], set()
    while pending:
        code = pending.pop()
        loaded.update(instruction.argval for instruction in dis.get_instructions(code) if instruction.opname == 'LOAD_GLOBAL')
        pending.extend(value for value in code.co_consts if isinstance(value, types.CodeType))
    return {name for name in loaded if name not in namespace and not hasattr(builtins, name)}


def validate_failure(failed, result, base_audit, replay, gate, log):
    if (failed.get('state') != 'failed' or failed.get('failed_stage') != 'audit'
            or failed.get('error') != 'NameError("name \'Counter\' is not defined")'
            or "NameError: name 'Counter' is not defined" not in log):
        raise ValueError('Recover only the exact missing-Counter audit-tail failure')
    if (result.get('data_admission_only') is not False or result.get('preflight_only') is not True
            or result.get('trainable_parameters') != 77998496
            or result['record']['accepted'] is not True or result['record']['accepted_updates'] != 1
            or result['record']['reason'] != 'accepted' or base_audit.get('complete') is not True
            or base_audit.get('training_tasks') != 28 or base_audit.get('prediction_grids') != 56
            or base_audit.get('cuda_initialized') is not False or base_audit.get('autograd_replayed') is not False
            or base_audit['raw_endpoint']['gradient_vectors'] != 84
            or base_audit['raw_endpoint']['accepted_updates'] != 1
            or base_audit['frozen']['frozen_tensors'] != 16
            or base_audit['teacher_replay']['complete'] is not True
            or base_audit['teacher_replay']['full_grids'] != 28
            or base_audit['teacher_replay']['blend_tasks'] != 26
            or base_audit['teacher_replay']['component_grids'] != 52
            or base_audit['teacher_replay']['observed_weights_refitted'] is not True
            or replay.get('complete') is not True or len(replay['records']) != 26
            or gate.get('passed') is not True or gate.get('preflight_only') is not True
            or gate.get('reason') != 'accepted' or gate.get('real_benchmark_success') is not False
            or gate.get('audit_sha256') != BASE_AUDIT_SHA256):
        raise ValueError('Preserve the complete original accepted update and original base audit')


def original_inventory():
    result, directories = {}, set()
    for directory, folders, files in os.walk(ORIGINAL, followlinks=True):
        directory = Path(directory)
        resolved = directory.resolve()
        if resolved in directories or not any(resolved.is_relative_to(owner) for owner in (ORIGINAL, probe.SCRATCH)):
            raise ValueError('Require an acyclic original probe and declared scratch tree')
        directories.add(resolved)
        for name in files:
            path = directory / name
            result[str(path)] = sha256(path)
    return result


def failure_proof():
    require_cpu()
    verify_hashes({str(ORIGINAL / 'config.json'): CONFIG_SHA256, str(ORIGINAL / 'audit.json'): BASE_AUDIT_SHA256,
                   str(ORIGINAL / 'gate.json'): BASE_GATE_SHA256})
    probe.verify(ORIGINAL)
    failed, result, base_audit, replay, gate = [probe.read(ORIGINAL / name) for name in
        ('status.json', 'result.json', 'audit.json', 'retention_replay/manifest.json', 'gate.json')]
    validate_failure(failed, result, base_audit, replay, gate, (ORIGINAL / 'original_probe_chain.log').read_text())
    if (ORIGINAL / 'retention_audit.json').exists():
        raise ValueError('Do not overwrite or reinterpret an already completed audit tail')
    for process_id in (failed['pid'], result['training_pid'], replay['pid']):
        if type(process_id) is not int or process_id <= 0 or Path(f'/proc/{process_id}').exists():
            raise ValueError('Original update, replay and failed auditor must all be stopped')
    if missing_globals(probe.audit, vars(probe)) != {'Counter'}:
        raise ValueError('Admit only the diagnosed missing standard-library dependency')
    verify_hashes(base_audit['artifact_sha256'])
    verify_hashes(base_audit['teacher_replay']['artifact_sha256'])
    return original_inventory()


def namespace(destination):
    current = dict(vars(probe))
    current['Counter'] = Counter
    for name, value in vars(probe).items():
        if inspect.isfunction(value) and value.__globals__ is vars(probe) and value.__closure__ is None:
            cloned = types.FunctionType(value.__code__, current, value.__name__, value.__defaults__)
            cloned.__kwdefaults__ = value.__kwdefaults__
            current[name] = cloned

    def reuse_base_audit(output):
        if not isinstance(output, AuditView) or output.original != ORIGINAL or output.destination != destination:
            raise ValueError('Reuse only the exact completed original base audit')
        verify_hashes({str(ORIGINAL / 'audit.json'): BASE_AUDIT_SHA256,
                       str(destination / 'audit.json'): BASE_AUDIT_SHA256,
                       str(ORIGINAL / 'gate.json'): BASE_GATE_SHA256,
                       str(destination / 'gate.json'): BASE_GATE_SHA256})
        original = probe.read(ORIGINAL / 'audit.json')
        verify_hashes(original['artifact_sha256'])
        verify_hashes(original['teacher_replay']['artifact_sha256'])

    def audit_write(path, payload):
        path = Path(path)
        if path.parent != destination or path.name not in {'retention_audit.json', 'status.json'}:
            raise ValueError('The isolated auditor may write only the new retention report and status')
        if path.name != 'status.json' and path.exists():
            raise FileExistsError('Never overwrite a recovered audit report')
        return write_json(path, payload)

    current['audit_original'] = reuse_base_audit
    current['write_json'] = audit_write
    if missing_globals(current['audit'], current):
        raise ValueError('Audit dependencies must be complete before recovery')
    return current


def recover():
    require_cpu()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError('Never repeat a started recovery')
    if shutil.disk_usage(OUTPUT.parent).free < 18 * 1024**3 + 16 * 1024**2:
        raise OSError('Keep the18 GiB reserve and16 MiB audit-output allowance')
    inputs = failure_proof()
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__) for module in tuple(sys.modules.values())
        if getattr(module, '__file__', None) and str(module.__file__).endswith('.py')
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file()}
    test_path = REPO / 'tests/test_single_model_stronger_mean_audit_recovery.py'
    sources[str(test_path)] = sha256(test_path)
    OUTPUT.mkdir()
    write_json(OUTPUT / 'recovery_config.json', {'policy': POLICY, 'input_sha256': inputs,
        'source_sha256': sources, 'original_config_sha256': CONFIG_SHA256, 'base_audit_sha256': BASE_AUDIT_SHA256,
        'base_gate_sha256': BASE_GATE_SHA256,
        'scientific_configuration_changed': False, 'reused_completed_base_audit': True})
    write_json(OUTPUT / 'recovery_registration.json', {'recovery_config_sha256': sha256(OUTPUT / 'recovery_config.json')})
    shutil.copyfile(ORIGINAL / 'audit.json', OUTPUT / 'audit.json')
    shutil.copyfile(ORIGINAL / 'gate.json', OUTPUT / 'gate.json')
    write_json(OUTPUT / 'status.json', {'state': 'awaiting_independent_audit', 'pid': os.getpid(),
        'original_failed_status_preserved': True, 'recovery_role': 'missing retention audit tail only'})
    try:
        current = namespace(OUTPUT)
        current['audit'](AuditView(ORIGINAL, OUTPUT))
        retention = probe.read(OUTPUT / 'retention_audit.json')
        if (retention['complete'] is not True or retention['eligibility_decisions'] != 28
                or retention['reference_grids'] != 26 or retention['reference_role_counts'] != {'v8': 26}
                or retention['original_audit_sha256'] != BASE_AUDIT_SHA256
                or retention['cuda_initialized'] is not False or retention['full_autograd_replayed'] is not False):
            raise ValueError('Keep complete original retention coverage and audit semantics')
        verify_hashes(retention['artifact_sha256'])
        verify_hashes(inputs)
        verify_hashes(sources)
        if original_inventory() != inputs:
            raise ValueError('Every original byte and path must remain unchanged')
        require_cpu()
        verify_hashes({str(OUTPUT / 'gate.json'): BASE_GATE_SHA256})
        write_json(OUTPUT / 'recovery_result.json', {'complete': True, 'accepted': True,
            'original_failure_preserved': True, 'original_base_audit_reused': True,
            'original_inventory_unchanged': True, 'training_repeated': False,
            'model_or_reference_inference_repeated': False, 'new_seeds_consumed': False,
            'selection_or_benchmarks_evaluated': False, 'cuda_initialized': False, 'gate_changed': False,
            'base_audit_sha256': BASE_AUDIT_SHA256, 'base_gate_sha256': BASE_GATE_SHA256,
            'retention_audit_sha256': sha256(OUTPUT / 'retention_audit.json'),
            'recovery_config_sha256': sha256(OUTPUT / 'recovery_config.json')})
        write_json(OUTPUT / 'status.json', {'state': 'complete', 'pid': os.getpid(), 'accepted': True,
            'original_failed_status_preserved': True, 'retention_audited': True})
    except BaseException as error:
        write_json(OUTPUT / 'status.json', {'state': 'failed', 'pid': os.getpid(), 'error': repr(error),
            'original_failed_status_preserved': True})
        raise


if __name__ == '__main__':
    with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        recover()
