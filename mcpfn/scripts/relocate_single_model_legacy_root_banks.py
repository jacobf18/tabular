"""Preserve four closed synthetic banks to restore root headroom during V46."""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import torch

import relocate_single_model_idle_diagnostics_to_root as readers
import relocate_single_model_idle_diagnostics_to_scratch as copying
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = copying.ROOT
SOURCE = REPO / 'runs/single_model_goal_20261005'
OUTPUT = ROOT / 'legacy_root_bank_preservation_v1'
DESTINATION = copying.SCRATCH / 'single_model_goal_20261005_owned/legacy_root_banks_v1'
OWNERS = {'mixed_distill_v2_confirmation': ('complete', 46, 'e2ac69a4063a8ce6b3bb9fa4e55e3d64577920509ba7a907ac0f7263638ccfab'),
    'mixed_distill_v2_tail_average_confirmation': ('complete', 46, '64a7c7f4ec12855111edd4be3581c2ea66a32ac55cb0650d3692df0bec45b31f'),
    'initializer_screen_v1': ('complete', 28, '13b84fb32d70a90f7f30a2695c76aef7d6bf19f5ed0065a46a66a99ec4158fbe'),
    'mixed_distill_v2': ('training_complete', 28, 'cc2a99023f188d75a6613f54dbfc7b01e7768c6daf43618149a3184ab6826af3')}
BYTES = 1019705284
POLICY = (
    'Storage only: preserve exactly four closed synthetic-bank files physically on root from the '
    'original V2 confirmation, V2 tail-average confirmation, initializer screen and V2 development. '
    'Their original published reference hashes, complete case inventories, terminal metadata and '
    'bank tensor/mask/seed identities must remain unchanged. Terminal metadata is storage provenance, '
    'not a new scientific audit of those old experiments. Keep all original model checkpoints, '
    'teachers, selected/published states, mounted archives, sources, configurations, failures and '
    'live V46 outputs untouched. Require stopped exact owners, their exclusive original locks and '
    'no open descriptors/mappings of any bank inode or alias. Unrelated live training is permitted. '
    'Copy/fsync/hash and CPU mmap-check every full bank before atomic canonical symlinks. Independently '
    'recheck bytes, tensor identities, original proofs and all442 V46 registered sources. Preserve '
    '18/6/50 GiB reserves plus16 MiB root headroom. UUID-bound scratch is machine-local, not external '
    'or provider-persistent backup. No evidence deletion, model move, regeneration, inference, '
    'training interruption, new seed, source change, selection, qualification or benchmark claim.')


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / 'status.json', {'state': state, 'pid': os.getpid(), 'updated_unix': time.time(), **fields})


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Require hidden, uninitialized CUDA for storage-only preservation')


def require_stopped():
    for name, (state, count, expected) in OWNERS.items():
        current = read(SOURCE / name / 'status.json')
        if (current['state'] != state or type(current['pid']) is not int or current['pid'] <= 0
                or Path(f'/proc/{current["pid"]}').exists()):
            raise ValueError('Only the exact stopped original owners may be preserved')


@contextmanager
def owner_locks():
    with ExitStack() as stack:
        for name in OWNERS:
            stream = stack.enter_context((SOURCE / name / 'run.lock').open('r'))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_stopped()
        yield


def validate_bank(bank, reference, expected_count):
    if not isinstance(bank, list) or len(bank) != expected_count or len(reference['cases']) != expected_count:
        raise ValueError('Keep the complete original synthetic-bank inventory')
    for index, (cached, case) in enumerate(zip(bank, reference['cases'])):
        shape = (1, case['rows'], case['columns'])
        if (case['index'] != index or case['gradient_data'] != 'synthetic'
                or cached['metadata']['gradient_data'] != 'synthetic'
                or cached['metadata']['seed'] != case['seed'] or cached['metadata']['domain'] != case['domain']
                or case['domain'] not in ('imputation', 'recsys')
                or any(not isinstance(cached[key], torch.Tensor) or tuple(cached[key].shape) != shape
                       or cached[key].device.type != 'cpu' for key in ('source', 'target', 'mask'))
                or cached['source'].dtype != torch.float32 or cached['target'].dtype != torch.float32
                or cached['mask'].dtype != torch.bool or not cached['mask'].any()
                or not torch.isfinite(cached['target']).all() or torch.isinf(cached['source']).any()):
            raise ValueError('Preserve every original typed full matrix, mask and synthetic case identity')


def check_file(path, name):
    reference = read(SOURCE / name / 'reference.json')
    state, count, expected = OWNERS[name]
    if reference['bank_sha256'] != expected or sha256(path) != expected:
        raise ValueError('Bind the original published bank hash, never regenerated serialization')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    validate_bank(payload, reference, count)
    del payload


def require_reserves(extra_scratch=0):
    if (shutil.disk_usage('/').free < 6 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(ROOT).free < 18 * 1024**3
            or shutil.disk_usage(copying.SCRATCH).free < 50 * 1024**3 + extra_scratch):
        raise OSError('Preserve every original reserve and16 MiB root qualification headroom')


def inventory():
    require_cpu()
    copying.require_mount()
    require_stopped()
    records, inputs = [], {}
    for name, (state, count, expected) in OWNERS.items():
        owner = SOURCE / name
        source = owner / 'synthetic_bank.pt'
        if (source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1
                or source.stat().st_dev != Path('/').stat().st_dev):
            raise ValueError('Move only the four original singly linked regular bank files physically on root')
        check_file(source, name)
        inputs.update({str(path): sha256(path) for path in owner.glob('*.json') if path.is_file()})
        records.append({'name': name, 'source': str(source), 'destination': str(DESTINATION / name / source.name),
            'sha256': expected, 'bytes': source.stat().st_size})
    if sum(row['bytes'] for row in records) != BYTES:
        raise ValueError('Require exactly the full four-bank byte inventory')
    for relative in ('mixed_distill_v1/best_validation_checkpoint.pth', 'mixed_distill_v2/best_validation_checkpoint.pth',
            'mixed_distill_v2_tail_average/averaged_checkpoint.pth'):
        path = SOURCE / relative
        inputs[str(path)] = sha256(path)
    training = ROOT / 'stronger_mean_distillation_v46'
    configuration = read(training / 'config.json')
    if read(training / 'registration.json')['config_sha256'] != sha256(training / 'config.json'):
        raise ValueError('Preserve the registered live V46 experiment')
    for name in ('config.json', 'registration.json'):
        inputs[str(training / name)] = sha256(training / name)
    verify_hashes(configuration['source_sha256'])
    readers.require_no_artifact_readers([row['source'] for row in records])
    return records, inputs, configuration['source_sha256']


def copy_and_verify(row):
    source, destination = Path(row['source']), Path(row['destination'])
    destination.parent.mkdir(parents=True, exist_ok=False)
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row['bytes']:
        raise ValueError('Never publish an incomplete serialized copy')
    check_file(destination, row['name'])


def recheck():
    require_cpu()
    copying.require_mount()
    require_stopped()
    config = read(OUTPUT / 'config.json')
    if (config['policy'] != POLICY or config['scratch_uuid'] != copying.UUID
            or read(OUTPUT / 'registration.json')['config_sha256'] != sha256(OUTPUT / 'config.json')):
        raise ValueError('Keep the exact registered storage-only operation')
    for field in ('input_sha256', 'source_sha256', 'protected_training_source_sha256'):
        verify_hashes(config[field])
    hashes = {}
    for row in config['records']:
        source, destination = Path(row['source']), Path(row['destination'])
        if (not source.is_symlink() or source.readlink() != destination or source.resolve() != destination
                or destination.stat().st_size != row['bytes']):
            raise ValueError('Preserve every exact canonical path and complete byte stream')
        check_file(source, row['name'])
        hashes[str(source)] = hashes[str(destination)] = row['sha256']
    verify_hashes(hashes)
    require_reserves()
    return {'complete': True, 'canonical_banks_checked': 4, 'cases_checked': sum(value[1] for value in OWNERS.values()),
        'bytes_relocated': sum(row['bytes'] for row in config['records']), 'artifact_sha256': hashes,
        'all_original_serialized_bytes_and_paths_preserved': True, 'model_checkpoints_unchanged': True,
        'protected_training_sources_checked': len(config['protected_training_source_sha256']),
        'provider_persistent_backup': False, 'scientific_training_or_evaluation_performed': False,
        'cuda_initialized': False, 'config_sha256': sha256(OUTPUT / 'config.json'),
        'root_available_bytes': shutil.disk_usage('/').free}


def preserve():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError('Never overwrite or restart a begun preservation')
    records, inputs, protected_sources = inventory()
    require_reserves(BYTES)
    sources = [Path(__file__).resolve(), Path(readers.__file__).resolve(), Path(copying.__file__).resolve(),
        REPO / 'tests/test_single_model_legacy_root_banks.py']
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    write_json(OUTPUT / 'config.json', {'policy': POLICY, 'records': records, 'input_sha256': inputs,
        'source_sha256': {str(path): sha256(path) for path in sources},
        'protected_training_source_sha256': protected_sources, 'scratch_uuid': copying.UUID})
    write_json(OUTPUT / 'registration.json', {'config_sha256': sha256(OUTPUT / 'config.json'), 'registered_unix': time.time()})
    write_json(DESTINATION / 'OWNERSHIP.json', {'registered_at': str(OUTPUT), 'config_sha256': sha256(OUTPUT / 'config.json')})
    status('copying_closed_synthetic_banks')
    try:
        for row in records:
            copy_and_verify(row)
        require_stopped()
        readers.require_no_artifact_readers([row['source'] for row in records])
        for row in records:
            copying.publish(row)
        write_json(OUTPUT / 'independent_recheck.json', recheck())
        status('complete', bytes_relocated=BYTES)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preserve', 'recheck'), required=True)
    stage = parser.parse_args().stage
    with owner_locks():
        if stage == 'preserve':
            preserve()
        else:
            receipt = OUTPUT / 'external_recheck.json'
            if receipt.exists():
                raise FileExistsError('Never replace an independent preservation receipt')
            write_json(receipt, recheck())
