"""Restore V53 qualification headroom without changing scientific artifacts."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

import relocate_single_model_mounted_archives as mounts
import relocate_single_model_idle_diagnostics_to_scratch as scratch
from relocate_single_model_gradient_archive import relocate
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


ROOT = Path('/mnt/checkpoints_200/single_model_goal_20261005')
OUTPUT = ROOT / 'v53_root_archive_preservation_v1'
ARCHIVE = ROOT / 'finite_endpoint_capacity_archive_v1/group_00'
SOURCE = REPO / 'runs/single_model_goal_20261005/mounted_archive_relocation_v2/finite_endpoint_capacity_archive_v1/gradients.squashfs'
DESTINATION = scratch.SCRATCH / 'single_model_goal_20261005_owned/v53_root_archive_v1/gradients.squashfs'
OWNER = ROOT / 'finite_guarded_v13'
IMAGE_SHA = '2555f4fc8e901462c99a1813ee070a3752d6ef15b0a86cf325dd73573dabbf19'
IMAGE_BYTES = 10127245312
RAW_SHA = '6ee030db1f1dbfd6788688c5ecac9a46a964bed8eb1300a1c7747e4c496572c6'
UNIT_SHA = 'fd4b8beec89be258ddea430c463e5b83475c2d68118d64b47f0c1b59b8317b06'
CONFIGS = {
    'balanced_structured_components_v53': 'f1e90786a292bb5054ab0eb8ab7916aad5539ddb7da2e4bd3ca453ddbf64b5da',
    'balanced_structured_components_v53_pipeline': 'c126b3a5739b4355aa7b816dcd9d9214f96ae964c959d1128dde3a2402d92a86',
}
POLICY = (
    'Storage recovery after original V53 pipeline failed before confirm on its unchanged root '
    'reserve plus16 MiB stage allowance. Preserve exactly the already archived V13 raw endpoint '
    'image, every original byte, both canonical image paths, raw tensor path and enabled read-only '
    'mount unit. Copy/fsync/hash before atomic physical-source symlink replacement; restart only '
    'the unchanged existing mount to release its old loop inode. Require stopped producers, '
    'exclusive original owner lock and no scientific worker or artifact reader. Independently '
    'recheck the full serialized image/raw artifact, CPU torch mmap and all555 V53 sources. '
    'Keep every selected checkpoint, teacher, baseline prediction and V53 result untouched. '
    'Record initial root shortfall rather than relaxing qualification: restore6 GiB plus16 MiB '
    'headroom, retain18 GiB checkpoint and50 GiB scratch reserves. No filesystem resizing, '
    'formatting, reserved-block changes, recompression, evidence deletion, inference, training, '
    'seed generation, checkpoint selection or scientific retry. Scratch is machine-local, '
    'not an external or provider-persistent backup. Original failed pipeline remains failed.'
)


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / 'status.json', {'state': state, 'pid': os.getpid(),
        'updated_unix': time.time(), **fields})


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Require hidden, uninitialized CUDA for storage recovery')


def validate_original(audit):
    if (audit.get('complete') is not True or audit.get('all_original_hashes_verified') is not True
            or audit.get('torch_mmap_readable') is not True or audit.get('persistent_mount_enabled') is not True
            or audit.get('image') != str(ARCHIVE / 'gradients.squashfs')
            or audit.get('image_bytes') != IMAGE_BYTES or audit.get('image_sha256') != IMAGE_SHA
            or audit.get('unit_sha256') != UNIT_SHA
            or audit.get('records') != [{'source': str(OWNER / 'raw_final_update.pt'), 'sha256': RAW_SHA}]):
        raise ValueError('Preserve exactly the original V13 endpoint archive, never a model or baseline')
    expected = subprocess.check_output(['systemd-escape', '--path', '--suffix=mount', str(ARCHIVE / 'files')], text=True).strip()
    if audit['unit'] != str(Path('/etc/systemd/system') / expected):
        raise ValueError('Preserve the exact existing persistent mount unit')


def require_terminal():
    mounts.terminal(OWNER / 'status.json')
    mounts.terminal(ROOT / 'balanced_structured_components_v53/status.json')
    pipeline = read(ROOT / 'balanced_structured_components_v53_pipeline/status.json')
    if (pipeline.get('state') != 'failed' or type(pipeline.get('pid')) is not int
            or pipeline['pid'] <= 0 or Path(f'/proc/{pipeline["pid"]}').exists()
            or 'before confirm' not in pipeline.get('error', '')
            or '6442450944 bytes on root' not in pipeline.get('error', '')):
        raise ValueError('Require the original stopped V53 pre-confirm capacity failure')


@contextmanager
def owner_lock():
    with (OWNER / 'run.lock').open('r') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_terminal()
        yield


def capacity(*, copying=False, restored=False):
    if (shutil.disk_usage(ROOT).free < 18 * 1024**3
            or shutil.disk_usage(scratch.SCRATCH).free < 50 * 1024**3 + (IMAGE_BYTES if copying else 0)):
        raise OSError('Keep checkpoint and scratch reserves throughout recovery')
    if restored and shutil.disk_usage('/').free < 6 * 1024**3 + 16 * 1024**2:
        raise OSError('Restore the unchanged root reserve and qualification stage allowance')


def inventory():
    require_cpu()
    scratch.require_mount()
    require_terminal()
    audit = read(ARCHIVE / 'audit.json')
    validate_original(audit)
    image = ARCHIVE / 'gradients.squashfs'
    if (not image.is_symlink() or image.resolve() != SOURCE or SOURCE.is_symlink()
            or not SOURCE.is_file() or SOURCE.stat().st_nlink != 1
            or SOURCE.stat().st_dev != Path('/').stat().st_dev or SOURCE.stat().st_size != IMAGE_BYTES):
        raise ValueError('Require the exact original singly linked physical root image')
    record = {'image': str(SOURCE), 'canonical_image': str(image), 'destination': str(DESTINATION),
        'image_bytes': IMAGE_BYTES, 'image_sha256': IMAGE_SHA, 'unit': audit['unit'],
        'mount': str(ARCHIVE / 'files'), 'records': audit['records'],
        'protected': [str(SOURCE), str(image), str(ARCHIVE / 'files'), str(OWNER / 'raw_final_update.pt')]}
    mounts.require_idle(record['protected'])
    mounts.check_mount(record, SOURCE)
    inputs = {str(image): IMAGE_SHA, str(SOURCE): IMAGE_SHA, audit['unit']: UNIT_SHA,
        str(OWNER / 'raw_final_update.pt'): RAW_SHA, str(ARCHIVE / 'audit.json'): sha256(ARCHIVE / 'audit.json')}
    sources = {}
    for name, expected in CONFIGS.items():
        folder = ROOT / name
        if sha256(folder / 'config.json') != expected or read(folder / 'registration.json')['config_sha256'] != expected:
            raise ValueError('Never modify either original V53 registration')
        for filename in ('config.json', 'registration.json', 'status.json'):
            inputs[str(folder / filename)] = sha256(folder / filename)
        for path, digest in read(folder / 'config.json')['source_sha256'].items():
            if path in sources and sources[path] != digest:
                raise ValueError('Never reconcile inconsistent registered sources')
            sources[path] = digest
    if len(sources) != 555:
        raise ValueError('Keep the complete unchanged V53 source inventory')
    inputs[str(OWNER / 'status.json')] = sha256(OWNER / 'status.json')
    verify_hashes(inputs)
    verify_hashes(sources)
    return record, inputs, sources


def check_preserved(config):
    record = config['record']
    require_cpu()
    scratch.require_mount()
    require_terminal()
    mounts.require_idle(record['protected'])
    if not SOURCE.is_symlink() or SOURCE.resolve() != DESTINATION or Path(record['canonical_image']).resolve() != DESTINATION:
        raise ValueError('Keep both exact canonical image paths linked to the preserved bytes')
    verify_hashes(config['input_sha256'])
    verify_hashes(config['source_sha256'])
    verify_hashes(config['protected_v53_sources'])
    if sha256(DESTINATION) != IMAGE_SHA or DESTINATION.stat().st_size != IMAGE_BYTES:
        raise ValueError('Never alter serialized archive bytes')
    mount = mounts.check_mount(record, DESTINATION)
    payload = torch.load(OWNER / 'raw_final_update.pt', map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError('The complete original raw endpoint must remain CPU mmap readable')
    del payload
    capacity(restored=True)
    return mount


def preserve():
    if OUTPUT.exists() or DESTINATION.parent.exists():
        raise FileExistsError('Never overwrite or restart this preservation')
    with owner_lock():
        record, inputs, protected = inventory()
        capacity(copying=True)
        sources = [Path(__file__).resolve(), Path(mounts.__file__).resolve(), Path(scratch.__file__).resolve(),
            REPO / 'scripts/relocate_single_model_gradient_archive.py',
            REPO / 'tests/test_single_model_v53_root_archive.py']
        config = {'policy': POLICY, 'record': record, 'input_sha256': inputs, 'protected_v53_sources': protected,
            'source_sha256': {str(path): sha256(path) for path in sources},
            'root_free_before': shutil.disk_usage('/').free, 'provider_persistent_backup': False}
        OUTPUT.mkdir()
        write_json(OUTPUT / 'config.json', config)
        status('preserving_closed_archive')
        result = relocate(SOURCE, DESTINATION, IMAGE_SHA, minimum_free=50 * 1024**3)
        write_json(OUTPUT / 'durable_copy.json', result)
        mounts.require_idle(record['protected'])
        subprocess.run(['systemctl', 'restart', Path(record['unit']).name], check=True)
        mount = check_preserved(config)
        write_json(OUTPUT / 'preservation.json', {'complete': True, 'copy': result, 'mount': mount,
            'config_sha256': sha256(OUTPUT / 'config.json'), 'root_free_after': shutil.disk_usage('/').free})
        status('awaiting_independent_recheck')


def recheck():
    if (OUTPUT / 'independent_recheck.json').exists() or read(OUTPUT / 'status.json')['state'] != 'awaiting_independent_recheck':
        raise ValueError('Require the one completed original preservation, without repeating recheck')
    with owner_lock():
        config = read(OUTPUT / 'config.json')
        if read(OUTPUT / 'preservation.json')['config_sha256'] != sha256(OUTPUT / 'config.json'):
            raise ValueError('Never change the registered preservation')
        mount = check_preserved(config)
        write_json(OUTPUT / 'independent_recheck.json', {'complete': True,
            'config_sha256': sha256(OUTPUT / 'config.json'), 'mount': mount,
            'bytes_relocated_to_local_scratch': IMAGE_BYTES, 'protected_v53_sources_checked': 555,
            'all_original_bytes_and_paths_preserved': True, 'canonical_torch_mmap_readable': True,
            'root_free_bytes': shutil.disk_usage('/').free, 'provider_persistent_backup': False,
            'scientific_training_or_evaluation_performed': False, 'original_failed_pipeline_unchanged': True})
        status('complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('preserve', 'recheck'))
    arguments = parser.parse_args()
    try:
        (preserve if arguments.stage == 'preserve' else recheck)()
    except BaseException as error:
        if OUTPUT.exists() and not (OUTPUT / 'independent_recheck.json').exists():
            status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
