"""Preserve eight closed early-candidate prediction files while V53 runs."""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import types

import numpy as np
import torch

import relocate_single_model_legacy_root_banks as blueprint
from relocate_single_model_legacy_root_banks import readers, copying
import relocate_single_model_gradient_archive as durable
from relocate_single_model_gradient_archive import sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT, SOURCE = blueprint.ROOT, blueprint.SOURCE
OUTPUT = ROOT / 'early_candidate_prediction_preservation_v1'
DESTINATION = copying.SCRATCH / 'single_model_goal_20261005_owned/early_candidate_predictions_v1'
OWNERS = {
    'mixed_distill_v1_benchmarks': ('complete', 4, '02f63915e516501576b7a73c05ef6b58b28c7b1d7d2691d60c35e96d30fe884a'),
    'mixed_distill_v2_benchmarks': ('complete', 4, 'c4f82e80cf391e8689a2573a4323cb4555e82fdfbacf3e670d3265789db4987d'),
}
QUERIES = {'100K': 19971, 'netflix': 121470, '1M': 200013, '10M': 1999971}
TARGETS = {
    'mixed_distill_v1_benchmarks/recsys/100K/student.npz': ('1ecc35fa728f7c896b6dd030ba9b364cbd871cc1128f4debd12aca12f26be849', 152136),
    'mixed_distill_v1_benchmarks/recsys/netflix/student.npz': ('7339b1ca7e278c030355cb1bc5ce3d1c917e6f35db2d53f517babec45f74eb2c', 904003),
    'mixed_distill_v1_benchmarks/recsys/1M/student.npz': ('55af3ba73a2749fd0abe4d8c3f69217fd9a8fd021e0af8e9e1d2abe924748a7f', 1516989),
    'mixed_distill_v1_benchmarks/recsys/10M/student.npz': ('487e422c829e53508e31c51173e523ccba183c2422f78491f6cd54a27f970016', 14963467),
    'mixed_distill_v2_benchmarks/recsys/100K/student.npz': ('bdbc394e281742336ca43cbf5d40066bca720d6ed291b3aaa8f6c44772f08679', 152162),
    'mixed_distill_v2_benchmarks/recsys/netflix/student.npz': ('38edcbc49646b6c8da2761551668d2d9de082dc26e4c5ef0e9e8d7bcc353dfc7', 904294),
    'mixed_distill_v2_benchmarks/recsys/1M/student.npz': ('3d020bddafbd4a3b5eb388e76315b3ea883cd18d8587aa5f0439aba7716fe4d1', 1517748),
    'mixed_distill_v2_benchmarks/recsys/10M/student.npz': ('0f98ea714eff7560131f9486a6e4a56102e344f70c458d576f5bd1df690e63c8', 14968221),
}
BYTES = 35079020
LIVE_CONFIGS = {
    'balanced_structured_components_v53': 'f1e90786a292bb5054ab0eb8ab7916aad5539ddb7da2e4bd3ca453ddbf64b5da',
    'balanced_structured_components_v53_pipeline': 'c126b3a5739b4355aa7b816dcd9d9214f96ae964c959d1128dde3a2402d92a86',
}
POLICY = (
    'Storage only: preserve all eight completed V1/V2 candidate RecSys prediction NPZ files, '
    'chosen by physical storage ownership rather than accuracy. Keep exact float64 predictions, '
    'uint8 overlap counts, every query, original serialized bytes and canonical paths. No '
    'rounding, recompression, metric recomputation, model inference, training, confirmation, '
    'selection, new seed or scientific-success claim. Original campaign audits establish '
    'storage provenance only; do not upgrade their scientific scope. Require stopped exact '
    'owners, exclusive original owner locks and no open artifact readers. Copy/fsync/hash/typed-'
    'read before atomic canonical symlink publication and separate independent recheck. '
    'Never move a model checkpoint, teacher, actual V29 initializer, baseline prediction, '
    'mounted archive, failed V28 evidence or active V53 output. Protect both registered V53 '
    'source sets/configurations. Preserve18/6/50 GiB checkpoint/root/scratch reserves and16 MiB '
    'root stage headroom. Scratch is machine-local, not external/provider-persistent backup. '
    'Do not stop, restart or modify the live scientific chains.'
)


for name in ('read', 'status', 'require_cpu', 'require_stopped', 'require_reserves', 'copy_and_verify'):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Storage callbacks must bind only the declared prediction scope')
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)


@contextmanager
def owner_locks():
    with ExitStack() as stack:
        for name in OWNERS:
            stream = stack.enter_context((SOURCE / name / 'run.lock').open('r'))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_stopped()
        yield


def validate_arrays(arrays, queries):
    if (set(arrays) != {'prediction', 'counts'} or type(queries) is not int or queries <= 0
            or arrays['prediction'].shape != (queries,) or arrays['prediction'].dtype != np.float64
            or arrays['counts'].shape != (queries,) or arrays['counts'].dtype != np.uint8
            or not np.isfinite(arrays['prediction']).all() or not (arrays['counts'] > 0).all()):
        raise ValueError('Preserve every finite float64 prediction and positive uint8 overlap count')


def check_file(path, name):
    if name not in TARGETS:
        raise ValueError('Only the eight declared closed candidate predictions may move')
    expected, size = TARGETS[name]
    if Path(path).stat().st_size != size or sha256(path) != expected:
        raise ValueError('Preserve the exact original serialized prediction bytes')
    with np.load(path, allow_pickle=False) as arrays:
        validate_arrays(arrays, QUERIES[Path(name).parent.name])


def validate_records(records):
    if len(records) != 8 or {row['name'] for row in records} != set(TARGETS):
        raise ValueError('Preserve each of the eight original prediction artifacts exactly once')
    for row in records:
        expected, size = TARGETS[row['name']]
        if (row['source'] != str(SOURCE / row['name']) or row['destination'] != str(DESTINATION / row['name'])
                or row['sha256'] != expected or row['bytes'] != size):
            raise ValueError('Bind only the exact declared original paths, bytes and hashes')
    if sum(row['bytes'] for row in records) != BYTES:
        raise ValueError('Preserve the entire original eight-file byte inventory')


def inventory():
    require_cpu()
    copying.require_mount()
    require_stopped()
    records, inputs, protected_sources = [], {}, {}
    for name, (_, count, digest) in OWNERS.items():
        owner = SOURCE / name
        campaign = read(owner / 'campaign_audit.json')
        audit = read(owner / 'recsys/audit.json')
        if (sha256(owner / 'campaign_audit.json') != digest or campaign['all_evaluations_complete'] is not True
                or campaign['goal_success_not_assessed'] is not True or count != len(QUERIES)
                or campaign['audits']['recsys']['path'] != str(owner / 'recsys/audit.json')
                or sha256(owner / 'recsys/audit.json') != campaign['audits']['recsys']['sha256']
                or audit['complete'] is not True or len(audit['cases']) != 8
                or {(row['dataset'], row['role'], row['queries']) for row in audit['cases']}
                    != {(dataset, role, queries) for dataset, queries in QUERIES.items() for role in ('student', 'parent_same_recipe')}):
            raise ValueError('Keep complete original closed campaign provenance without new efficacy claims')
        checkpoint = SOURCE / name.removesuffix('_benchmarks') / 'best_validation_checkpoint.pth'
        if campaign['candidate']['checkpoint'] != str(checkpoint):
            raise ValueError('Never substitute or move an original selected model')
        inputs[str(checkpoint)] = campaign['candidate']['checkpoint_sha256']
        inputs.update({str(path): sha256(path) for path in owner.glob('*.json')})
        inputs[str(owner / 'recsys/audit.json')] = campaign['audits']['recsys']['sha256']
        for dataset in QUERIES:
            directory = owner / 'recsys' / dataset
            report = read(directory / 'student.json')
            target = f'{name}/recsys/{dataset}/student.npz'
            if (report['role'] != 'student' or report['dataset'] != dataset
                    or report['prediction_sha256'] != TARGETS[target][0]
                    or report['checkpoint_sha256'] != campaign['candidate']['checkpoint_sha256']
                    or report['manifest_sha256'] != audit['manifest_sha256']):
                raise ValueError('Keep the exact original candidate and prediction identities')
            for path in directory.glob('*.json'):
                inputs[str(path)] = sha256(path)
            inputs[str(directory / 'parent_same_recipe.npz')] = read(directory / 'parent_same_recipe.json')['prediction_sha256']
    for name, (digest, size) in TARGETS.items():
        source = SOURCE / name
        if (source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1
                or source.stat().st_dev != Path('/').stat().st_dev):
            raise ValueError('Move only singly linked closed prediction files physically on root')
        check_file(source, name)
        records.append({'name': name, 'source': str(source), 'destination': str(DESTINATION / name), 'sha256': digest, 'bytes': size})
    validate_records(records)
    for name, digest in LIVE_CONFIGS.items():
        path = ROOT / name / 'config.json'
        if sha256(path) != digest or read(path.with_name('registration.json'))['config_sha256'] != digest:
            raise ValueError('Preserve both exact registered live V53 configurations')
        inputs[str(path)] = digest
        inputs[str(path.with_name('registration.json'))] = sha256(path.with_name('registration.json'))
        for source, expected in read(path)['source_sha256'].items():
            if source in protected_sources and protected_sources[source] != expected:
                raise ValueError('Never reconcile inconsistent original source identities')
            protected_sources[source] = expected
    inputs[str(ROOT / 'legacy_strata_v29/selected_checkpoint.pth')] = '9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32'
    inputs[str(SOURCE / 'mixed_distill_v2_tail_average/averaged_checkpoint.pth')] = '3246a013eee512677d195a00b3c0aa37614c2d50a4edbe785c0fd15ae7fddffd'
    verify_hashes(inputs)
    verify_hashes(protected_sources)
    readers.require_no_artifact_readers([row['source'] for row in records])
    return records, inputs, protected_sources


RECHECK_SOURCE = inspect.getsource(blueprint.recheck)
for old, new in (
    ('    hashes = {}', "    validate_records(config['records'])\n    hashes = {}"),
    ("'canonical_banks_checked': 4, 'cases_checked': sum(value[1] for value in OWNERS.values())",
     "'canonical_prediction_files_checked': 8, 'prediction_entries_checked': 2 * sum(QUERIES.values())"),
):
    RECHECK_SOURCE = replace_once(RECHECK_SOURCE, old, new)
exec(compile(RECHECK_SOURCE, __file__, 'exec'), globals())
PRESERVE_SOURCE = inspect.getsource(blueprint.preserve)
for old, new in (
    ('tests/test_single_model_legacy_root_banks.py', 'tests/test_single_model_early_candidate_predictions.py'),
    ('sources = [Path(__file__).resolve(),', 'sources = [Path(__file__).resolve(), Path(blueprint.__file__).resolve(), Path(durable.__file__).resolve(),'),
    ('copying_closed_synthetic_banks', 'copying_closed_candidate_predictions'),
):
    PRESERVE_SOURCE = replace_once(PRESERVE_SOURCE, old, new)
exec(compile(PRESERVE_SOURCE, __file__, 'exec'), globals())


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
