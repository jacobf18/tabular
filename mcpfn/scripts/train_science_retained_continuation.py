"""Longer synthetic continuation with real scientific and legacy retention gates."""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
from torch.nn import functional

import evaluate_cylinder_completion as shared
import science_continuation_monitor as science_monitor
import science_retention_benchmarks as retention
import train_scientific_matrixformer as pilot
from single_model_distillation import Episode
from streamed_large_matrix_training import atomic_torch_save, learning_rate, restore_rng, rng_state
from tabimpute.prior.scientific_panels import TRAIN_FAMILIES, MASK_PATTERNS, sample_scientific_episode
from tabimpute.prior.scientific_processes import PROCESS_FAMILIES, sample_process_episode, sample_score_panel
from tabimpute.streamed_inference import streamed_point_predictions
from tabimpute.streamed_training import streamed_training_loss


ROOT = retention.ROOT
OUTPUT = ROOT / 'retained_v2'
FAMILIES = TRAIN_FAMILIES + PROCESS_FAMILIES
DRAW_FAMILIES = FAMILIES + ('treatment_process', 'assay_covariates')
WEIGHTS = dict(science=.35, imputation=.35, recsys=.20, score_panel=.10)


def from_generated(generated, domain):
    return Episode(torch.from_numpy(generated.source)[None].cuda(), torch.from_numpy(generated.target)[None].cuda(),
                   torch.from_numpy(generated.query)[None].cuda(), dict(generated.metadata, domain=domain))


def make_episode(spec, sampler, step=1):
    spec = dict(spec)
    domain = spec.pop('domain')
    if domain == 'science':
        generator = sample_process_episode if spec['family'] in PROCESS_FAMILIES else sample_scientific_episode
        return from_generated(generator(**spec), domain)
    if domain == 'score_panel':
        return from_generated(sample_score_panel(**spec), domain)
    return sampler.sample(domain, step, **spec)


def validation_specs():
    result = []
    for index, family in enumerate(FAMILIES):
        for draw, shape in enumerate(((96, 16), (256, 48), (192, 32), (64, 12))):
            result.append(dict(domain='science', family=family, seed=940110800 + index * 10 + draw,
                               rows=shape[0], columns=shape[1], pattern=MASK_PATTERNS[(index + draw) % len(MASK_PATTERNS)]))
    for domain_index, domain in enumerate(('imputation', 'recsys', 'score_panel')):
        for draw, shape in enumerate(((64, 16), (128, 48), (256, 96), (384, 128), (192, 64), (256, 32))):
            if domain == 'recsys' and shape[0] < 128:
                shape = (128, 48)
            spec = dict(domain=domain, seed=941110800 + domain_index * 100 + draw)
            spec.update(dict(rows=shape[0], columns=shape[1]) if domain == 'score_panel' else dict(shape=list(shape)))
            result.append(spec)
    return result


def register(arguments):
    output = arguments.output
    if arguments.steps < 2 or not 1 <= arguments.monitor_every <= arguments.steps:
        raise ValueError('Invalid horizon or monitoring interval')
    parent = science_monitor.PILOT / 'best_validation_checkpoint.pth'
    baseline_screen = retention.read(ROOT / 'pilot_retention_screen_v1/gate.json')
    if not baseline_screen['complete']:
        raise ValueError('Finish actual old-benchmark evaluation of the pilot first')
    config = pilot.make_config(argparse.Namespace(steps=arguments.steps, validate_every=100, peak_lr=5e-6, memory_fraction=.25))
    config.update(
        parent_checkpoint=str(parent), parent_sha256=shared.sha256(parent), parent_config_sha256=shared.sha256(parent.with_name('config.json')),
        anchor_checkpoint=str(shared.SINGLE), anchor_sha256=shared.SINGLE_SHA256,
        seed=8110802, data_seed=82110800, validation=validation_specs(), phases=[dict(name='retained_rehearsal', end_step=arguments.steps,
                           rows_low=64, rows_high=512, cols_low=16, cols_high=128, batch_size=1)],
        science_families=list(FAMILIES), science_draw_schedule=list(DRAW_FAMILIES), mixture_weights=WEIGHTS,
        monitor_every=arguments.monitor_every, objective='Science: true NLL +4*true Huber+.05*CDF; old imputation/recsys/score panels: .2/.1/.2 true NLL +8*Huber to original combined parent+.05*CDF',
        mixture='35% scientific truth, 35% old imputation replay, 20% recommender replay, 10% synthetic LLM-like score-panel replay; one independent episode each per update',
        selection='Real development selection: strictly lower mean error on each of seven science datasets, minimize their geometric error ratio, and pass all old-benchmark screen limits. Full old benchmark gate required before recommendation.',
        transfer='No fresh transfer claim: previously inspected scientific and legacy benchmarks are explicit development gates; MRI and MovieLens10M excluded',
        retention_screen=str(ROOT / 'retention_screen_v1'), retention_full=str(ROOT / 'retention_full_v1'), science_registry=str(ROOT / 'science_monitor_v2'),
        pilot_retention_failures=baseline_screen['failures'],
        pilot_retention_gate_sha256=shared.sha256(ROOT / 'pilot_retention_screen_v1/gate.json'),
    )
    for path in [Path(__file__), Path(retention.__file__), Path(science_monitor.__file__)]:
        config['source_sha256'][str(path.resolve())] = shared.sha256(path)
    config['registry_sha256'] = {config[key]: shared.sha256(Path(config[key]) / 'protocol.json')
                                 for key in ('retention_screen', 'retention_full', 'science_registry')}
    output.mkdir(parents=True, exist_ok=False)
    pilot.storage_guard(output)
    (output / 'sources').mkdir()
    for name in config['source_sha256']:
        path = Path(name)
        destination = output / 'sources' / path.relative_to(shared.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shared.write_json(output / 'config.json', config)
    shared.write_json(output / 'status.json', dict(state='registered', updated=time.time()))


def episodes(config, sampler, step):
    random = np.random.default_rng(config['data_seed'] + step)
    family = DRAW_FAMILIES[(step - 1) % len(DRAW_FAMILIES)]
    shape = int(random.choice([32, 64, 128, 256, 384])), int(random.choice([8, 16, 24, 32, 48, 64]))
    yield make_episode(dict(domain='science', family=family, seed=config['data_seed'] + step, rows=shape[0], columns=shape[1],
                            pattern=MASK_PATTERNS[((step-1) // len(DRAW_FAMILIES)) % len(MASK_PATTERNS)]), sampler)
    for index, domain in enumerate(('imputation', 'recsys')):
        shape = [(64, 16), (128, 48), (256, 96), (512, 128)][int(random.integers(4))]
        if domain == 'recsys' and shape[0] < 128:
            shape = (128, 48)
        yield make_episode(dict(domain=domain, seed=config['data_seed'] + step * 16 + index + 300000,
                                shape=shape), sampler, step)
    yield make_episode(dict(domain='score_panel', seed=config['data_seed'] + step + 600000,
                            rows=int(random.choice([96, 160, 256])), columns=int(random.choice([16, 32, 64]))), sampler)


def objective(distribution, logits, truth, point_target, domain):
    mean = distribution.mean(logits)
    point = functional.huber_loss(mean, point_target.detach(), delta=1., reduction='none')
    cdf_target = distribution.bucket_means[None, :] >= truth[:, None]
    cdf = (logits.softmax(-1).cumsum(-1) - cdf_target.float()).square().mean(-1)
    likelihood_weight = 1. if domain == 'science' else (.1 if domain == 'recsys' else .2)
    point_weight = 4. if domain == 'science' else 8.
    return likelihood_weight * distribution(logits, truth) + point_weight * point + .05 * cdf


def backward(model, anchor, distribution, episode):
    domain = episode.metadata['domain']
    if domain == 'science':
        target = episode.target.detach()
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            target = streamed_point_predictions(anchor, episode.source, distribution, **pilot.OPTIONS).clone()

    def loss_function(logits, truth, reference):
        return objective(distribution, logits, truth, reference, domain)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = streamed_training_loss(model, episode.source, episode.target, episode.mask, loss_function,
                                      auxiliary_targets=target, **pilot.TRAIN_OPTIONS)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective')
    (WEIGHTS[domain] * loss).backward()
    return float(loss.detach())


@torch.no_grad()
def synthetic_validation(model, distribution, sampler, config, output, step):
    model.eval()
    rows = []
    for index, spec in enumerate(config['validation']):
        episode = make_episode(spec, sampler)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = streamed_point_predictions(model, episode.source, distribution, **pilot.OPTIONS)
        if not torch.isfinite(prediction[episode.mask]).all():
            raise FloatingPointError('Nonfinite validation prediction')
        rows.append(dict(index=index, group=spec.get('family', spec['domain']), **pilot.score_episode(prediction, episode)))
    summary = {group: float(np.mean([row['rmse'] for row in rows if row['group'] == group])) for group in sorted({row['group'] for row in rows})}
    shared.write_json(output / f'synthetic_{step:04}.json', dict(step=step, cases=rows, summary=summary))
    model.train()
    return summary


def run(output, preflight=False, resume=False):
    config = retention.read(output / 'config.json')
    pilot.verify(config)
    assert shared.sha256(config['anchor_checkpoint']) == config['anchor_sha256']
    for registry, expected in config['registry_sha256'].items():
        assert shared.sha256(Path(registry) / 'protocol.json') == expected
    fingerprint = shared.sha256(output / 'config.json')
    model, anchor, optimizer, distribution, sampler = pilot.setup(config)
    anchor.load_state_dict(torch.load(config['anchor_checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
    anchor.eval().requires_grad_(False)
    model.train()
    if preflight:
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        records = []
        for episode in episodes(config, sampler, 1):
            records.append(dict(domain=episode.metadata['domain'], loss=backward(model, anchor, distribution, episode)))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
        assert finite
        for specification in config['validation']:
            episode = make_episode(specification, sampler)
            assert episode.mask.any() and torch.isfinite(episode.target[episode.mask]).all()
        shared.write_json(output / 'preflight.json', dict(config_sha256=fingerprint, records=records,
                          finite=finite, validation_cases=len(config['validation']),
                          gradient_norm=float(norm), peak_gib=torch.cuda.max_memory_allocated()/1024**3))
        return
    preflight_record = retention.read(output / 'preflight.json')
    assert preflight_record['config_sha256'] == fingerprint and preflight_record['finite']
    step, best_step, best_score, safe_step, safe_score = 0, -1, 1., -1, 1.
    stop = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: stop.append(received))
    if resume:
        state = torch.load(output / 'latest_checkpoint.pth', map_location='cpu', weights_only=True)
        assert state['config_sha256'] == fingerprint
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, best_step, best_score = state['step'], state['best_step'], state['best_score']
        safe_step, safe_score = state['safe_step'], state['safe_score']
        restore_rng(state['rng'])
        del state
    else:
        if (output / 'latest_checkpoint.pth').exists():
            raise ValueError('Use --resume for an existing run')
        for name in ('best_validation_checkpoint.pth', 'best_retention_checkpoint.pth'):
            atomic_torch_save(dict(model=anchor.state_dict(), step=-1, epoch=-1, config_sha256=fingerprint,
                                   role='unchanged_original_fallback'), output / name)

    def save_latest():
        pilot.storage_guard(output)
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, epoch=step,
                              best_step=best_step, best_score=best_score, safe_step=safe_step, safe_score=safe_score,
                              rng=rng_state(), config_sha256=fingerprint), output / 'latest_checkpoint.pth')

    def monitor():
        nonlocal best_step, best_score, safe_step, safe_score
        snapshot = output / 'snapshots' / f'step_{step:04}'
        attempt = 1
        while snapshot.exists():
            attempt += 1
            snapshot = output / 'snapshots' / f'step_{step:04}_attempt_{attempt:02}'
        snapshot.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(output / 'config.json', snapshot / 'config.json')
        checkpoint = snapshot / 'checkpoint.pth'
        atomic_torch_save(dict(model=model.state_dict(), step=step, epoch=step, config_sha256=fingerprint), checkpoint)
        shared.write_json(output / 'status.json', dict(state='benchmark_monitor', step=step, total=config['steps'], pid=os.getpid(), updated=time.time()))
        science = science_monitor.evaluate(checkpoint, Path(config['science_registry']), snapshot / 'science')
        if step == 0:
            assert shared.sha256(ROOT / 'pilot_retention_screen_v1/gate.json') == config['pilot_retention_gate_sha256']
            initial = torch.load(config['parent_checkpoint'], map_location='cpu', weights_only=True)['model']
            assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
            del initial
            legacy = retention.read(ROOT / 'pilot_retention_screen_v1/gate.json')
            shared.write_json(snapshot / 'retention_reuse.json', dict(source=str(ROOT / 'pilot_retention_screen_v1'),
                              initial_parameters_identical=True, gate_sha256=config['pilot_retention_gate_sha256']))
        else:
            legacy = retention.evaluate(checkpoint, Path(config['retention_screen']), snapshot / 'retention')
        if legacy['passed'] and science['score'] < safe_score:
            safe_step, safe_score = step, science['score']
            atomic_torch_save(dict(model=model.state_dict(), step=step, epoch=step, config_sha256=fingerprint), output / 'best_retention_checkpoint.pth')
        eligible = legacy['passed'] and science['all_not_worse'] and science['improved_datasets'] == 7
        if eligible and science['score'] < best_score:
            best_step, best_score = step, science['score']
            atomic_torch_save(dict(model=model.state_dict(), step=step, epoch=step, config_sha256=fingerprint), output / 'best_validation_checkpoint.pth')
        pilot.append_record(output / 'monitoring.jsonl', dict(step=step, science=science, retention=legacy,
                            eligible=eligible, best_step=best_step, safe_step=safe_step))
        torch.cuda.set_per_process_memory_fraction(config['memory_fraction'])
        model.train()
        save_latest()

    if not resume:
        synthetic_validation(anchor, distribution, sampler, config, output, -1)
        anchor.eval()
        synthetic_validation(model, distribution, sampler, config, output, 0)
        monitor()
    for step in range(step + 1, config['steps'] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step, config['steps'], config['peak_lr'], config['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = rate
        records = []
        for episode in episodes(config, sampler, step):
            records.append(dict(episode.metadata, loss=backward(model, anchor, distribution, episode)))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        pilot.append_record(output / 'training.jsonl', dict(step=step, lr=rate, gradient_norm=float(norm),
                            seconds=time.monotonic()-started, episodes=records))
        shared.write_json(output / 'status.json', dict(state='training', step=step, total=config['steps'], pid=os.getpid(),
                          best_step=best_step, safe_step=safe_step, updated=time.time()))
        if step % 10 == 0:
            print(json.dumps(dict(step=step, losses=[record['loss'] for record in records])), flush=True)
        if step % 100 == 0 or step == config['steps']:
            summary = synthetic_validation(model, distribution, sampler, config, output, step)
            pilot.append_record(output / 'synthetic_validation.jsonl', dict(step=step, summary=summary))
        if step % config['monitor_every'] == 0 or step == config['steps']:
            monitor()
        elif step % 50 == 0:
            save_latest()
        if stop:
            save_latest()
            shared.write_json(output / 'status.json', dict(state='interrupted', step=step, best_step=best_step,
                              safe_step=safe_step, updated=time.time()))
            return
    selected = output / ('best_validation_checkpoint.pth' if best_step >= 0 else 'best_retention_checkpoint.pth')
    del model, anchor, optimizer, distribution, sampler
    gc.collect()
    torch.cuda.empty_cache()
    shared.write_json(output / 'status.json', dict(state='full_retention_gate', step=step, best_step=best_step,
                      safe_step=safe_step, pid=os.getpid(), updated=time.time()))
    full_gate = retention.evaluate(selected, Path(config['retention_full']), output / 'final_full_retention')
    recommended = best_step >= 0 and full_gate['passed']
    if recommended:
        shutil.copyfile(selected, output / 'recommended_checkpoint.pth')
    shared.write_json(output / 'status.json', dict(state='complete', step=step, best_step=best_step, safe_step=safe_step,
                      all_science_improved=best_step >= 0, full_retention_passed=full_gate['passed'],
                      recommended=recommended, evaluated_checkpoint=str(selected), updated=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'preflight', 'train'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--monitor-every', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'register':
        register(args)
    else:
        with (args.output / 'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(args.output, preflight=args.stage == 'preflight', resume=args.resume)
            except Exception as error:
                shared.write_json(args.output / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid(), updated=time.time()))
                raise
