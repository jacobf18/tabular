"""Synthetic-only continuation with scientific priors and frozen-parent rehearsal."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from importlib import resources
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import torch
from torch.nn import functional

import evaluate_cylinder_completion as shared
from single_model_distillation import Episode, MixedSampler, episode_metrics
from streamed_large_matrix_training import atomic_torch_save, learning_rate, restore_rng, rng_state
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.prior.scientific_panels import MASK_PATTERNS, TRAIN_FAMILIES, sample_scientific_episode
from tabimpute.streamed_inference import streamed_point_predictions
from tabimpute.streamed_training import streamed_training_loss


DEFAULT_OUTPUT = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v1')
OPTIONS = dict(attention_batch_size=32, cell_batch_size=4096, encoder_row_batch_size=32)
TRAIN_OPTIONS = dict(**OPTIONS, decoder_cell_batch_size=1024)


def objective(distribution, logits, truth, point_target, *, nll_weight=1.):
    mean = distribution.mean(logits)
    huber = functional.huber_loss(mean, point_target.detach(), delta=1., reduction='none')
    target_cdf = distribution.bucket_means[None, :] >= truth[:, None]
    cdf = (logits.softmax(-1).cumsum(-1) - target_cdf.float()).square().mean(-1)
    return nll_weight * distribution(logits, truth) + 2. * huber + .05 * cdf


def append_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()


def storage_guard(output):
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise OSError('Require 8 GiB free for atomic checkpoints and logs')


def validation_specs():
    specs = []
    for index, family in enumerate(TRAIN_FAMILIES):
        for draw, (rows, columns) in enumerate(((96, 16), (288, 48), (192, 24), (128, 32))):
            specs.append(dict(domain='science', family=family, rows=rows, columns=columns,
                              pattern=MASK_PATTERNS[(index + draw) % len(MASK_PATTERNS)],
                              seed=99110600 + index * 10 + draw))
    for domain in ('imputation', 'recsys'):
        for draw, shape in enumerate(((128, 48), (192, 64), (256, 96), (384, 128))):
            specs.append(dict(domain=domain, shape=list(shape), seed=99210600 + draw + 100 * (domain == 'recsys')))
    return specs


def make_config(arguments):
    if arguments.steps < 2 or not 1 <= arguments.validate_every <= arguments.steps:
        raise ValueError('Invalid training horizon or validation interval')
    if not 0 < arguments.memory_fraction <= .25:
        raise ValueError('Shared-GPU pilot is capped at 25% of device memory')
    if shared.sha256(shared.SINGLE) != shared.SINGLE_SHA256:
        raise ValueError('Combined parent fingerprint mismatch')
    sources = {Path(__file__).resolve()}
    sources.update((shared.ROOT / 'src/tabimpute').rglob('*.py'))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, '__file__', None) and str(Path(module.__file__).resolve()).startswith(str(shared.ROOT))
                   and str(module.__file__).endswith('.py') and Path(module.__file__).is_file())
    return dict(
        created=datetime.now(timezone.utc).isoformat(), model=json.loads(shared.SINGLE.with_name('config.json').read_text())['model'],
        parent_checkpoint=str(shared.SINGLE), parent_sha256=shared.SINGLE_SHA256,
        parent_config_sha256=shared.sha256(shared.SINGLE.with_name('config.json')),
        source_sha256={str(path): shared.sha256(path) for path in sorted(sources)},
        steps=arguments.steps, validate_every=arguments.validate_every, peak_lr=arguments.peak_lr,
        seed=8110601, data_seed=82110600, memory_fraction=arguments.memory_fraction,
        environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__),
        warmup_steps=min(30, arguments.steps - 1), weight_decay=.01, grad_clip=.5,
        phases=[dict(name='rehearsal', end_step=arguments.steps, rows_low=96, rows_high=384,
                     cols_low=32, cols_high=128, batch_size=1)],
        science_families=list(TRAIN_FAMILIES), missingness_patterns=list(MASK_PATTERNS),
        validation=validation_specs(), train_options=TRAIN_OPTIONS,
        mixture='Every update: one science matrix and one alternating old imputation/recsys matrix; equal matrix weights',
        objective='NLL + 2*Huber(mean,target,delta=1) + .05*CDF-MSE; science point target=true synthetic measurement; replay point target=frozen combined parent; recsys NLL weight=.1',
        selection='Earliest minimum geometric mean of six science family RMSE/initial ratios, provided each legacy domain RMSE <=1.03*initial; step zero eligible',
        training_data='Only generated synthetic arrays. Previously evaluated real datasets are development benchmarks, never gradient inputs.',
        transfer='Damped-oscillator and spatial-field generators excluded from training and checkpoint selection; no claim of untouched real-domain confirmation',
        inference='Same architecture and interface; standard orientation, full provided context, observed-only normalization; no adapter or ensemble required',
    )


def verify(config):
    for path, expected in config['source_sha256'].items():
        if shared.sha256(path) != expected:
            raise ValueError(f'Registered source changed: {path}')
    if shared.sha256(config['parent_checkpoint']) != config['parent_sha256']:
        raise ValueError('Parent changed')
    if shared.sha256(Path(config['parent_checkpoint']).with_name('config.json')) != config['parent_config_sha256']:
        raise ValueError('Parent configuration changed')


def setup(config):
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(config['memory_fraction'])
    model = TabImputeModelStable(**config['model']).cuda()
    payload = torch.load(config['parent_checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    teacher = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['peak_lr'], weight_decay=config['weight_decay'], fused=True)
    borders = torch.load(resources.files('tabimpute.data').joinpath('borders.pt'), weights_only=True, map_location='cuda')
    distribution = FullSupportBarDistribution(borders)
    return model, teacher, optimizer, distribution, MixedSampler(config)


def science_episode(spec, device='cuda'):
    fields = {key: value for key, value in spec.items() if key != 'domain'}
    generated = sample_scientific_episode(**fields)
    return Episode(torch.from_numpy(generated.source)[None].to(device),
                   torch.from_numpy(generated.target)[None].to(device),
                   torch.from_numpy(generated.query)[None].to(device),
                   dict(generated.metadata, domain='science'))


def training_episodes(config, sampler, step):
    random = np.random.default_rng(config['data_seed'] + step)
    science = science_episode(dict(seed=config['data_seed'] + step, family=TRAIN_FAMILIES[(step - 1) % len(TRAIN_FAMILIES)],
                     rows=int(random.choice([64, 128, 256, 384])), columns=int(random.choice([8, 16, 32, 48, 64])),
                     pattern=MASK_PATTERNS[((step - 1) // len(TRAIN_FAMILIES)) % len(MASK_PATTERNS)]))
    yield science
    domain = 'imputation' if step % 2 else 'recsys'
    shape = (512, 128) if step % 20 == 0 else [(128, 48), (192, 64), (256, 96)][int(random.integers(3))]
    yield sampler.sample(domain, step, shape=shape)


def backward_episode(model, teacher, distribution, episode, weight=.5):
    if episode.metadata['domain'] == 'science':
        target = episode.target.detach()
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            target = streamed_point_predictions(teacher, episode.source, distribution, **OPTIONS).clone()
    nll_weight = .1 if episode.metadata['domain'] == 'recsys' else 1.

    def loss_function(logits, truth, frozen):
        return objective(distribution, logits, truth, frozen, nll_weight=nll_weight)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = streamed_training_loss(model, episode.source, episode.target, episode.mask,
                                      loss_function, auxiliary_targets=target, **TRAIN_OPTIONS)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    (weight * loss).backward()
    return float(loss.detach())


def score_episode(prediction, episode):
    if episode.metadata['domain'] != 'science':
        return episode_metrics(prediction, episode)
    error = (prediction - episode.target).double()
    counts = episode.mask.sum(dim=1)
    squared = error.square().masked_fill(~episode.mask, 0.).sum(dim=1) / counts.clamp_min(1)
    absolute = error.abs().masked_fill(~episode.mask, 0.).sum(dim=1) / counts.clamp_min(1)
    return dict(rmse=float(squared[counts > 0].sqrt().mean()), mae=float(absolute[counts > 0].mean()))


@torch.no_grad()
def validate(model, distribution, sampler, config, output, step):
    model.eval()
    rows = []
    for index, spec in enumerate(config['validation']):
        if spec['domain'] == 'science':
            episode = science_episode(spec)
        else:
            episode = sampler.sample(spec['domain'], 1, seed=spec['seed'], shape=spec['shape'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = streamed_point_predictions(model, episode.source, distribution, **OPTIONS)
        if not torch.isfinite(prediction[episode.mask]).all():
            raise FloatingPointError('Nonfinite validation prediction')
        rows.append(dict(index=index, group=spec.get('family', spec['domain']), **score_episode(prediction, episode)))
        if step == 0:
            np.savez_compressed(output / 'validation_inputs' / f'{index:02}.npz',
                                source=episode.source.cpu().numpy(), target=episode.target.cpu().numpy(),
                                query=episode.mask.cpu().numpy())
    groups = sorted({record['group'] for record in rows})
    summary = {group: float(np.mean([record['rmse'] for record in rows if record['group'] == group])) for group in groups}
    shared.write_json(output / f'validation_{step:04}.json', dict(step=step, summary=summary, cases=rows))
    model.train()
    return summary


def selection_score(current, baseline):
    ratios = {group: current[group] / max(baseline[group], 1e-8) for group in baseline}
    score = math.exp(np.mean([math.log(max(ratios[family], 1e-8)) for family in TRAIN_FAMILIES]))
    eligible = all(ratios[domain] <= 1.03 for domain in ('imputation', 'recsys'))
    return score, eligible, ratios


def register(arguments):
    config = make_config(arguments)
    arguments.output.mkdir(parents=True, exist_ok=False)
    storage_guard(arguments.output)
    (arguments.output / 'validation_inputs').mkdir()
    (arguments.output / 'sources').mkdir()
    for path in config['source_sha256']:
        relative = Path(path).relative_to(shared.ROOT)
        destination = arguments.output / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shared.write_json(arguments.output / 'config.json', config)
    shared.write_json(arguments.output / 'status.json', dict(state='registered', updated=time.time()))
    print('Registered', arguments.output, flush=True)


def train(output, resume=False, preflight=False):
    config = json.loads((output / 'config.json').read_text())
    verify(config)
    fingerprint = shared.sha256(output / 'config.json')
    model, teacher, optimizer, distribution, sampler = setup(config)
    model.train()
    step, best_step, best_score = 0, 0, 1.
    stop = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: stop.append(received))
    if preflight:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for domain in ('science', 'imputation', 'recsys'):
            episode = science_episode(dict(seed=521106, rows=96, columns=16)) if domain == 'science' else sampler.sample(domain, 1, shape=(128, 48))
            loss = backward_episode(model, teacher, distribution, episode, weight=1 / 3)
            rows.append(dict(domain=domain, loss=loss, queries=int(episode.mask.sum())))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        shared.write_json(output / 'preflight.json', dict(config_sha256=fingerprint, records=rows,
                          gradient_norm=float(norm), peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
                          finite=all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())))
        print('Capacity/gradient preflight passed; disposable update not saved', flush=True)
        return
    if not (output / 'preflight.json').is_file():
        raise ValueError('Run the gradient/memory preflight first')
    if json.loads((output / 'preflight.json').read_text())['config_sha256'] != fingerprint:
        raise ValueError('Preflight configuration mismatch')
    if resume:
        state = torch.load(output / 'latest_checkpoint.pth', map_location='cpu', weights_only=True)
        if state['config_sha256'] != fingerprint:
            raise ValueError('Checkpoint configuration mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, best_step, best_score = state['step'], state['best_step'], state['best_score']
        restore_rng(state['rng'])
        del state
        baseline = json.loads((output / 'validation_0000.json').read_text())['summary']
    else:
        if (output / 'latest_checkpoint.pth').exists():
            raise ValueError('Existing run requires --resume')
        baseline = validate(model, distribution, sampler, config, output, 0)
        atomic_torch_save(dict(model=model.state_dict(), step=0, epoch=0, config_sha256=fingerprint), output / 'best_validation_checkpoint.pth')
        append_record(output / 'validation.jsonl', dict(step=0, summary=baseline, score=1., eligible=True, best_step=0))

    def save_latest():
        storage_guard(output)
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, epoch=step,
                              best_step=best_step, best_score=best_score, rng=rng_state(), config_sha256=fingerprint),
                          output / 'latest_checkpoint.pth')

    save_latest()
    for step in range(step + 1, config['steps'] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step, config['steps'], config['peak_lr'], config['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = rate
        microsteps = []
        for episode in training_episodes(config, sampler, step):
            loss = backward_episode(model, teacher, distribution, episode)
            microsteps.append(dict(episode.metadata, loss=loss, queries=int(episode.mask.sum())))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        record = dict(step=step, lr=rate, gradient_norm=float(norm), seconds=time.monotonic() - started, microsteps=microsteps)
        append_record(output / 'training.jsonl', record)
        shared.write_json(output / 'status.json', dict(state='training', pid=os.getpid(), step=step, total=config['steps'],
                          best_step=best_step, best_score=best_score, updated=time.time()))
        print(json.dumps(dict(step=step, losses=[item['loss'] for item in microsteps], seconds=record['seconds'])), flush=True)
        if step % config['validate_every'] == 0 or step == config['steps']:
            summary = validate(model, distribution, sampler, config, output, step)
            score, eligible, ratios = selection_score(summary, baseline)
            if eligible and score < best_score:
                best_step, best_score = step, score
                storage_guard(output)
                atomic_torch_save(dict(model=model.state_dict(), step=step, epoch=step, config_sha256=fingerprint),
                                  output / 'best_validation_checkpoint.pth')
            append_record(output / 'validation.jsonl', dict(step=step, summary=summary, score=score, eligible=eligible,
                          ratios=ratios, best_step=best_step))
            print(json.dumps(dict(validation_step=step, score=score, eligible=eligible, ratios=ratios, best_step=best_step)), flush=True)
            save_latest()
        elif step % 25 == 0:
            save_latest()
        if stop:
            save_latest()
            shared.write_json(output / 'status.json', dict(state='interrupted', step=step, best_step=best_step, updated=time.time()))
            return
    shared.write_json(output / 'status.json', dict(state='complete', step=step, best_step=best_step,
                      best_score=best_score, updated=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'preflight', 'train'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--validate-every', type=int, default=50)
    parser.add_argument('--peak-lr', type=float, default=1e-5)
    parser.add_argument('--memory-fraction', type=float, default=.16)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    if arguments.stage == 'register':
        register(arguments)
        return
    with (arguments.output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            train(arguments.output, resume=arguments.resume, preflight=arguments.stage == 'preflight')
        except Exception as error:
            shared.write_json(arguments.output / 'status.json', dict(state='failed', pid=os.getpid(), error=repr(error), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
