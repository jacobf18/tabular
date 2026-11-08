"""One million additional synthetic matrices, with frozen UCI-first monitoring."""

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

import evaluate_cylinder_completion as shared
import million_matrix_monitor as monitoring
import science_retention_benchmarks as retention
import train_scientific_matrixformer as pilot
from million_matrix_prior import (
    DOMAINS, SCIENCE_FAMILIES, MASK_FAMILIES, COMPONENTS, NEURAL_FAMILIES, WEIGHTS,
    PREFLIGHT_SEED, MillionSampler, teacher_weight, training_specs, validation_specs,
)
from recsys_evaluation_common import sha256, write_json
from streamed_large_matrix_training import atomic_torch_save, rng_state, restore_rng
from tabimpute.streamed_inference import streamed_point_predictions
from tabimpute.streamed_training import streamed_training_loss


OUTPUT = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261108/million_v1')
MILESTONES = (25_000, 100_000, 250_000, 500_000, 1_000_000)


def learning_rate(step, config):
    if step <= config['warmup_steps']:
        return config['peak_lr'] * step / config['warmup_steps']
    if step <= config['plateau_steps']:
        return config['peak_lr']
    progress = min(1., (step - config['plateau_steps']) / (config['steps'] - config['plateau_steps']))
    return config['peak_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def objective(distribution, logits, truth, reference, domain, matrices):
    mean = distribution.mean(logits)
    truth_huber = functional.huber_loss(mean, truth, delta=1., reduction='none')
    truth_weight = 4. if domain == 'science' else (2. if domain == 'imputation' else 1.)
    cdf_target = distribution.bucket_means[None, :] >= truth[:, None]
    cdf = (logits.softmax(-1).cumsum(-1) - cdf_target.float()).square().mean(-1)
    nll_weight = .2 if domain == 'recsys' else 1.
    result = nll_weight * distribution(logits, truth) + truth_weight * truth_huber + .05 * cdf
    if domain != 'science':
        result = result + teacher_weight(domain, matrices) * functional.huber_loss(mean, reference.detach(), delta=1., reduction='none')
    return result


def setup(config):
    model, anchor, optimizer, distribution, unused = pilot.setup(config)
    del unused
    anchor.load_state_dict(torch.load(config['anchor_checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
    anchor.eval().requires_grad_(False)
    return model, anchor, optimizer, distribution, MillionSampler(config)


def backward(model, anchor, distribution, episode, matrices):
    domain = episode.metadata['domain']
    if domain == 'science':
        reference = episode.target.detach()
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            reference = streamed_point_predictions(anchor, episode.source, distribution, **pilot.OPTIONS).clone()

    def loss_function(logits, truth, target):
        return objective(distribution, logits, truth, target, domain, matrices)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = streamed_training_loss(model, episode.source, episode.target, episode.mask, loss_function,
                                      auxiliary_targets=reference, **pilot.TRAIN_OPTIONS)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training objective')
    (WEIGHTS[domain] * loss).backward()
    return float(loss.detach())


def register(output):
    warm = monitoring.warm_reference()
    if output.exists():
        raise FileExistsError('Use a new experiment directory; never overwrite a registered run')
    output.mkdir(parents=True)
    monitoring.register_protocol(output / 'monitor_protocol')
    config = pilot.make_config(argparse.Namespace(steps=250_000, validate_every=1000, peak_lr=5e-6, memory_fraction=.25))
    config.update(
        created=datetime.now(timezone.utc).isoformat(), experiment='million_matrix_continuation_v1',
        parent_checkpoint=str(monitoring.WARM_CHECKPOINT), parent_sha256=sha256(monitoring.WARM_CHECKPOINT),
        parent_config_sha256=sha256(monitoring.WARM_CHECKPOINT.with_name('config.json')),
        anchor_checkpoint=str(shared.SINGLE), anchor_sha256=shared.SINGLE_SHA256,
        steps=250_000, matrices_per_update=4, target_matrices=1_000_000, milestones=list(MILESTONES),
        seed=8110803, data_seed=1_810_000_000, warmup_steps=1000, plateau_steps=50_000,
        peak_lr=5e-6, minimum_lr=5e-7, memory_fraction=.85, grad_clip=.5,
        phases=[dict(name='million_matrix', end_step=250_000, rows_low=32, rows_high=2000, cols_low=8, cols_high=1000, batch_size=1)],
        validation=validation_specs(), science_families=list(SCIENCE_FAMILIES),
        structured_families=list(MASK_FAMILIES), structured_components=list(COMPONENTS),
        mixture_weights=WEIGHTS, warm_reference=warm,
        mixture='One matrix from each of science, imputation, ratings, and score panels per update; 250k draws per domain. Loss weights .35/.35/.20/.10. Imputation alternates 8 native components and all48 structured component/mask combinations.',
        shape_policy='Requested shapes:80% small,18% medium,2% large; .1% each1536x512 and2000x1000. Actual latent/neural contexts remain intrinsically small; score panels capped512x128. Log actual dimensions and cell counts.',
        objective='True NLL + truth Huber (science4/imputation2/others1) +.05 CDF; rating NLL .2. Original-model Huber regularizer on other domains: imputation2->.5, ratings/score8->4 over250k matrices. No science teacher loss.',
        selection='At25k/100k/250k/500k/1M matrices evaluate full UCI,72 science cases,and legacy screen. Preserve best UCI; joint best must not worsen warm-start UCI or science aggregate and allows <=10% original aggregate MissBench regression. Balanced best also retains original LLM RMSE/R2 and RecSys within registered limits. No individual MissBench veto.',
        inference='Architecture and native-orientation inference unchanged; benchmark normalization/chunking/scorers unchanged',
        training_data='Synthetic only. No real measurements, masks, labels or fitted benchmark parameters enter gradients.',
        transfer='Previously inspected real benchmarks are development monitors. Damped-oscillator and spatial-field families remain excluded from training. MRI and MovieLens10M excluded.',
        monitor_registry=str(output / 'monitor_protocol'), science_registry=str(retention.ROOT / 'science_monitor_v2'),
        retention_full=str(retention.ROOT / 'retention_full_v1'), checkpoint_every=250, checkpoint_seconds=300,
    )
    sources = dict(config['source_sha256'])
    for module in (monitoring, retention):
        sources[str(Path(module.__file__).resolve())] = sha256(module.__file__)
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    config['source_sha256'] = sources
    config['registry_sha256'] = {config[key]: sha256(Path(config[key]) / 'protocol.json')
                                 for key in ('monitor_registry', 'science_registry', 'retention_full')}
    baseline_files = [monitoring.WARM_UCI / name for name in ('manifest.json', 'per_case.csv', 'status.json')]
    baseline_files += [monitoring.WARM_RUN / 'snapshots/step_3000' / name for name in
                      ('science/manifest.json', 'science/summary.csv', 'retention/manifest.json', 'retention/records.jsonl')]
    config['baseline_sha256'] = {str(path): sha256(path) for path in baseline_files}
    for name in sources:
        destination = output / 'sources' / Path(name).relative_to(shared.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    write_json(output / 'config.json', config)
    write_json(output / 'status.json', dict(state='registered', target_matrices=1_000_000, updated=time.time()))


def verify(config):
    pilot.verify(config)
    if sha256(config['anchor_checkpoint']) != config['anchor_sha256']:
        raise ValueError('Frozen retention teacher changed')
    for name, expected in config['registry_sha256'].items():
        if sha256(Path(name) / 'protocol.json') != expected:
            raise ValueError('Benchmark registry changed')
    for name, expected in config['baseline_sha256'].items():
        if sha256(name) != expected:
            raise ValueError('Warm-start reference changed')


def group_name(spec):
    if spec['domain'] == 'science':
        return f'science/{spec["family"]}'
    if spec.get('kind') == 'structured':
        return f'imputation/{spec["family"]}'
    return spec['domain']


@torch.no_grad()
def synthetic_validation(model, distribution, sampler, config, output, step):
    saved_rng, was_training = rng_state(), model.training
    model.eval()
    rows = []
    try:
        for spec in config['validation']:
            episode = sampler.draw(spec)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = streamed_point_predictions(model, episode.source, distribution, **pilot.OPTIONS)
            error = (prediction[episode.mask] - episode.target[episode.mask]).double()
            if not torch.isfinite(error).all():
                raise FloatingPointError('Nonfinite synthetic validation')
            rows.append(dict(group=group_name(spec), domain=spec['domain'], seed=spec['seed'], rmse=float(error.square().mean().sqrt())))
            del episode, prediction
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
    summary = {group: float(np.mean([row['rmse'] for row in rows if row['group'] == group])) for group in sorted({row['group'] for row in rows})}
    write_json(output / 'synthetic' / f'step_{step:06}.json', dict(step=step, matrices=4 * step, cases=rows, summary=summary))
    return summary


def preflight(config, output):
    model, anchor, optimizer, distribution, sampler = setup(config)
    records = []
    for spec in config['validation']:
        episode = sampler.draw(spec)
        assert episode.source.shape == episode.target.shape == episode.mask.shape
        del episode
    for family in MASK_FAMILIES:
        for component in COMPONENTS:
            spec = dict(domain='imputation', kind='structured', family=family, component=component,
                        shape=[48, 24] if family in NEURAL_FAMILIES else [240, 80], regime='area_scaled',
                        requested_missingness=.6, seed=PREFLIGHT_SEED + 16 * (100 + len(records)))
            episode = sampler.draw(spec)
            records.append(dict(kind='coverage', family=family, component=component, rows=episode.source.shape[1],
                                columns=episode.source.shape[2], rejected_draws=episode.metadata['rejected_draws']))
            del episode
    specs = [dict(spec, seed=PREFLIGHT_SEED + 16 * index, shape=[128, 48]) for index, spec in enumerate(training_specs(1))]
    specs += [dict(domain='imputation', kind='structured', family='mar_neural', component='slow_tail_mcar',
                   shape=[48, 24], regime='area_scaled', requested_missingness=.6, seed=PREFLIGHT_SEED + 10000),
              dict(domain='imputation', kind='native', component=0, regime='area_scaled', shape=[512, 128], seed=PREFLIGHT_SEED + 10016),
              dict(domain='imputation', kind='native', component=0, regime='area_scaled', shape=[2000, 1000], seed=PREFLIGHT_SEED + 10032)]
    for spec in specs:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        episode = sampler.draw(spec)
        loss = backward(model, anchor, distribution, episode, 0)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError('Nonfinite preflight optimizer update')
        row = dict(kind='optimizer', domain=spec['domain'], requested=spec, rows=episode.source.shape[1], columns=episode.source.shape[2],
                   loss=loss, gradient_norm=float(norm), peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
                   seconds=time.monotonic() - started)
        records.append(row)
        print(json.dumps(row), flush=True)
        del episode
    write_json(output / 'preflight.json', dict(passed=True, config_sha256=sha256(output / 'config.json'),
               validation_cases=len(config['validation']), records=records, gradients_synthetic_only=True))


def unique_directory(parent, name):
    path = parent / name
    suffix = 1
    while path.exists():
        suffix += 1
        path = parent / f'{name}_attempt_{suffix:02}'
    return path


def run(config, output, resume=False):
    fingerprint = sha256(output / 'config.json')
    preflight_record = monitoring.read(output / 'preflight.json')
    if not preflight_record['passed'] or preflight_record['config_sha256'] != fingerprint:
        raise ValueError('Require matching successful GPU preflight')
    model, anchor, optimizer, distribution, sampler = setup(config)
    state = dict(step=0, matrices=0, observed_cells=0, query_cells=0, generated_draws=0, rejected_draws=0,
                 domain_counts=dict.fromkeys(DOMAINS, 0), training_seconds=0., last_monitored_step=0, best={}, synthetic_best=1.)
    if resume:
        payload = torch.load(output / 'latest_checkpoint.pth', map_location='cpu', weights_only=True)
        if payload['config_sha256'] != fingerprint:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        state = payload['training_state']
        restore_rng(payload['rng'])
        del payload
    elif (output / 'latest_checkpoint.pth').exists():
        raise ValueError('Existing checkpoint: use --resume')
    session = str(time.time_ns())
    write_json(output / 'sessions' / f'{session}.json', dict(pid=os.getpid(), resume=resume,
               starting_step=state['step'], starting_matrices=state['matrices'], config_sha256=fingerprint))
    stop = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: stop.append(received))
    durations = deque(maxlen=250)
    last_save = time.monotonic()

    def status(kind, **extra):
        write_json(output / 'status.json', dict(state=kind, step=state['step'], matrices=state['matrices'],
                   target_matrices=config['target_matrices'], total_steps=config['steps'], pid=os.getpid(),
                   updated=time.time(), best={name: record['step'] for name, record in state['best'].items()}, **extra))

    def checkpoint(path, role):
        atomic_torch_save(dict(model=model.state_dict(), step=state['step'], epoch=state['step'], matrices=state['matrices'],
                              role=role, config_sha256=fingerprint), path)

    def save_latest():
        nonlocal last_save
        pilot.storage_guard(output)
        atomic_torch_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=state['step'], epoch=state['step'],
                              training_state=state, rng=rng_state(), config_sha256=fingerprint), output / 'latest_checkpoint.pth')
        last_save = time.monotonic()

    def retain(result, science_summary):
        for name, score, eligible in [('uci', result['uci'], True), ('joint', result['joint_score'], result['joint_eligible']),
                                      ('balanced', result['joint_score'], result['balanced_eligible'])]:
            if eligible and (name not in state['best'] or score < state['best'][name]['score']):
                path = output / f'best_{name}_checkpoint.pth'
                checkpoint(path, f'best_{name}' if state['step'] else 'warm_start')
                state['best'][name] = dict(step=state['step'], matrices=state['matrices'], score=score,
                                          checkpoint=str(path), science_summary=str(science_summary), metrics=result)
        write_json(output / 'best_checkpoints.json', state['best'])

    def monitor():
        saved_rng = rng_state()
        save_latest()
        snapshot = unique_directory(output / 'snapshots', f'matrices_{state["matrices"]:07}')
        snapshot.mkdir(parents=True)
        shutil.copyfile(output / 'config.json', snapshot / 'config.json')
        checkpoint(snapshot / 'checkpoint.pth', 'fixed_milestone')
        status('benchmark_monitor', snapshot=str(snapshot))
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(.45)
        try:
            result = monitoring.evaluate(snapshot / 'checkpoint.pth', Path(config['monitor_registry']), Path(config['science_registry']),
                                         snapshot / 'evaluation', config['warm_reference'])
            retain(result, snapshot / 'evaluation/science/summary.csv')
            state['last_monitored_step'] = state['step']
            pilot.append_record(output / 'monitoring.jsonl', dict(step=state['step'], matrices=state['matrices'], result=result, snapshot=str(snapshot)))
        finally:
            torch.cuda.set_per_process_memory_fraction(config['memory_fraction'])
            restore_rng(saved_rng)
        save_latest()

    if not resume:
        status('synthetic_baseline')
        baseline = synthetic_validation(model, distribution, sampler, config, output, 0)
        state['synthetic_baseline'] = baseline
        retain(config['warm_reference'], monitoring.WARM_RUN / 'snapshots/step_3000/science/summary.csv')
        save_latest()
    if state['matrices'] in config['milestones'] and state['last_monitored_step'] < state['step']:
        monitor()
    for step in range(state['step'] + 1, config['steps'] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step, config)
        for group in optimizer.param_groups:
            group['lr'] = rate
        records = []
        for spec in training_specs(step):
            episode = sampler.draw(spec, step)
            loss = backward(model, anchor, distribution, episode, state['matrices'])
            records.append(dict(episode.metadata, loss=loss))
            del episode
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        state['step'], state['matrices'] = step, 4 * step
        duration = time.monotonic() - started
        state['training_seconds'] += duration
        durations.append(duration)
        for record in records:
            state['domain_counts'][record['domain']] += 1
            state['observed_cells'] += record['observed_cells']
            state['query_cells'] += record['query_cells']
            state['rejected_draws'] += len(record['rejected_draws'])
            state['generated_draws'] += 1 + len(record['rejected_draws'])
        pilot.append_record(output / 'training.jsonl', dict(session=session, step=step, matrices=4 * step, lr=rate, gradient_norm=float(norm),
                            seconds=duration, episodes=records))
        if step % 10 == 0 or step == 1:
            status('training', domain_counts=state['domain_counts'], observed_cells=state['observed_cells'], query_cells=state['query_cells'],
                   generated_draws=state['generated_draws'], rejected_draws=state['rejected_draws'],
                   recent_seconds_per_update=float(np.mean(durations)), training_seconds=state['training_seconds'])
        if step % 100 == 0:
            print(json.dumps(dict(step=step, matrices=4 * step, lr=rate, recent_seconds_per_update=float(np.mean(durations)))), flush=True)
        if step % config['validate_every'] == 0:
            status('synthetic_validation')
            summary = synthetic_validation(model, distribution, sampler, config, output, step)
            score = float(np.exp(np.mean([np.log(summary[group] / max(state['synthetic_baseline'][group], 1e-12)) for group in summary])))
            if score < state['synthetic_best']:
                state['synthetic_best'] = score
                checkpoint(output / 'best_synthetic_checkpoint.pth', 'synthetic_diagnostic_only')
            pilot.append_record(output / 'synthetic_validation.jsonl', dict(step=step, matrices=4 * step, score=score, summary=summary))
        if state['matrices'] in config['milestones']:
            monitor()
        elif step % config['checkpoint_every'] == 0 or time.monotonic() - last_save >= config['checkpoint_seconds']:
            save_latest()
        if stop:
            save_latest()
            status('interrupted', signal=stop[-1])
            return
    save_latest()
    selected = state['best'].get('balanced', state['best']['joint'])
    status('final_full_retention', selected=selected['checkpoint'])
    del model, anchor, optimizer, distribution, sampler
    gc.collect()
    torch.cuda.empty_cache()
    full_output = unique_directory(output, 'final_full_retention')
    retention.evaluate(Path(selected['checkpoint']), Path(config['retention_full']), full_output)
    rows = [json.loads(line) for line in (full_output / 'records.jsonl').read_text().splitlines()]
    science_summary = pd.read_csv(selected['science_summary']).to_dict('records')
    result = monitoring.summarize(rows, science_summary, config['warm_reference'])
    recommended = selected['step'] > 0 and result['balanced_eligible']
    write_json(full_output / 'million_matrix_selection.json', dict(result=result, recommended=recommended, selected=selected))
    if recommended:
        shutil.copyfile(selected['checkpoint'], output / 'recommended_checkpoint.pth')
    status('complete', recommended=recommended, final_full_retention=str(full_output), selected=selected['checkpoint'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'preflight', 'train'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == 'register':
        register(output)
        return
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            config = monitoring.read(output / 'config.json')
            verify(config)
            if arguments.stage == 'preflight':
                preflight(config, output)
            else:
                run(config, output, arguments.resume)
        except Exception:
            write_json(output / 'error.json', dict(error=traceback.format_exc(), updated=time.time()))
            write_json(output / 'status.json', dict(state='failed', pid=os.getpid(), updated=time.time()))
            raise


if __name__ == '__main__':
    main()
