"""Synthetic curriculum and streamed teacher-mean objective for the 98k student."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import random

import torch
from torch.nn import functional

from dwip_teacher_mean_training import (
    base, build_training_prior, compute_context, control_recipe, equal_teacher_mean,
)
from rowcol_fixed_validation import preserved_rng
from spectral_support_normalization import apply_shrunk_support_mask
from tabimpute.streamed_inference import streamed_point_predictions
from tabimpute.streamed_training import streamed_training_loss


CHECKPOINT_SHA256 = "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d"
DEFAULT_PHASES = [
    dict(name="bridge", end_step=500, rows_low=256, rows_high=512, cols_low=128, cols_high=256, batch_size=1),
    dict(name="large", end_step=1500, rows_low=512, rows_high=1024, cols_low=256, cols_high=512, batch_size=1),
    dict(name="larger", end_step=3000, rows_low=768, rows_high=1536, cols_low=384, cols_high=768, batch_size=1),
    dict(name="two_million", end_step=5000, rows_low=1024, rows_high=2000, cols_low=512, cols_high=1000, batch_size=1),
]
STREAM_OPTIONS = dict(attention_batch_size=32, cell_batch_size=8192, encoder_row_batch_size=32)
TRAIN_OPTIONS = {**STREAM_OPTIONS, "decoder_cell_batch_size": 2048}


def parse_curriculum(specifications):
    if not specifications:
        raise ValueError("A nonempty curriculum is required")
    phases = [base.CurriculumPhase(**specification) for specification in specifications]
    previous = 0
    names = set()
    for phase in phases:
        bounds = (phase.end_step, phase.rows_low, phase.rows_high, phase.cols_low, phase.cols_high, phase.batch_size)
        if any(type(value) is not int for value in bounds):
            raise ValueError("Curriculum bounds must be integers")
        if phase.end_step <= previous or phase.batch_size != 1 or not phase.name or phase.name in names:
            raise ValueError("Require distinct named increasing phases with one matrix per microstep")
        if not 2 <= phase.rows_low <= phase.rows_high or not 2 <= phase.cols_low <= phase.cols_high:
            raise ValueError("Invalid curriculum dimensions")
        previous = phase.end_step
        names.add(phase.name)
    return phases


def phase_at(phases, step):
    if step < 1 or step > phases[-1].end_step:
        raise ValueError("Step is outside the curriculum")
    return next(phase for phase in phases if step <= phase.end_step)


def learning_rate(step, total_steps, peak=5e-6, warmup=200, floor=.05):
    if not 1 <= step <= total_steps or not 0 <= warmup < total_steps:
        raise ValueError("Invalid learning-rate step or warmup")
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return peak * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress)))


class CurriculumSampler:
    def __init__(self, phases, *, data_seed=2150930, spectral_replay=.2):
        if not 0 <= spectral_replay <= 1:
            raise ValueError("Invalid spectral replay probability")
        self.args = control_recipe()
        self.phases = phases
        self.data_seed = data_seed
        self.spectral_replay = spectral_replay
        self.original_phase = base.parse_phases(self.args)[0]
        self.priors = {}

    def prior(self, phase):
        key = tuple(asdict(phase).values())
        if key not in self.priors:
            self.priors[key] = build_training_prior(self.args, phase=phase, device="cpu")
        return self.priors[key]

    def sample(self, step, microstep, *, device="cpu", forced_shape=None, component=None,
               seed=None, support_mode="broad_support", transpose=None):
        seed = self.data_seed + 16 * step + microstep if seed is None else seed
        phase = phase_at(self.phases, step)
        with preserved_rng(seed):
            replay = random.random() < self.spectral_replay
            chosen = self.original_phase if replay else phase
            dimensions = asdict(chosen)
            dimensions["batch_size"] = 1
            if forced_shape is not None:
                rows, columns = forced_shape
                dimensions.update(rows_low=rows, rows_high=rows, cols_low=columns, cols_high=columns)
                replay = False
            prior = self.prior(base.CurriculumPhase(**dimensions))
            index = random.choices(range(len(prior.children)), weights=prior.weights)[0] if component is None else component
            (source, target, _, _, _), _ = prior.children[index].get_batch()
            transposed = random.random() < .5 if transpose is None else transpose
            if transposed:
                source, target = source.transpose(1, 2).contiguous(), target.transpose(1, 2).contiguous()
        source, target, mask, stats = apply_shrunk_support_mask(
            source.float(), target.float(), support_mode, seed=seed + 10000000, variance_df=1.,
        )
        if not mask.any() or not torch.isfinite(target[mask]).all():
            raise ValueError("Nonempty finite synthetic targets are required")
        return source.to(device), target.to(device), mask.to(device), {
            **stats, "phase": phase.name, "prior": prior.names[index], "component": index,
            "spectral_replay": replay and index < 6, "transposed": transposed, "seed": seed,
            "rows": source.shape[1], "columns": source.shape[2], "targets": int(mask.sum()),
        }


def teacher_mean_grid(teachers, source, mask, options=STREAM_OPTIONS):
    if not mask.any() or not torch.isnan(source[mask]).all():
        raise ValueError("Teacher targets must refer only to missing inputs")
    means = []
    for model in teachers.models:
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Teachers must remain frozen in eval mode")
        with compute_context(source.device):
            predictions = streamed_point_predictions(model, source, teachers.criterion, **options)
        means.append(predictions[mask].clone())
        del predictions
    grid = torch.zeros_like(source)
    grid[mask] = equal_teacher_mean(means)
    return grid


def per_cell_objective(criterion, logits, truth, teacher):
    logits, truth = logits.float(), truth.float()
    nll = criterion(logits, truth)
    mean = criterion.mean(logits)
    point = functional.huber_loss(mean, teacher.detach().float(), delta=1., reduction="none")
    target_cdf = criterion.bucket_means.float().unsqueeze(0) >= truth.unsqueeze(1)
    cdf = (logits.softmax(-1).cumsum(-1) - target_cdf.float()).square().mean(-1)
    return nll + 2. * point + .05 * cdf


def backward_microstep(model, teachers, criterion, source, target, mask, *, accumulation=2, options=TRAIN_OPTIONS):
    teacher = teacher_mean_grid(teachers, source, mask, {key: options[key] for key in STREAM_OPTIONS})
    with compute_context(source.device):
        loss = streamed_training_loss(
            model, source, target, mask,
            lambda logits, truth, auxiliary: per_cell_objective(criterion, logits, truth, auxiliary),
            auxiliary_targets=teacher, **options,
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite streamed objective")
    (loss / accumulation).backward()
    return float(loss.detach())


def validation_specs(phases):
    specs = []
    for component in range(8):
        specs.append(dict(group="retention", component=component, shape=(240, 80), transpose=False))
    for shape in dict.fromkeys(((512, 256), (phases[-1].rows_high, phases[-1].cols_high))):
        for component in range(6):
            specs.append(dict(group="large", component=component, shape=shape, transpose=bool(component % 2)))
    return specs


@torch.no_grad()
def validate(model, criterion, sampler, *, validation_seed=9150930, options=STREAM_OPTIONS):
    was_training = model.training
    model.eval()
    cases = []
    try:
        for index, spec in enumerate(validation_specs(sampler.phases)):
            source, target, mask, stats = sampler.sample(
                1, 0, device=next(model.parameters()).device, forced_shape=spec["shape"],
                component=spec["component"], seed=validation_seed + index,
                support_mode="sparse_query" if spec["component"] % 2 else "mixed",
                transpose=spec["transpose"],
            )
            with compute_context(source.device):
                prediction = streamed_point_predictions(model, source, criterion, **options)
            error = prediction[mask].double() - target[mask].double()
            rmse = float(error.square().mean().sqrt())
            if not math.isfinite(rmse):
                raise FloatingPointError("Nonfinite fixed validation score")
            cases.append({**stats, "group": spec["group"], "rmse": rmse})
    finally:
        model.train(was_training)
    return {"cases": cases, **{
        group: sum(case["rmse"] for case in cases if case["group"] == group) /
        sum(case["group"] == group for case in cases) for group in ("retention", "large")
    }}


def selection_score(metrics, baseline):
    ratios = {group: metrics[group] / max(baseline[group], 1e-12) for group in ("retention", "large")}
    return .5 * (ratios["retention"] + ratios["large"]), ratios["retention"] <= 1.03


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
