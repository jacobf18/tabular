"""Pure recipe, teacher-target and loss helpers for one unchanged DWIP student."""
import argparse
from contextlib import contextmanager
import importlib.resources
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

import train_direct_pure_rabel_curriculum as base
from general_spectral_geometry import install_geometry_generator
from rowcol_fixed_validation import preserved_rng
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK
from spectral_support_normalization import apply_shrunk_support_mask
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.prediction_distribution import prediction_distribution, distribution_target
from train_dwip_anchor_mix import ShapeAwareMixedMatrixPrior
from train_dwip_spectral_geometry import DwipTrainingAdapter


CONTROL = "dwip_spectral_anchor_mix25_huber1_64k_20260916"
DESIGN = REPO / "runs/objective_queues/dwip_top2_independent_complementarity_20260917"
CONTROL_COMMAND_SHA256 = "35d5694bcb73f9d3dafb20fe818c16fc3403e353b1c2caa8a9ebdbee6fddc82f"


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Registered input changed: {path}")


def control_recipe():
    path = REPO / "runs/objective_queues" / CONTROL / "dwip1e5_command.json"
    verify_hashes({path: CONTROL_COMMAND_SHA256})
    command = read_json(path)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--support-variance-df", type=float, required=True)
    extra, remainder = parser.parse_known_args(command[2:])
    original_argv = sys.argv
    try:
        sys.argv = [command[1], *remainder]
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    args.support_variance_df = extra.support_variance_df
    expected = {"precision_mode": "amp_bfloat16", "seed": 1201, "data_seed": 1150924,
                "prior_kind": "mixed_matrix", "support_mask_mode": "broad_support",
                "point_loss_kind": "huber", "point_loss_prediction": "mean", "point_loss_weight": 2.,
                "huber_delta": 1., "nll_loss_weight": 1., "cdf_loss_weight": .05,
                "distribution_loss_chunk_size": 2048, "support_variance_df": 1.}
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("The frozen control recipe no longer has the reviewed loss/data semantics")
    if json.loads(args.model_overrides_json) != dwip_spec()["model_config"]:
        raise ValueError("Require the unchanged public DWIP architecture")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def compute_context(device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        yield


def initialize_student(args, device):
    seed_all(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    student = DwipTrainingAdapter(**json.loads(args.model_overrides_json)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [{"params": list(student.parameters()), "lr": args.lr}], lr=args.lr,
        weight_decay=args.weight_decay, betas=(.9, .999), eps=1e-8, fused=device.type == "cuda",
    )
    scheduler = base.make_scheduler(
        optimizer, warmup_steps=args.warmup_steps, stable_steps=args.stable_steps,
        decay_steps=args.decay_steps, min_lr_ratio=args.min_lr_ratio,
    )
    payload = torch.load(args.init_model_checkpoint, map_location="cpu", weights_only=True)
    student.load_state_dict(payload["model"], strict=True)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in student.state_dict().values()):
        raise ValueError("Require finite FP32 initial model state")
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    criterion = FullSupportBarDistribution(torch.load(borders, map_location=device, weights_only=True))
    return student.train(), optimizer, scheduler, criterion


def build_training_prior(args, phase=None, phase_index=0, device=None):
    install_geometry_generator()
    phase = phase or base.parse_phases(args)[phase_index]
    with preserved_rng(args.data_seed + phase_index):
        return ShapeAwareMixedMatrixPrior(
            phase=phase, device=str(device or args.device), missingness_type=args.missingness_type,
            mixed_config=base.build_base_prior_config(args),
        )


def sample_training_batch(prior, args, step, device):
    with preserved_rng(args.data_seed + step):
        (source, target, _, _, _), _ = prior.get_batch()
    source, target = source.to(device=device, dtype=torch.float32), target.to(device=device, dtype=torch.float32)
    source, target, mask, stats = apply_shrunk_support_mask(
        source, target, args.support_mask_mode,
        seed=args.data_seed + 10000000 + step, variance_df=args.support_variance_df,
    )
    source = source.float()
    if not mask.any() or not torch.isfinite(target[mask]).all():
        raise ValueError("Require finite nonempty true missing-cell targets")
    return source, target, mask, {**stats, "sampled_prior_index": int(prior.last_prior_index),
                                  "sampled_prior_name": prior.last_prior_name}


def equal_teacher_mean(means):
    if len(means) != 2:
        raise ValueError("Require exactly the two frozen teachers")
    if any(mean.dtype != torch.float32 or mean.shape != means[0].shape or not mean.numel()
           or not torch.isfinite(mean).all() for mean in means):
        raise ValueError("Require aligned finite nonempty FP32 teacher means")
    if means[0].device != means[1].device:
        raise ValueError("Teacher predictions must share a device")
    return ((means[0].detach().double() + means[1].detach().double()) / 2).float()


class FrozenMeanTeachers:
    def __init__(self, models, criterion):
        if len(models) != 2 or models[0] is models[1]:
            raise ValueError("Require two distinct external teacher model instances")
        self.models = tuple(models)
        self.criterion = criterion
        for model in self.models:
            model.eval().requires_grad_(False)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise ValueError("Teacher parameters must remain FP32")

    @classmethod
    def load_registered(cls, args, criterion, device):
        proposal = read_json(DESIGN / "next_single_model_compression_proposal.json")
        models = []
        with preserved_rng():
            for member in proposal["teachers"]:
                path = WORK / f"tabimpute-{member['queue']}-dwip1e5/best_validation_checkpoint.pth"
                verify_hashes({path: member["sha256"]})
                model = DwipTrainingAdapter(**json.loads(args.model_overrides_json)).to(device=device, dtype=torch.float32)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
                models.append(model)
        return cls(models, criterion)

    def target(self, source, decode_mask):
        if source.ndim != 3 or decode_mask.dtype != torch.bool or decode_mask.shape != source.shape:
            raise ValueError("Require a matching boolean missing-cell decode mask")
        if not decode_mask.any() or not torch.isnan(source[decode_mask]).all():
            raise ValueError("Teachers may only target unobserved cells")
        if any(model.training or any(parameter.requires_grad for parameter in model.parameters()) for model in self.models):
            raise ValueError("Teachers must remain eval-only and frozen")
        means = []
        with preserved_rng(), torch.no_grad():
            for model in self.models:
                with compute_context(source.device):
                    logits = model(source, decode_mask=decode_mask)
                mean = prediction_distribution(model, self.criterion).mean(logits.float())
                if mean.shape != source[decode_mask].shape:
                    raise ValueError("Teacher output does not match the scored missing cells")
                means.append(mean)
        return equal_teacher_mean(means)


def objective_terms(model, criterion, logits, true_target, teacher_target=None):
    if true_target.ndim != 1 or logits.shape[0] != true_target.numel() or not true_target.numel():
        raise ValueError("Require aligned nonempty missing-cell logits and true targets")
    output = prediction_distribution(model, criterion)
    nll = output(logits.float(), true_target.float()).mean()
    point_mean = output.mean(logits.float())
    point_target = true_target.float() if teacher_target is None else teacher_target.detach()
    if point_target.shape != true_target.shape or point_target.dtype != torch.float32 or not torch.isfinite(point_target).all():
        raise ValueError("Invalid point target")
    point = torch.nn.functional.huber_loss(point_mean.float(), point_target, delta=1., reduction="mean")
    cdf = base.cdf_alignment_loss(logits, distribution_target(model, true_target), criterion, chunk_size=2048)
    loss = nll + 2. * point + .05 * cdf
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite teacher-mean training objective")
    return {"loss": loss, "nll": nll, "point": point, "cdf": cdf,
            "true_rmse": (point_mean.detach() - true_target.float()).square().mean().sqrt(),
            "teacher_rmse": (point_mean.detach() - point_target).square().mean().sqrt()}


def update_diagnostics(model, logits, terms, step, interval):
    if interval <= 0 or step % interval:
        return {}, {}
    stats = base.objective_logit_gradient_stats(logits, (("nll", terms["nll"], 1.), ("point", terms["point"], 2.)))
    samples = {name: parameter.detach().view(-1)[::max(1, parameter.numel() // 4096)][:4096].float().clone()
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    return stats, samples


def optimizer_step(model, optimizer, scheduler, terms, clip_norm):
    terms["loss"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return float(gradient_norm)
