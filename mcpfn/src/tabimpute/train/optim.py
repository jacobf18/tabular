"""Learning rate scheduler."""

from __future__ import annotations

from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


import math
from functools import partial
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_with_restarts_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    amplitude_decay: float,
    lr_end: float = 0.0,
    lr_init: float = 1.0,
):
    """
    Compute the learning rate factor for a cosine schedule with warmup, hard restarts, and amplitude scaling.
    """
    if current_step < num_warmup_steps:
        # Warmup phase: Linearly increase learning rate
        return float(current_step) / float(max(1, num_warmup_steps))

    # After warmup: Apply cosine schedule with hard restarts and amplitude scaling
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    if progress >= 1.0:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init

    # Determine which cycle the current step is in
    cycle_progress = (float(num_cycles) * progress) % 1.0
    current_cycle = int(float(num_cycles) * progress)
    amplitude = (
        amplitude_decay**current_cycle
    )  # Exponentially decay amplitude per cycle

    # Calculate the current learning rate with proper scaling
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    current_lr = lr_end + (lr_init - lr_end) * cosine_factor * amplitude
    return current_lr / lr_init  # as LambdaLR multiplies by lr_init


def get_cosine_with_restarts(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    amplitude_decay: float = 1.0,
    lr_end: float = 0.0,
    last_epoch: int = -1,
):
    """
    Create a learning rate scheduler with warmup, cosine decay, hard restarts, and amplitude scaling.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        num_cycles (int, optional): Number of hard restarts. Defaults to 1.
        amplitude_decay (float, optional): Factor to exponentially decay the max LR per cycle. Defaults to 1.0.
        lr_end (float, optional): Minimum learning rate at the end of each cycle. Defaults to 0.0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.

    Returns:
        LambdaLR: A learning rate scheduler.
    """
    lr_init = optimizer.defaults["lr"]
    if lr_end > lr_init:
        raise ValueError(
            f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})"
        )

    lr_lambda = partial(
        _get_cosine_with_restarts_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        amplitude_decay=amplitude_decay,
        lr_end=lr_end,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(config, optimizer):
    """Get the learning rate scheduler based on configuration."""

    if config.warmup_proportion >= 0:
        warmup_steps = config.max_steps * config.warmup_proportion
    else:
        warmup_steps = config.warmup_steps

    if config.scheduler == "constant":
        scheduler = get_constant_schedule(optimizer=optimizer)
    elif config.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
        )
    elif config.scheduler == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
        )
    elif config.scheduler == "cosine_with_restarts":
        scheduler = get_cosine_with_restarts(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            num_cycles=config.cosine_num_cycles,
            amplitude_decay=config.cosine_amplitude_decay,
            lr_end=config.cosine_lr_end,
        )
    elif config.scheduler == "polynomial_decay_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            lr_end=config.poly_decay_lr_end,
            power=config.poly_decay_power,
        )
    else:
        raise NotImplementedError

    return scheduler


class GradNormMultiTask:
    """
    Implements GradNorm (Chen et al., ICML 2018) for multi-task loss balancing,
    with optional EMA-smoothed loss tracking for stability.

    Usage:
        gradnorm = GradNormMultiTask(num_tasks=3, alpha=0.5, ema_beta=0.9, device='cuda')
        ...
        gradnorm.update_weights(losses, shared_params)
        total_loss = sum(gradnorm.weights[i] * losses[i] for i in range(3))
    """
    def __init__(self, num_tasks, alpha=0.5, ema_beta=0.9, device='cuda', lr=0.025):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        self.ema_beta = ema_beta

        # task weights
        self.weights = torch.ones(num_tasks, requires_grad=True, device=device)
        self.opt = torch.optim.Adam([self.weights], lr=lr)

        # initialize running loss EMA
        self.loss_ema = torch.zeros(num_tasks, device=device)

    @torch.no_grad()
    def normalize_weights(self):
        """Keep weights positive and normalized (sum = num_tasks)."""
        self.weights.data = torch.clamp(self.weights.data, min=1e-8)
        self.weights.data = self.weights.data * self.num_tasks / self.weights.data.sum()

    def update_weights(self, losses, shared_params):
        """
        Compute GradNorm loss and update task weights.
        Arguments:
            losses: list of per-task scalar losses (torch.Tensors)
            shared_params: iterable of shared model parameters (e.g., backbone)
        Returns:
            grad_norms (tensor): gradient norms per task
            gradnorm_loss (float): GradNorm auxiliary loss value
        """
        assert len(losses) == self.num_tasks

        # ---- Step 1: update EMA of losses ----
        with torch.no_grad():
            curr_losses = torch.tensor([l.item() for l in losses], device=self.device)
            self.loss_ema = self.ema_beta * self.loss_ema + (1 - self.ema_beta) * curr_losses

        # ---- Step 2: compute gradient norms for each task ----
        grad_norms = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(self.weights[i] * loss, shared_params,
                                        retain_graph=True, create_graph=False)
            total_norm = torch.norm(torch.stack([g.norm() for g in grads if g is not None]))
            grad_norms.append(total_norm.detach())

        grad_norms = torch.stack(grad_norms)
        mean_grad_norm = grad_norms.mean()

        # ---- Step 3: compute target relative training rates ----
        with torch.no_grad():
            L_mean = self.loss_ema.mean()
            loss_ratios = (self.loss_ema / (L_mean + 1e-8)) ** self.alpha

        # ---- Step 4: GradNorm objective ----
        gradnorm_loss = torch.sum(torch.abs(grad_norms - mean_grad_norm * loss_ratios))

        # ---- Step 5: backprop through weights ----
        self.opt.zero_grad()
        gradnorm_loss.backward()
        self.opt.step()
        self.normalize_weights()

        return grad_norms, gradnorm_loss.item()