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


class SharedGradNormWeighterEMA:
    """
    GradNorm (Chen et al., ICML 2018) for shared-parameter networks
    with EMA smoothing on task losses for stability.

    Works even when all tasks share exactly the same network.
    """

    def __init__(self, num_tasks, alpha=0.5, lr=0.025, beta=0.9, device="cuda"):
        """
        Args:
            num_tasks (int): number of tasks
            alpha (float): restoring force (0.3–1.0 typical)
            lr (float): learning rate for loss weight updates
            beta (float): EMA smoothing coefficient for task losses
            device (str): device for internal tensors
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.lr = lr
        self.beta = beta
        self.device = device

        # Per-task adaptive weights
        self.weights = torch.ones(num_tasks, device=device, requires_grad=True)

        # EMA of each task's loss
        self.loss_ema = torch.ones(num_tasks, device=device)

    @torch.no_grad()
    def normalize_weights(self):
        """Ensure weights remain positive and normalized (sum = num_tasks)."""
        self.weights.data.clamp_(min=1e-6)
        self.weights.data[-1] *= 0.1
        weight_sum = self.weights.data.sum()
        if torch.isnan(weight_sum) or weight_sum <= 0:
            print(f"WARNING: Invalid weight_sum in normalize_weights: {weight_sum}, resetting weights")
            self.weights.data = torch.ones_like(self.weights.data)
        else:
            self.weights.data *= self.num_tasks / weight_sum

    def update(self, losses, shared_params):
        """
        Update task weights using smoothed loss baselines.

        Args:
            losses: list of scalar loss tensors for each task
            shared_params: iterable of shared model parameters

        Returns:
            grad_norms (tensor): per-task gradient norms
            gradnorm_loss (float): auxiliary GradNorm balancing loss
        """
        assert len(losses) == self.num_tasks

        # --- Check for NaN in losses before processing
        raw_losses = torch.stack([L.detach() for L in losses])
        if torch.isnan(raw_losses).any():
            print(f"WARNING: NaN detected in task losses: {raw_losses}")
            # Replace NaN losses with previous EMA values to prevent propagation
            raw_losses = torch.where(torch.isnan(raw_losses), self.loss_ema, raw_losses)

        # --- Update EMA of losses (smooth the moving average)
        self.loss_ema = self.beta * self.loss_ema + (1 - self.beta) * raw_losses
        
        # Check for NaN in EMA
        if torch.isnan(self.loss_ema).any():
            print(f"WARNING: NaN detected in loss_ema: {self.loss_ema}")
            # Reset EMA to ones if it becomes NaN
            self.loss_ema = torch.ones_like(self.loss_ema)

        # --- Check weights before computing gradients
        if torch.isnan(self.weights).any():
            print(f"WARNING: NaN detected in weights before gradient computation: {self.weights}")
            # Reset weights to ones if they contain NaN
            self.weights.data = torch.ones_like(self.weights.data)
        
        # --- Compute gradient norms per task
        grad_norms = []
        for i, L in enumerate(losses):
            # Check if weight is NaN before using it
            if torch.isnan(self.weights[i]):
                print(f"WARNING: NaN weight detected for task {i}, using 1.0 instead")
                weight_val = torch.tensor(1.0, device=self.device)
            else:
                weight_val = self.weights[i]
            
            grads = torch.autograd.grad(
                L * weight_val,
                shared_params,
                retain_graph=True,
                create_graph=False
            )
            # Check each gradient for NaN before computing norm
            grad_norms_list = []
            for g in grads:
                if g is not None:
                    g_norm = g.norm()
                    if torch.isnan(g_norm):
                        print(f"WARNING: NaN detected in gradient norm for task {i}")
                        # Use a small positive value instead of NaN
                        grad_norms_list.append(torch.tensor(1e-6, device=g.device, dtype=g.dtype))
                    else:
                        grad_norms_list.append(g_norm)
            
            if len(grad_norms_list) == 0:
                # If no valid gradients, use a small default value
                total_norm = torch.tensor(1e-6, device=self.device)
            else:
                total_norm = torch.norm(torch.stack(grad_norms_list))
                if torch.isnan(total_norm):
                    print(f"WARNING: NaN detected in total gradient norm for task {i}")
                    total_norm = torch.tensor(1e-6, device=self.device)
            grad_norms.append(total_norm)

        grad_norms = torch.stack(grad_norms)
        
        # Check for NaN in grad_norms
        if torch.isnan(grad_norms).any():
            print(f"WARNING: NaN detected in grad_norms: {grad_norms}")
            # Replace NaN with small positive values
            grad_norms = torch.where(torch.isnan(grad_norms), torch.tensor(1e-6, device=self.device), grad_norms)
        
        mean_grad_norm = grad_norms.mean()
        
        # Check for NaN or zero mean_grad_norm
        if torch.isnan(mean_grad_norm) or mean_grad_norm <= 0:
            print(f"WARNING: Invalid mean_grad_norm: {mean_grad_norm}, using default")
            mean_grad_norm = torch.tensor(1e-6, device=self.device)

        with torch.no_grad():
            # --- Target gradient norms using EMA-smoothed losses
            # Check for NaN in loss_ema before computing ratios
            if torch.isnan(self.loss_ema).any():
                print(f"WARNING: NaN detected in loss_ema before computing loss_ratios: {self.loss_ema}")
                # Replace NaN with small positive values
                self.loss_ema = torch.where(torch.isnan(self.loss_ema), torch.tensor(1e-6, device=self.device), self.loss_ema)
            
            # Use absolute values for ratio computation to handle negative losses
            # GradNorm balances based on loss magnitudes, not signs
            # loss_ema_abs = torch.abs(self.loss_ema)
            loss_ema_abs = self.loss_ema + 4.0
            
            L_mean_abs = loss_ema_abs.mean()
            
            if torch.isnan(L_mean_abs) or L_mean_abs <= 0:
                print(f"WARNING: Invalid L_mean_abs: {L_mean_abs}, using default")
                L_mean_abs = torch.tensor(1e-6, device=self.device)
            
            # Compute ratios using absolute values - ensures positive ratios for fractional power
            ratios = loss_ema_abs / (L_mean_abs + 1e-8)
            # Ensure ratios are positive (should be, but add safeguard)
            ratios = torch.clamp(ratios, min=1e-8)
            loss_ratios = ratios ** self.alpha
            
            if torch.isnan(loss_ratios).any():
                print(f"WARNING: NaN detected in loss_ratios: {loss_ratios}")
                print(f"  loss_ema: {self.loss_ema}")
                print(f"  loss_ema_abs: {loss_ema_abs}")
                print(f"  L_mean_abs: {L_mean_abs}")
                print(f"  ratios: {ratios}")
                print(f"  alpha: {self.alpha}")
                # Replace NaN with ones (equal weighting)
                loss_ratios = torch.where(torch.isnan(loss_ratios), torch.ones_like(loss_ratios), loss_ratios)
            
            target_grad_norms = mean_grad_norm * loss_ratios
            if torch.isnan(target_grad_norms).any():
                print(f"WARNING: NaN detected in target_grad_norms: {target_grad_norms}")
                target_grad_norms = grad_norms.clone()  # Fallback to current grad_norms

            # --- Compute GradNorm auxiliary loss
            gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms)).detach()
            if torch.isnan(gradnorm_loss):
                gradnorm_loss = torch.tensor(0.0, device=self.device)

            # --- Manual update step on task weights
            diff = grad_norms - target_grad_norms
            if torch.isnan(diff).any():
                print(f"WARNING: NaN detected in diff: {diff}")
                # Skip weight update if diff contains NaN
                print(f"Skipping weight update due to NaN in diff")
            else:
                # Check weights before update
                if torch.isnan(self.weights).any():
                    print(f"WARNING: NaN detected in weights before update: {self.weights}")
                    # Reset weights to ones if they contain NaN
                    self.weights.data = torch.ones_like(self.weights.data)
                
                self.weights -= self.lr * diff / (mean_grad_norm + 1e-8)
                
                # Check weights after update
                if torch.isnan(self.weights).any():
                    print(f"WARNING: NaN detected in weights after update: {self.weights}")
                    print(f"  diff: {diff}")
                    print(f"  mean_grad_norm: {mean_grad_norm}")
                    print(f"  lr: {self.lr}")
                    # Reset weights to ones if they contain NaN
                    self.weights.data = torch.ones_like(self.weights.data)

            # normalize and return diagnostics
            self.normalize_weights()
            
            # Final check after normalization
            if torch.isnan(self.weights).any():
                print(f"WARNING: NaN detected in weights after normalization: {self.weights}")
                # Reset weights to ones as last resort
                self.weights.data = torch.ones_like(self.weights.data)
                
        return grad_norms.detach(), gradnorm_loss.item()

    def weighted_total_loss(self, losses):
        """Return the weighted total loss for the main backward pass."""
        return sum(self.weights[i] * losses[i] for i in range(self.num_tasks))
