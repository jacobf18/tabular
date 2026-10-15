"""Joint synthetic ranking slates across independently encoded item windows."""
from contextlib import nullcontext

import torch

from dwip_recsys_experiment import training_logits


def windowed_training_logits(model, source, query, activation_checkpointing=False, window_cols=0):
    if window_cols < 0 or source.ndim != 3 or query.shape != source.shape or query.dtype != torch.bool:
        raise ValueError("Require matching three-dimensional context/query tensors and nonnegative window width")
    if not query.any():
        raise ValueError("Require at least one synthetic query")
    if window_cols == 0 or window_cols >= source.shape[-1]:
        return training_logits(model, source, query, activation_checkpointing)
    positions = torch.full_like(query, -1, dtype=torch.long)
    positions[query] = torch.arange(int(query.sum()), device=query.device)
    outputs = []
    indices = []
    for start in range(0, source.shape[-1], window_cols):
        local_query = query[:, :, start:start + window_cols].contiguous()
        if not local_query.any():
            continue
        local_source = source[:, :, start:start + window_cols].contiguous()
        logits = training_logits(model, local_source, local_query, activation_checkpointing)
        if logits.shape[0] != int(local_query.sum()):
            raise ValueError("Window logits do not match their synthetic queries")
        outputs.append(logits)
        indices.append(positions[:, :, start:start + window_cols][local_query])
    ordering = torch.cat(indices).argsort()
    return torch.cat(outputs)[ordering]


class WindowGradientReplay:
    def __init__(self, model, source, query, entries, activation_checkpointing, shape, offload_activations=False):
        self.model = model
        self.source = source
        self.query = query
        self.entries = entries
        self.activation_checkpointing = activation_checkpointing
        self.shape = shape
        self.offload_activations = offload_activations
        self.training = model.training
        self.consumed = False

    def backward(self, gradient, context_factory=nullcontext):
        if self.consumed:
            raise ValueError("Window gradients have already been replayed")
        if gradient is None or gradient.shape != self.shape or gradient.device != self.source.device or gradient.dtype != self.entries[0][3].dtype or not torch.isfinite(gradient).all():
            raise ValueError("Require finite gradients aligned to the joint query logits")
        if self.model.training != self.training:
            raise ValueError("Model training mode changed before window replay")
        self.consumed = True
        devices = [self.source.device.index] if self.source.is_cuda else []
        for start, stop, indices, expected, cpu_state, cuda_state in self.entries:
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, self.source.device)
                saved_tensors = torch.autograd.graph.save_on_cpu(pin_memory=self.source.is_cuda) if self.offload_activations else nullcontext()
                with saved_tensors, context_factory():
                    logits = training_logits(self.model, self.source[:, :, start:stop].contiguous(),
                                             self.query[:, :, start:stop].contiguous(), self.activation_checkpointing)
                if logits.dtype != expected.dtype or not torch.equal(logits.detach(), expected.to(logits.device)):
                    raise ValueError("Recomputed window logits differ; do not apply an inconsistent gradient")
                logits.backward(gradient[indices])
                del logits
        return len(self.entries)


def recomputed_window_logits(model, source, query, activation_checkpointing=False, window_cols=0, offload_activations=False):
    """Detach one graph at a time, then replay its joint-loss gradient with identical RNG."""
    if not torch.is_grad_enabled():
        raise ValueError("Window replay requires an enabled training gradient context")
    if window_cols < 0 or source.ndim != 3 or query.shape != source.shape or query.dtype != torch.bool:
        raise ValueError("Require matching three-dimensional context/query tensors and nonnegative window width")
    if not query.any():
        raise ValueError("Require at least one synthetic query")
    if any(module.training and getattr(module, "track_running_stats", False) for module in model.modules()):
        raise ValueError("Window replay does not support mutable running-stat modules")
    width = window_cols or source.shape[-1]
    positions = torch.full_like(query, -1, dtype=torch.long)
    positions[query] = torch.arange(int(query.sum()), device=query.device)
    entries = []
    for start in range(0, source.shape[-1], width):
        stop = min(start + width, source.shape[-1])
        local_query = query[:, :, start:stop].contiguous()
        if not local_query.any():
            continue
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(source.device) if source.is_cuda else None
        logits = training_logits(model, source[:, :, start:stop].contiguous(), local_query, activation_checkpointing).detach()
        if logits.shape[0] != int(local_query.sum()):
            raise ValueError("Window logits do not match their synthetic queries")
        indices = positions[:, :, start:stop][local_query]
        if offload_activations:
            logits = logits.cpu()
        entries.append((start, stop, indices, logits, cpu_state, cuda_state))
    ordering = torch.cat([entry[2] for entry in entries]).argsort()
    logits = torch.cat([entry[3].to(source.device) for entry in entries])[ordering].detach().requires_grad_(True)
    return logits, WindowGradientReplay(model, source, query, entries, activation_checkpointing, logits.shape, offload_activations)
