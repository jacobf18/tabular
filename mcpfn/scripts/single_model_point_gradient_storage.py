"""Losslessly share every raw row with the full point-inclusive Adam proposal."""

from copy import copy

import torch

from single_model_composite_gradient_storage import byte_equal


POLICY = ("Preserve one complete CPU FP32 raw-gradient tensor and an exact full-row proposal alias. "
          "Never substitute a composite-only prefix, silently drop point rows, or change proposal bytes.")


def share(gradients, snapshot, composite_tasks):
    proposal = snapshot["task_gradients"]
    if (type(composite_tasks) is not int or composite_tasks <= 0
            or gradients.ndim != 2 or gradients.dtype != torch.float32
            or gradients.shape[0] <= composite_tasks or gradients.shape[1] == 0
            or proposal.device.type != "cpu" or proposal.dtype != torch.float32
            or proposal.shape != gradients.shape or not proposal.is_contiguous()):
        raise ValueError("Require all original composite/point rows and the complete CPU proposal")
    complete = gradients.detach().to(device="cpu").clone(memory_format=torch.contiguous_format)
    if not byte_equal(complete, proposal) or complete.stride() != proposal.stride():
        raise ValueError("Never substitute different bytes or layout for the full point-inclusive proposal")
    shared = copy(snapshot)
    shared["task_gradients"] = complete
    return complete, shared


def check_alias(payload):
    gradients, proposal = payload["gradients"], payload["snapshot"]["task_gradients"]
    record = payload["record"]
    composite_tasks, protected_tasks = record["composite_tasks"], record["protected_tasks"]
    if (type(composite_tasks) is not int or type(protected_tasks) is not int
            or not 0 < composite_tasks < protected_tasks or gradients.ndim != 2
            or gradients.shape[0] != protected_tasks or gradients.shape[1] == 0
            or gradients.dtype != torch.float32 or gradients.device.type != "cpu"
            or not gradients.is_contiguous() or proposal.device.type != "cpu"
            or proposal.dtype != torch.float32 or proposal.shape != gradients.shape
            or proposal.stride() != gradients.stride() or proposal.data_ptr() != gradients.data_ptr()
            or proposal.untyped_storage().data_ptr() != gradients.untyped_storage().data_ptr()
            or not byte_equal(proposal, gradients)):
        raise ValueError("Retain every raw gradient and its exact full point-inclusive proposal alias")
