"""Share full protected gradients with an unchanged composite-only proposal prefix."""

from copy import copy

import torch


POLICY = ("Preserve all protected raw FP32 gradient rows in one CPU tensor. The composite-only Adam proposal "
          "retains its original shorter prefix view, with identical shape, stride and bytes. This corrects "
          "serialization only; never add point gradients to the actual composite Adam proposal.")


def byte_equal(first, second):
    return (first.shape == second.shape and first.dtype == second.dtype
            and torch.equal(first.contiguous().reshape(-1).view(torch.uint8),
                            second.contiguous().reshape(-1).view(torch.uint8)))


def share(gradients, snapshot, composite_tasks):
    proposal = snapshot["task_gradients"]
    if (type(composite_tasks) is not int or composite_tasks <= 0
            or gradients.ndim != 2 or gradients.dtype != torch.float32
            or gradients.shape[0] <= composite_tasks or gradients.shape[1] == 0
            or proposal.device.type != "cpu" or proposal.dtype != torch.float32
            or proposal.shape != (composite_tasks, gradients.shape[1]) or not proposal.is_contiguous()):
        raise ValueError("Require every protected row and the original shorter CPU composite prefix")
    complete = gradients.detach().to(device="cpu").clone(memory_format=torch.contiguous_format)
    prefix = complete[:composite_tasks]
    if not byte_equal(prefix, proposal) or prefix.stride() != proposal.stride():
        raise ValueError("Never substitute different bytes or layout for the original composite proposal")
    shared = copy(snapshot)
    shared["task_gradients"] = prefix
    return complete, shared


def check_alias(payload):
    gradients = payload["gradients"]
    proposal = payload["snapshot"]["task_gradients"]
    record = payload["record"]
    composite_tasks, protected_tasks = record["composite_tasks"], record["protected_tasks"]
    if (type(composite_tasks) is not int or type(protected_tasks) is not int
            or not 0 < composite_tasks < protected_tasks or gradients.ndim != 2
            or gradients.shape[0] != protected_tasks or gradients.dtype != torch.float32
            or gradients.device.type != "cpu" or not gradients.is_contiguous()
            or proposal.device.type != "cpu" or proposal.dtype != torch.float32
            or proposal.shape != (composite_tasks, gradients.shape[1])
            or proposal.stride() != gradients.stride() or proposal.data_ptr() != gradients.data_ptr()
            or proposal.untyped_storage().data_ptr() != gradients.untyped_storage().data_ptr()
            or not byte_equal(proposal, gradients[:composite_tasks])):
        raise ValueError("Retain one complete protected-gradient tensor and its exact shorter proposal alias")
