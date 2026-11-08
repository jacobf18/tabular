"""Stage complete FP32 gradients on CPU during capture, without changing arithmetic."""

import torch

import single_model_four_draw_protection as protection


CAPACITY = 168
CHUNK_ELEMENTS = 2**20
POLICY = (
    "Prospective memory integration for56-task batches. Capture each original raw gradient into "
    "one complete contiguous CPU FP32 bank, retaining all168 possible rows. After all capture "
    "and task/layout checks, copy only the complete defined prefix to the explicitly requested "
    "update device. Verify every transferred FP32 byte in bounded chunks, including signed zero; "
    "never quantize, compress, recompute, aggregate or drop a defined row. Unused undefined-slot "
    "capacity is not a measured gradient and is not transferred. The scientific caller must "
    "release the CPU capture bank before optimizer work and use CUDA for the unchanged original "
    "Adam/projection/quarter arithmetic. This reduces persistent capture-time GPU residency, "
    "not the total recorded gradient count. CPU memory, update-time GPU memory, serialization "
    "and complete disk budgets still require independent admission and a fresh measured probe. "
    "No allocator-cap increase, scientific retry, seed authorization or model qualification."
)


def stage_for_update(captured, entries, tasks, before, device, *, chunk_elements=CHUNK_ELEMENTS):
    protection.validate_layout(entries, tasks, before)
    destination = torch.device(device)
    if (captured.ndim != 2 or captured.shape[0] != CAPACITY or not captured.shape[1]
            or captured.device.type != "cpu" or captured.dtype != torch.float32
            or not captured.is_contiguous() or captured.requires_grad
            or type(chunk_elements) is not int or chunk_elements < 1
            or destination.type not in {"cpu", "cuda"}):
        raise ValueError("Require a complete CPU FP32 capture bank and explicit CPU-test or CUDA-update device")
    source = captured[:len(entries)].reshape(-1)
    for start in range(0, source.numel(), chunk_elements):
        if not torch.isfinite(source[start:start + chunk_elements]).all():
            raise ValueError("Every defined captured gradient must be finite before transfer")
    transferred = captured[:len(entries)].to(device=destination, copy=True)
    target = transferred.reshape(-1)
    for start in range(0, source.numel(), chunk_elements):
        original = source[start:start + chunk_elements].view(torch.uint8)
        replay = target[start:start + chunk_elements].to(device="cpu").view(torch.uint8)
        if not torch.equal(original, replay):
            raise ValueError("Never proceed with a changed transferred FP32 gradient byte")
    return transferred, {"policy": POLICY, "capacity_rows": CAPACITY, "defined_rows": len(entries),
        "parameters": captured.shape[1], "capture_bytes": captured.numel() * captured.element_size(),
        "transferred_bytes": transferred.numel() * transferred.element_size(), "source_device": "cpu",
        "destination_device": str(transferred.device), "verified_elements": source.numel(),
        "exact_fp32_transfer": True, "unused_capacity_rows": CAPACITY - len(entries),
        "peak_memory_measured": False, "scientific_feasibility_established": False}
