import pytest
import torch

import single_model_four_draw_gradient_staging as staging
from test_single_model_four_draw_protection import metrics, tasks


def fixture(undefined=False):
    before = metrics()
    if undefined:
        before["legacy_draw_3/common_nrmse"] = None
    labels, names, matrix = staging.protection.gradient_layout(before, 56)
    entries = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    bank = torch.arange(168 * 7, dtype=torch.float32).reshape(168, 7)
    bank[0, 0], bank[0, 1] = -0., torch.finfo(torch.float32).tiny
    if undefined:
        bank[-1] = float("nan")
    return bank, entries, tasks(), before


@pytest.mark.parametrize("undefined", [False, True])
def test_exact_transfer_includes_all_defined_bits_and_excludes_only_unused_capacity(undefined):
    bank, entries, batch, before = fixture(undefined)
    result, report = staging.stage_for_update(bank, entries, batch, before, "cpu", chunk_elements=11)
    assert result.data_ptr() != bank.data_ptr()
    assert torch.equal(result.view(torch.uint8), bank[:len(entries)].view(torch.uint8))
    assert report["defined_rows"] == (167 if undefined else 168)
    assert report["verified_elements"] == result.numel()
    assert report["unused_capacity_rows"] == int(undefined)
    assert report["capture_bytes"] == 168 * 7 * 4
    assert report["transferred_bytes"] == len(entries) * 7 * 4
    assert report["exact_fp32_transfer"] and not report["scientific_feasibility_established"]
    assert torch.signbit(result[0, 0])


@pytest.mark.parametrize("mutation", ["short_bank", "double", "noncontiguous", "nonfinite", "grad", "layout", "chunk", "meta"])
def test_incomplete_lossy_or_unverifiable_transfer_is_rejected(mutation):
    bank, entries, batch, before = fixture()
    device, chunk_elements = "cpu", 17
    if mutation == "short_bank":
        bank = bank[:84]
    elif mutation == "double":
        bank = bank.double()
    elif mutation == "noncontiguous":
        bank = bank[:, ::2]
    elif mutation == "nonfinite":
        bank[80, 1] = float("nan")
    elif mutation == "grad":
        bank.requires_grad_()
    elif mutation == "layout":
        entries.pop()
    elif mutation == "chunk":
        chunk_elements = True
    elif mutation == "meta":
        device = "meta"
    with pytest.raises(ValueError):
        staging.stage_for_update(bank, entries, batch, before, device, chunk_elements=chunk_elements)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU suite does not claim CUDA transfer feasibility")
def test_small_deterministic_cuda_transfer_preserves_bits_without_scientific_sampling():
    bank, entries, batch, before = fixture(True)
    result, report = staging.stage_for_update(bank, entries, batch, before, "cuda", chunk_elements=31)
    assert torch.equal(result.cpu().view(torch.uint8), bank[:len(entries)].view(torch.uint8))
    assert report["destination_device"].startswith("cuda") and report["exact_fp32_transfer"]
    assert not report["scientific_feasibility_established"]


def test_byte_verification_detects_even_a_signed_zero_change(monkeypatch):
    bank, entries, batch, before = fixture()
    original = torch.Tensor.to

    def corrupt(tensor, *arguments, **keywords):
        result = original(tensor, *arguments, **keywords)
        if tensor.shape == bank.shape and keywords.get("copy"):
            result[0, 0] = 0.
        return result

    monkeypatch.setattr(torch.Tensor, "to", corrupt)
    with pytest.raises(ValueError, match="changed transferred FP32"):
        staging.stage_for_update(bank, entries, batch, before, "cpu", chunk_elements=11)
