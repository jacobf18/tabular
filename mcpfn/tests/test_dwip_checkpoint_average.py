from copy import deepcopy
import json

import pytest
import torch

from evaluate_dwip_checkpoint_average import SOURCE_STEPS, average_states, choose_average, prepare


def state(value):
    return {"weight": torch.tensor([value, value + 1], dtype=torch.float32), "count": torch.tensor(3)}


def test_average_is_fp32_and_does_not_mutate_sources():
    states = [state(1), state(3), state(5)]
    before = deepcopy(states)
    averaged = average_states(states)
    torch.testing.assert_close(averaged["weight"], torch.tensor([3., 4.]), rtol=0, atol=0)
    assert averaged["weight"].dtype == torch.float32
    assert averaged["count"].dtype == torch.int64
    for original, unchanged in zip(states, before):
        for key in original:
            torch.testing.assert_close(original[key], unchanged[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["keys", "shape", "dtype", "nonfinite", "buffer"])
def test_average_rejects_incompatible_states(kind):
    states = [state(1), state(3)]
    if kind == "keys":
        states[1].pop("count")
    elif kind == "shape":
        states[1]["weight"] = torch.ones(3)
    elif kind == "dtype":
        states[1]["weight"] = states[1]["weight"].double()
    elif kind == "nonfinite":
        states[1]["weight"][0] = float("nan")
    else:
        states[1]["count"] = torch.tensor(4)
    with pytest.raises(ValueError):
        average_states(states)


def test_average_refuses_bfloat16_model_weights():
    source = state(1)
    source["weight"] = source["weight"].bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        average_states([source])


def test_selection_ignores_benchmark_scores_and_keeps_raw_on_tie():
    raw = {"macro_rmse": 1., "mteb": .1}
    assert not choose_average(raw, {"macro_rmse": 1., "mteb": .99})
    assert not choose_average(raw, {"macro_rmse": 1.1, "mteb": .99})
    assert choose_average(raw, {"macro_rmse": .9, "mteb": 0.})
    with pytest.raises(ValueError, match="Nonfinite"):
        choose_average(raw, {"macro_rmse": float("nan")})


def test_prepared_average_detects_tampering_and_preserves_source_files(tmp_path):
    root, source, work = (tmp_path / name for name in ("output", "source", "work"))
    for path in (root, source, work):
        path.mkdir()
    (work / "config.json").write_text(json.dumps({"model_family": "new_stable", "model": {"embedding_size": 2}}))
    (source / "validation.pth").write_bytes(b"fixed-cache-fixture")
    for index, step in enumerate(SOURCE_STEPS):
        torch.save({"model": state(index)}, work / f"checkpoint_{step}.pth")
    torch.save({"model": state(1)}, work / "best_validation_checkpoint.pth")
    originals = {path.name: path.read_bytes() for path in work.glob("*.pth")}
    checkpoint, _, _ = prepare(root, source, work)
    prepare(root, source, work)
    torch.save({"model": state(99)}, checkpoint)
    with pytest.raises(ValueError, match="integrity mismatch"):
        prepare(root, source, work)
    assert {path.name: path.read_bytes() for path in work.glob("*.pth")} == originals
