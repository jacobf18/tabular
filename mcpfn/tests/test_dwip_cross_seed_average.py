import json

import pytest
import torch

import evaluate_dwip_cross_seed_average as runner
from evaluate_dwip_cross_seed_average import select_variant


def scores(seed1=1., seed2=.99, average=.98):
    return {"seed1": {"macro_rmse": seed1, "mteb": .1},
            "seed2": {"macro_rmse": seed2, "mteb": .2},
            "average": {"macro_rmse": average, "mteb": .99}}


def test_average_must_strictly_beat_both_replicas():
    assert select_variant(scores()) == "average"
    assert select_variant(scores(average=.995)) == "seed2"
    assert select_variant(scores(average=.99)) == "seed2"
    assert select_variant(scores(seed1=.97)) == "seed1"
    assert select_variant(scores(seed1=1., seed2=1., average=1.)) == "seed1"


def test_cross_seed_selection_rejects_incomplete_or_nonfinite_scores():
    with pytest.raises(ValueError, match="Nonfinite"):
        select_variant(scores(seed1=float("nan")))
    incomplete = scores()
    incomplete.pop("seed1")
    with pytest.raises(ValueError, match="both raw"):
        select_variant(incomplete)


def test_cross_seed_preparation_preserves_sources_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    queues = {"seed1": "first", "seed2": "second"}
    monkeypatch.setattr(runner, "QUEUES", queues)
    source_names = ["scripts/evaluate_dwip_checkpoint_average.py", "scripts/rowcol_fixed_validation.py",
                    "src/tabimpute/model/model_new_stable.py", "src/tabimpute/model/bar_distribution.py",
                    "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"]
    for name in source_names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixed-fixture")
    originals = {}
    for index, name in enumerate(queues.values()):
        root = tmp_path / "runs/objective_queues" / name
        work = tmp_path / "work" / f"tabimpute-{name}-dwip1e5"
        root.mkdir(parents=True)
        work.mkdir(parents=True)
        (root / "status.json").write_text(json.dumps({"state": "complete"}))
        (work / "config.json").write_text(json.dumps({"model_family": "new_stable", "model": {"embedding_size": 2}}))
        (work / "best_validation_metrics.json").write_text(json.dumps({"macro_rmse": 1.}))
        checkpoint = work / "best_validation_checkpoint.pth"
        torch.save({"model": {"weight": torch.tensor([float(index)])}}, checkpoint)
        originals[checkpoint] = checkpoint.read_bytes()
        if index == 0:
            (root / "validation.pth").write_bytes(b"fixed-selector")
            (root / "holdout.pth").write_bytes(b"fixed-holdout")
        else:
            (root / "validation.pth").symlink_to(root.parent / "first/validation.pth")
    output = tmp_path / "average"
    output.mkdir()
    (output / "prespecification.json").write_text("{}")
    prepared = runner.prepare(output)
    candidate = prepared[2]["average"]
    runner.prepare(output)
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    torch.save({"model": {"weight": torch.tensor([99.])}}, candidate)
    with pytest.raises(ValueError, match="integrity mismatch"):
        runner.prepare(output)
    (tmp_path / "runs/objective_queues/second/status.json").write_text(json.dumps({"state": "training"}))
    with pytest.raises(RuntimeError, match="training to finish"):
        runner.prepare(output)
