#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModel as StableTabImputeModel
from tabimpute.model.model_new_stable_memory_efficient import (
    TabImputeModel as MemoryEfficientTabImputeModel,
)


def parse_size(raw: str) -> tuple[int, int]:
    if "x" not in raw.lower():
        raise argparse.ArgumentTypeError(f"Expected ROWSxCOLS, got {raw}")
    rows_raw, cols_raw = raw.lower().split("x", 1)
    return int(rows_raw), int(cols_raw)


def mib(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def load_model_config(run_dir: Path) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return dict(cfg["model"])


def build_model(kind: str, cfg: dict, device: str):
    cls = {
        "stable": StableTabImputeModel,
        "memory_efficient": MemoryEfficientTabImputeModel,
    }[kind]
    return cls(**cfg).to(device=device, dtype=torch.bfloat16).train()


def run_trial(
    *,
    model_kind: str,
    cfg: dict,
    criterion: FullSupportBarDistribution,
    batch_size: int,
    rows: int,
    cols: int,
    missing_rate: float,
    decode_only_missing: bool,
    device: str,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(model_kind, cfg, device)
    x = torch.randn(batch_size, rows, cols, device=device, dtype=torch.bfloat16)
    y = x.clone()
    mask = torch.rand(batch_size, rows, cols, device=device) < missing_rate
    x[mask] = float("nan")

    if decode_only_missing:
        output = model(x, decode_mask=mask)
        loss = criterion(output, y[mask]).mean()
    else:
        output = model(x)
        losses = criterion(output, y)
        loss = losses[mask].mean()

    loss.backward()
    torch.cuda.synchronize()

    result = {
        "model": model_kind,
        "decode_only_missing": decode_only_missing,
        "batch_size": batch_size,
        "rows": rows,
        "cols": cols,
        "missing": int(mask.sum().item()),
        "output_shape": tuple(output.shape),
        "final_mib": round(mib(torch.cuda.memory_allocated()), 1),
        "peak_mib": round(mib(torch.cuda.max_memory_allocated()), 1),
    }

    del model, x, y, mask, output, loss
    if not decode_only_missing:
        del losses
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/root/tabular/mcpfn/src/tabimpute/workdir/"
            "tabimpute-new-stable_v2-rank_10_10-50x50"
        ),
    )
    parser.add_argument("--size", type=parse_size, default=parse_size("300x300"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["stable", "memory_efficient"],
        default=["stable", "memory_efficient"],
    )
    parser.add_argument(
        "--decode-modes",
        nargs="+",
        choices=["full", "missing"],
        default=["missing"],
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows, cols = args.size
    cfg = load_model_config(args.run_dir)
    borders = torch.load(
        Path("/root/tabular/mcpfn/src/tabimpute/data/borders.pt"),
        map_location=args.device,
    )
    criterion = FullSupportBarDistribution(borders=borders)

    for model_kind in args.models:
        for decode_mode in args.decode_modes:
            result = run_trial(
                model_kind=model_kind,
                cfg=cfg,
                criterion=criterion,
                batch_size=args.batch_size,
                rows=rows,
                cols=cols,
                missing_rate=args.missing_rate,
                decode_only_missing=decode_mode == "missing",
                device=args.device,
            )
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
