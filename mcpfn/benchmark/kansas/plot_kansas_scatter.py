#!/usr/bin/env python3
"""Kansas: entrywise treated-block errors, MatFormer variants vs SDiD.

Single panel per method (one PDF per variant) plus a 1x3 combined figure
showing MatFormer / MatFormer (transpose) / MatFormer (avg) side-by-side
on Kansas (kansas_factor_r3_200).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from plot_options import HIGHLIGHT_COLOR, PLOT_PARAMS  # noqa: E402

matplotlib.rcParams["font.family"] = "serif"

CAUSAL_EVAL_DIR = Path(__file__).resolve().parent / "causal_eval"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
BENCHMARK = "kansas_factor_r3_200"

DWIP_METHODS = ["dwip-v2-2", "dwip-v2-2-T", "dwip-v2-2-avg"]
DWIP_LABELS = {
    "dwip-v2-2": "MatFormer",
    "dwip-v2-2-T": "MatFormer (transpose)",
    "dwip-v2-2-avg": "MatFormer (avg)",
}


def list_draws(combo_dir: Path) -> list[str]:
    return sorted(
        d.name for d in combo_dir.iterdir()
        if d.is_dir() and d.name.startswith("draw_")
    )


def gather_errors(
    combo_dir: Path, dwip_method: str
) -> tuple[np.ndarray, np.ndarray]:
    sdid_errs: list[np.ndarray] = []
    dwip_errs: list[np.ndarray] = []
    for draw in list_draws(combo_dir):
        ddir = combo_dir / draw
        sdid_path = ddir / "sdid.npy"
        dwip_path = ddir / f"{dwip_method}.npy"
        if not (sdid_path.is_file() and dwip_path.is_file()):
            continue
        true = np.load(ddir / "true.npy")
        mask = np.load(ddir / "treatment_mask.npy").astype(bool)
        sdid = np.load(sdid_path)
        dwip = np.load(dwip_path)
        sdid_errs.append((true[mask] - sdid[mask]).ravel())
        dwip_errs.append((true[mask] - dwip[mask]).ravel())
    return np.concatenate(sdid_errs), np.concatenate(dwip_errs)


def plot_entrywise_scatter(out: Path, dwip_method: str) -> None:
    dwip_label = DWIP_LABELS.get(dwip_method, dwip_method)
    combo_dir = CAUSAL_EVAL_DIR / BENCHMARK
    sdid_err, dwip_err = gather_errors(combo_dir, dwip_method)

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = float(max(
        np.abs(np.percentile(dwip_err, [1, 99])).max(),
        np.abs(np.percentile(sdid_err, [1, 99])).max(),
    ))
    ax.scatter(dwip_err, sdid_err, s=6, alpha=0.15,
               color=HIGHLIGHT_COLOR, linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], color="black",
            linewidth=0.7, linestyle="--", label="y=x")
    ax.axhline(0, color="#cccccc", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.4, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{dwip_label} error (true - pred)")
    ax.set_ylabel("SDiD error (true - pred)")
    ax.set_title(f"Kansas: entrywise treated-block errors ({dwip_label})")
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)

    n = len(dwip_err)
    dwip_rmse = float(np.sqrt(np.mean(dwip_err ** 2)))
    sdid_rmse = float(np.sqrt(np.mean(sdid_err ** 2)))
    print(f"  points={n}  RMSE {dwip_label}={dwip_rmse:.4f}  "
          f"SDiD={sdid_rmse:.4f}")


def plot_combined_scatter(out: Path) -> None:
    combo_dir = CAUSAL_EVAL_DIR / BENCHMARK
    fig, axes = plt.subplots(1, len(DWIP_METHODS),
                             figsize=(5 * len(DWIP_METHODS), 5), squeeze=False)
    for idx, method in enumerate(DWIP_METHODS):
        ax = axes[0][idx]
        label = DWIP_LABELS.get(method, method)
        sdid_err, dwip_err = gather_errors(combo_dir, method)
        if sdid_err.size == 0:
            ax.set_visible(False)
            print(f"  [skip] no matching draws for {method}")
            continue
        lim = float(max(
            np.abs(np.percentile(dwip_err, [1, 99])).max(),
            np.abs(np.percentile(sdid_err, [1, 99])).max(),
        ))
        ax.scatter(dwip_err, sdid_err, s=6, alpha=0.15,
                   color=HIGHLIGHT_COLOR, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color="black",
                linewidth=0.7, linestyle="--", label="y=x")
        ax.axhline(0, color="#cccccc", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#cccccc", linewidth=0.4, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"{label} error (true - pred)")
        if idx == 0:
            ax.set_ylabel("SDiD error (true - pred)")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        n = len(dwip_err)
        dwip_rmse = float(np.sqrt(np.mean(dwip_err ** 2)))
        sdid_rmse = float(np.sqrt(np.mean(sdid_err ** 2)))
        print(f"  {method}: points={n}  RMSE {label}={dwip_rmse:.4f}  "
              f"SDiD={sdid_rmse:.4f}")
    fig.suptitle("Kansas entrywise treated-block errors: MatFormer variants vs SDiD",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def main() -> None:
    out_dir = FIGURES_DIR / BENCHMARK
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in DWIP_METHODS:
        slug = method.replace("-", "_")
        out = out_dir / f"kansas_entrywise_scatter_sdid_vs_{slug}.pdf"
        print(f"[{method}]")
        plot_entrywise_scatter(out, method)
        print(f"Wrote {out}")
    print("[combined 1x3]")
    combined_out = out_dir / "kansas_entrywise_scatter_sdid_vs_matformer_variants.pdf"
    plot_combined_scatter(combined_out)
    print(f"Wrote {combined_out}")


if __name__ == "__main__":
    main()
