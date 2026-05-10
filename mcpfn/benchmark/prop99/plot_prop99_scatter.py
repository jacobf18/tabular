#!/usr/bin/env python3
"""prop99: entrywise treated-block errors, TabImpute vs SDiD.

Three panels — one per benchmark (ca_simul, ca_stagg, stocks). Each point =
one treated cell from one draw. x = TabImpute error (true - pred),
y = SDiD error. Produces one figure per TabImpute variant (v2-1, v2-2).
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

BENCHMARKS = ["ca_simul", "ca_stagg", "stocks"]
BENCHMARK_LABELS = {
    "ca_simul": "California (simultaneous)",
    "ca_stagg": "California (staggered)",
    "stocks": "Stocks",
}

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
        if not mask.any():
            continue
        sdid = np.load(sdid_path)
        dwip = np.load(dwip_path)
        sdid_errs.append((true[mask] - sdid[mask]).ravel())
        dwip_errs.append((true[mask] - dwip[mask]).ravel())
    if not sdid_errs:
        return np.array([]), np.array([])
    return np.concatenate(sdid_errs), np.concatenate(dwip_errs)


def plot_entrywise_scatter(out: Path, dwip_method: str) -> None:
    dwip_label = DWIP_LABELS.get(dwip_method, dwip_method)
    ncol = len(BENCHMARKS)
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5), squeeze=False)

    for idx, combo in enumerate(BENCHMARKS):
        ax = axes[0][idx]
        combo_dir = CAUSAL_EVAL_DIR / combo
        if not combo_dir.is_dir():
            ax.set_visible(False)
            print(f"  [skip] missing {combo_dir}")
            continue
        sdid_err, dwip_err = gather_errors(combo_dir, dwip_method)
        if sdid_err.size == 0:
            ax.set_visible(False)
            print(f"  [skip] no matching draws for {combo}")
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
        ax.set_title(BENCHMARK_LABELS.get(combo, combo), fontsize=11)
        ax.set_xlabel(f"{dwip_label} error (true - pred)")
        if idx == 0:
            ax.set_ylabel("SDiD error (true - pred)")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        n = len(dwip_err)
        dwip_rmse = float(np.sqrt(np.mean(dwip_err ** 2)))
        sdid_rmse = float(np.sqrt(np.mean(sdid_err ** 2)))
        print(f"  {combo}: points={n}  RMSE {dwip_label}={dwip_rmse:.4f}  "
              f"SDiD={sdid_rmse:.4f}")

    fig.suptitle(
        f"prop99 entrywise treated-block errors: {dwip_label} vs SDiD",
        y=1.00, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def plot_combined_scatter(out: Path, combo: str) -> None:
    combo_dir = CAUSAL_EVAL_DIR / combo
    fig, axes = plt.subplots(1, len(DWIP_METHODS),
                             figsize=(5 * len(DWIP_METHODS), 5), squeeze=False)
    for idx, method in enumerate(DWIP_METHODS):
        ax = axes[0][idx]
        label = DWIP_LABELS.get(method, method)
        sdid_err, dwip_err = gather_errors(combo_dir, method)
        if sdid_err.size == 0:
            ax.set_visible(False)
            print(f"  [skip] no matching draws for {combo}/{method}")
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
        print(f"  {combo}/{method}: points={n}  RMSE {label}={dwip_rmse:.4f}  "
              f"SDiD={sdid_rmse:.4f}")
    fig.suptitle(
        f"{BENCHMARK_LABELS.get(combo, combo)} entrywise treated-block errors: "
        f"MatFormer variants vs SDiD",
        y=1.00, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for method in DWIP_METHODS:
        slug = method.replace("-", "_")
        out = FIGURES_DIR / f"prop99_entrywise_scatter_sdid_vs_{slug}.pdf"
        print(f"[{method}]")
        plot_entrywise_scatter(out, method)
        print(f"Wrote {out}")
    print("[combined ca_stagg 1x3]")
    combined_out = FIGURES_DIR / "prop99_ca_stagg_entrywise_scatter_sdid_vs_matformer_variants.pdf"
    plot_combined_scatter(combined_out, "ca_stagg")
    print(f"Wrote {combined_out}")


if __name__ == "__main__":
    main()
