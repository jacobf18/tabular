#!/usr/bin/env python3
"""Scatter MatFormer prediction errors vs Boost prediction errors.

For each missing cell across (settings x seeds) of the synthetic_context
benchmark with block missingness, build the figure:

  rows: alpha in {0.05, 0.10}
  left col : e_mf  = true - y_mf        (x)  vs  e_boost = true - y_boost (y)
  right col: |e_mf|                      (x)  vs  |y_boost - y_mf| = |alpha*r_pred| (y)

The left panel shows whether xRFM's residual correction shrinks MatFormer's
errors toward zero (points pulled inside the y=x diagonal). The right panel
shows whether the magnitude of the per-cell correction scales with the
MatFormer error - i.e. does xRFM apply bigger fixes where MatFormer is
more wrong.

alpha=0.05 predictions are derived from alpha=0.10 via the linear-blend
identity  y_boost(alpha) = y_mf + (alpha/0.1) * (y_boost(0.1) - y_mf),
so only one regeneration pass is needed.

Predictions per draw are cached under scatter_data/ as .npz so re-runs
just replot.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/root/tabular/mcpfn/src')
sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from generate import ALPHA_SETTINGS, make_draw, observed_matrix  # noqa: E402
from tabimpute.interface_rfm import BoostMatFormerImputer  # noqa: E402


ROUND2_T2_CONFIG = {'config': {
    'embedding_size': 768, 'num_attention_heads': 16, 'mlp_hidden_size': 1536,
    'num_layers': 8, 'num_outputs': 5000, 'num_cls': 12, 'use_rope': True,
    'rope_base': 10000.0, 'rope_fraction': 0.5,
    'use_absolute_positional_embeddings': False, 'positional_damping_factor': 0.1,
    'attention_dropout': 0.0, 'ffn_dropout': 0.05, 'drop_path_rate': 0.01,
    'residual_scale_init': 0.5, 'embedding_dropout': 0.0, 'rms_norm_eps': 1e-6,
}}
CHECKPOINT_PATH = '/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth'

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "scatter_data"
FIG_PATH = HERE / "figures" / "residuals_scatter.pdf"


def run_matformer(imputer: BoostMatFormerImputer, X: np.ndarray) -> np.ndarray:
    import torch
    from tabimpute.interface import ImputePFN
    with torch.no_grad():
        X_imputed, _ = ImputePFN.impute(imputer, X, return_full=True)
    return np.asarray(X_imputed, dtype=np.float32)


def gather_predictions(settings, n_seeds, missingness, force=False):
    """Return dict (setting, seed) -> (true, mask, y_mf, y_boost_a0p1)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    imputer = None
    cache = {}
    for setting in settings:
        if setting not in ALPHA_SETTINGS:
            raise SystemExit(f"unknown setting: {setting}")
        for seed in range(n_seeds):
            p = CACHE_DIR / f"{missingness}_{setting}_seed{seed:03d}.npz"
            if p.exists() and not force:
                d = np.load(p)
                cache[(setting, seed)] = (
                    d['true'], d['mask'].astype(bool), d['y_mf'], d['y_boost'],
                )
                continue
            if imputer is None:
                print("Loading BoostMatFormerImputer (alpha=0.1) ...", flush=True)
                imputer = BoostMatFormerImputer(
                    device='cuda', entry_wise_features=False,
                    json_config=ROUND2_T2_CONFIG, checkpoint_path=CHECKPOINT_PATH,
                    max_num_rows=200, max_num_cols=200, blend_alpha=0.1,
                )
            t0 = time.time()
            draw = make_draw(N=200, T=200, alphas=ALPHA_SETTINGS[setting],
                             seed=seed, missingness=missingness)
            Xobs = observed_matrix(draw)
            y_mf = run_matformer(imputer, Xobs)
            y_boost = np.asarray(imputer.impute(Xobs), dtype=np.float32)
            true = draw.M.astype(np.float32)
            mask = draw.mask.astype(bool)
            cache[(setting, seed)] = (true, mask, y_mf, y_boost)
            np.savez(p, true=true, mask=mask, y_mf=y_mf, y_boost=y_boost)
            print(f"[{setting} s{seed}] cached  ({time.time() - t0:.1f}s)",
                  flush=True)
    return cache


def pool_residuals(cache, alpha_blend: float):
    """Pool (e_mf, e_boost) across all draws for one alpha value.

    e = true - pred  (matches plot_prop99_scatter.py convention).
    """
    e_mfs, e_boosts = [], []
    for (true, mask, y_mf, y_boost_a0p1) in cache.values():
        # Derive y_boost at any alpha via linear-blend identity.
        y_boost = y_mf + (alpha_blend / 0.1) * (y_boost_a0p1 - y_mf)
        e_mfs.append((true - y_mf)[mask].ravel())
        e_boosts.append((true - y_boost)[mask].ravel())
    return np.concatenate(e_mfs), np.concatenate(e_boosts)


def plot(cache, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alphas = [0.05, 0.10, 0.15, 0.20]

    # Shared axis range across all panels (so the visual "rotation" is comparable).
    all_residuals = []
    for alpha in alphas:
        e_mf, e_boost = pool_residuals(cache, alpha)
        all_residuals.extend([e_mf, e_boost])
    v = float(np.percentile(np.abs(np.concatenate(all_residuals)), 99.5))
    v = max(v, 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), constrained_layout=True)
    axes_flat = axes.ravel()
    for idx, alpha in enumerate(alphas):
        e_mf, e_boost = pool_residuals(cache, alpha)

        ax = axes_flat[idx]
        ax.scatter(e_mf, e_boost, s=2, c='0.55', alpha=0.5,
                   edgecolors='none', rasterized=True)

        ax.axhline(0, color='gray', lw=0.5, alpha=0.5, zorder=1)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.5, zorder=1)

        # y = x reference (boost did nothing).
        ax.plot([-v, v], [-v, v], color='crimson', lw=2.5, linestyle='--',
                label='y = x  (no change)', zorder=3)

        # OLS fit  e_boost ~ a * e_mf + b.
        slope, intercept = np.polyfit(e_mf, e_boost, 1)
        xs = np.array([-v, v])
        ax.plot(xs, slope * xs + intercept, color='royalblue', lw=2.8,
                label=f'OLS fit  (slope = {slope:.3f})', zorder=4)

        corr = float(np.corrcoef(e_mf, e_boost)[0, 1])
        n = len(e_mf)
        rmse_mf = float(np.sqrt(np.mean(e_mf ** 2)))
        rmse_boost = float(np.sqrt(np.mean(e_boost ** 2)))
        ax.text(0.04, 0.96, (
            f"alpha       = {alpha}\n"
            f"slope       = {slope:.3f}\n"
            f"correlation = {corr:.3f}\n"
            f"RMSE  MF    = {rmse_mf:.3f}\n"
            f"RMSE  boost = {rmse_boost:.3f}\n"
            f"n cells     = {n:,}"
        ), transform=ax.transAxes, va='top', ha='left',
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='white', alpha=0.9))

        ax.set_xlim(-v, v); ax.set_ylim(-v, v)
        ax.set_xlabel(r'MatFormer error    $e_{\rm MF} = y_{\rm true} - y_{\rm MF}$')
        ax.set_ylabel(r'Boost error        $e_{\rm boost} = y_{\rm true} - y_{\rm boost}$')
        ax.set_title(f'alpha = {alpha}')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')

    fig.suptitle(
        "Synthetic context (block missingness): how xRFM corrects MatFormer's per-cell errors\n"
        "Slope < 1 means boost shrinks errors toward zero.",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--settings", nargs="+", default=list(ALPHA_SETTINGS.keys()))
    ap.add_argument("--missingness",
                    choices=["block", "mcar", "mar", "mnar"], default="block")
    ap.add_argument("--force", action="store_true",
                    help="Re-run predictions even if a cache file exists.")
    ap.add_argument("--out", type=Path, default=FIG_PATH)
    args = ap.parse_args()

    cache = gather_predictions(args.settings, args.n_seeds, args.missingness,
                               force=args.force)
    plot(cache, args.out)


if __name__ == "__main__":
    main()
