#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabimpute.prior.rec_sys_spectrum_exposure import RecSysSpectrumExposureGenerator


def spectrum_stats(singular_values: np.ndarray) -> dict[str, float]:
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return {
            "rank90": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "top10_energy": 0.0,
            "s10_over_s1": float("nan"),
            "s20_over_s1": float("nan"),
        }
    energy = s**2
    total = float(energy.sum())
    cumulative = np.cumsum(energy) / max(total, 1e-300)
    probs = energy / max(total, 1e-300)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-300))))

    def ratio_at(k: int) -> float:
        if s.size < k:
            return float("nan")
        return float(s[k - 1] / max(s[0], 1e-300))

    return {
        "rank90": float(np.searchsorted(cumulative, 0.90) + 1),
        "stable_rank": float(total / max(float(s[0] ** 2), 1e-300)),
        "effective_rank": float(np.exp(entropy)),
        "top10_energy": float(energy[: min(10, s.size)].sum() / max(total, 1e-300)),
        "s10_over_s1": ratio_at(10),
        "s20_over_s1": ratio_at(20),
    }


def observed_zero(observed_matrix: np.ndarray, *, centered: bool) -> np.ndarray:
    observed = np.isfinite(observed_matrix)
    out = np.zeros_like(observed_matrix, dtype=np.float64)
    if centered and observed.any():
        out[observed] = observed_matrix[observed] - float(np.nanmean(observed_matrix))
    else:
        out[observed] = observed_matrix[observed]
    return out


def parse_shape(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def diagnose_one(
    *,
    label: str,
    config: dict,
    chunk_shapes: list[tuple[int, int]],
    chunk_samples: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    np.random.seed(seed)
    bundle = RecSysSpectrumExposureGenerator(config).generate_matrix_bundle()
    observed = bundle["observed_matrix"].numpy()
    mask = bundle["observed_mask"].numpy().astype(bool)
    raw = observed_zero(observed, centered=False)
    centered = observed_zero(observed, centered=True)
    raw_s = np.linalg.svd(raw, compute_uv=False, full_matrices=False)
    centered_s = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    global_row = {
        "label": label,
        "rows": int(observed.shape[0]),
        "cols": int(observed.shape[1]),
        "density_realized": float(mask.mean()),
        "density_target": float(bundle["density_target"]),
        "row_count_median": float(np.median(row_counts)),
        "row_count_p99": float(np.percentile(row_counts, 99)),
        "col_count_median": float(np.median(col_counts)),
        "col_count_p99": float(np.percentile(col_counts, 99)),
        **{f"raw_{key}": value for key, value in spectrum_stats(raw_s).items()},
        **{f"centered_{key}": value for key, value in spectrum_stats(centered_s).items()},
    }
    chunk_rows = []
    rng = np.random.default_rng(seed + 1)
    for chunk_rows_n, chunk_cols_n in chunk_shapes:
        row_take = min(chunk_rows_n, observed.shape[0])
        col_take = min(chunk_cols_n, observed.shape[1])
        for sample_idx in range(chunk_samples):
            row_idx = rng.choice(observed.shape[0], size=row_take, replace=False)
            col_idx = rng.choice(observed.shape[1], size=col_take, replace=False)
            chunk = centered[np.ix_(row_idx, col_idx)]
            chunk_mask = mask[np.ix_(row_idx, col_idx)]
            if chunk_mask.any():
                s = np.linalg.svd(chunk, compute_uv=False, full_matrices=False)
            else:
                s = np.array([], dtype=np.float64)
            chunk_rows.append(
                {
                    "label": label,
                    "shape": f"{chunk_rows_n}x{chunk_cols_n}",
                    "sample": sample_idx,
                    "observed_fraction": float(chunk_mask.mean()),
                    **spectrum_stats(s),
                }
            )
    return global_row, chunk_rows, {"raw": raw_s, "centered": centered_s}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/root/tabular/mcpfn/runs/rec_sys_prior_diagnostics"))
    parser.add_argument("--chunk-shapes", nargs="+", default=["50x50", "100x100", "100x300", "300x100", "300x300"])
    parser.add_argument("--chunk-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_shapes = [parse_shape(value) for value in args.chunk_shapes]
    configs = {
        "movielens_scaled_300x500_p05": {
            "num_rows_low": 300,
            "num_rows_high": 300,
            "num_cols_low": 500,
            "num_cols_high": 500,
            "recsys_density_low": 0.05,
            "recsys_density_high": 0.05,
            "recsys_spectrum_rank_low": 80,
            "recsys_spectrum_rank_high": 128,
        },
        "netflix_scaled_300x500_p01": {
            "num_rows_low": 300,
            "num_rows_high": 300,
            "num_cols_low": 500,
            "num_cols_high": 500,
            "recsys_density_low": 0.01,
            "recsys_density_high": 0.01,
            "recsys_spectrum_rank_low": 80,
            "recsys_spectrum_rank_high": 128,
            "recsys_user_activity_sigma_low": 1.5,
            "recsys_user_activity_sigma_high": 1.5,
            "recsys_item_popularity_sigma_low": 2.5,
            "recsys_item_popularity_sigma_high": 2.5,
        },
    }
    global_rows = []
    chunk_rows = []
    spectra = {}
    for idx, (label, config) in enumerate(configs.items()):
        global_row, chunk_row, spectrum = diagnose_one(
            label=label,
            config=config,
            chunk_shapes=chunk_shapes,
            chunk_samples=args.chunk_samples,
            seed=args.seed + idx * 100,
        )
        global_rows.append(global_row)
        chunk_rows.extend(chunk_row)
        spectra[label] = spectrum

    write_csv(args.output_dir / "global_summary.csv", global_rows)
    write_csv(args.output_dir / "chunk_samples.csv", chunk_rows)
    with (args.output_dir / "global_summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_rows, f, indent=2, sort_keys=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for label, spectrum in spectra.items():
        for variant, s in spectrum.items():
            s = s[: min(120, s.size)]
            ax.semilogy(
                np.arange(1, s.size + 1),
                s / max(float(s[0]), 1e-300),
                label=f"{label} {variant}",
            )
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value / top singular value")
    ax.set_title("Rec-sys synthetic prior spectra")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "global_spectra.png")
    plt.close(fig)
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
