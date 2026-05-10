"""Aggregate per-replicate metrics and generate plots.

Inputs:
  outputs/metrics/<dataset>_<size>_paper.csv      (from run_paper_benchmarks)
  outputs/metrics/<dataset>_<size>_tabimpute.csv  (from run_tabimpute_eval)

Outputs:
  outputs/metrics/aggregate_paper.csv
  outputs/metrics/aggregate_tabimpute.csv
  outputs/plots/<dataset>_<size>_paper_rmse.png
  outputs/plots/<dataset>_<size>_tabimpute_rmse.png
  outputs/plots/<dataset>_<size>_paper_plus_tabimpute.png

Aggregation columns per (dataset, size, method, withhold):
  mean_rmse, std_rmse, n, ci95_low, ci95_high
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_ROOT))

from src.utils.logging_utils import get_logger  # noqa
from src.utils.paths import METRICS_DIR, PLOTS_DIR  # noqa

LOGGER = get_logger("aggregation")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ci95(values: np.ndarray) -> Tuple[float, float]:
    n = values.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1 or np.std(values, ddof=1) == 0.0:
        return mean, mean
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    half = 1.96 * se
    return mean - half, mean + half


def _aggregate(rows: List[Dict[str, str]], group_cols: List[str]) -> List[Dict]:
    groups: Dict[Tuple, List[float]] = {}
    for row in rows:
        if row.get("status") == "error":
            continue
        if row.get("rmse") in (None, "", "nan"):
            continue
        rmse = _as_float(row["rmse"])
        if not math.isfinite(rmse):
            continue
        key = tuple(row.get(c, "") for c in group_cols)
        groups.setdefault(key, []).append(rmse)

    out = []
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = _ci95(arr)
        row = {c: k for c, k in zip(group_cols, key)}
        row.update({
            "n": int(arr.size),
            "mean_rmse": float(np.mean(arr)),
            "std_rmse": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
            "min_rmse": float(np.min(arr)),
            "max_rmse": float(np.max(arr)),
        })
        out.append(row)
    return out


def _write_agg(path: Path, rows: List[Dict]) -> None:
    if not rows:
        LOGGER.warning("No rows to write to %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    LOGGER.info("Wrote %d rows to %s", len(rows), path)


# ----------------------------------------------------------------- plotting
def _load_plot_options():
    """Reuse benchmark/plot_options.py if importable; else sensible defaults."""
    bench_dir = Path("/root/tabular/mcpfn/benchmark")
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    try:
        import plot_options as po  # type: ignore
        return po
    except Exception:
        LOGGER.warning("plot_options.py not available; using matplotlib defaults.")
        return None


def _method_display(name: str) -> str:
    mapping = {
        "paper_gaussian_kernel": "User-Item Gaussian Kernel (paper)",
        "user_user_cf": "User-User CF",
        "item_item_cf": "Item-Item CF",
        "softimpute": "SoftImpute",
    }
    return mapping.get(name, name)


def _plot_paper(dataset: str, size: str, rows: List[Dict]) -> Optional[Path]:
    import matplotlib.pyplot as plt

    po = _load_plot_options()

    subset = [r for r in rows if r["dataset"] == dataset and r["size"] == size]
    if not subset:
        return None

    methods = sorted({r["method"] for r in subset})
    withholds = sorted({float(r["withhold"]) for r in subset})

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in methods:
        xs, ys, lo, hi = [], [], [], []
        for w in withholds:
            matches = [r for r in subset if r["method"] == method and float(r["withhold"]) == w]
            if not matches:
                continue
            m = matches[0]
            xs.append(w * 100)
            ys.append(float(m["mean_rmse"]))
            lo.append(float(m["ci95_low"]))
            hi.append(float(m["ci95_high"]))
        if not xs:
            continue
        color = None
        if po is not None:
            color = po.METHOD_COLORS.get(_method_display(method), None)
        line, = ax.plot(xs, ys, marker="o", label=_method_display(method), color=color)
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())

    ax.set_xlabel("% withheld ratings")
    ax.set_ylabel("RMSE on withheld entries")
    ax.set_title(f"{dataset.capitalize()} {size} — 100 x 2000x2000 paper subsamples")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    out = PLOTS_DIR / f"{dataset}_{size}_paper_rmse.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


def _plot_tabimpute(dataset: str, size: str, rows: List[Dict]) -> Optional[Path]:
    import matplotlib.pyplot as plt

    subset = [r for r in rows if r["dataset"] == dataset and r["size"] == size and r.get("input", "subsamples") == "subsamples"]
    if not subset:
        return None

    # Group by (model, eval_mode_used)
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in subset:
        k = (r["model"], r.get("eval_mode_used", r.get("eval_mode_requested", "?")))
        groups.setdefault(k, []).append(r)
    if not groups:
        return None

    withholds = sorted({float(r["withhold"]) for r in subset})

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for (model, mode), items in sorted(groups.items()):
        xs, ys, lo, hi = [], [], [], []
        for w in withholds:
            matches = [r for r in items if float(r["withhold"]) == w]
            if not matches:
                continue
            m = matches[0]
            xs.append(w * 100)
            ys.append(float(m["mean_rmse"]))
            lo.append(float(m["ci95_low"]))
            hi.append(float(m["ci95_high"]))
        if not xs:
            continue
        line, = ax.plot(xs, ys, marker="s", label=f"{model} [{mode}]")
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())

    ax.set_xlabel("% withheld ratings")
    ax.set_ylabel("RMSE on withheld entries")
    ax.set_title(f"{dataset.capitalize()} {size} — TabImpute on paper subsamples")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    out = PLOTS_DIR / f"{dataset}_{size}_tabimpute_rmse.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


def _plot_combined(dataset: str, size: str, paper_rows: List[Dict], ti_rows: List[Dict]) -> Optional[Path]:
    import matplotlib.pyplot as plt

    paper_sub = [r for r in paper_rows if r["dataset"] == dataset and r["size"] == size]
    ti_sub = [r for r in ti_rows if r["dataset"] == dataset and r["size"] == size and r.get("input", "subsamples") == "subsamples"]
    if not paper_sub and not ti_sub:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    withholds = sorted({float(r["withhold"]) for r in paper_sub + ti_sub})
    for method in sorted({r["method"] for r in paper_sub}):
        xs, ys, lo, hi = [], [], [], []
        for w in withholds:
            matches = [r for r in paper_sub if r["method"] == method and float(r["withhold"]) == w]
            if not matches:
                continue
            m = matches[0]
            xs.append(w * 100)
            ys.append(float(m["mean_rmse"]))
            lo.append(float(m["ci95_low"]))
            hi.append(float(m["ci95_high"]))
        if xs:
            line, = ax.plot(xs, ys, marker="o", label=_method_display(method))
            ax.fill_between(xs, lo, hi, alpha=0.12, color=line.get_color())

    for model in sorted({r["model"] for r in ti_sub}):
        xs, ys, lo, hi = [], [], [], []
        for w in withholds:
            matches = [r for r in ti_sub if r["model"] == model and float(r["withhold"]) == w]
            if not matches:
                continue
            m = matches[0]
            xs.append(w * 100)
            ys.append(float(m["mean_rmse"]))
            lo.append(float(m["ci95_low"]))
            hi.append(float(m["ci95_high"]))
        if xs:
            line, = ax.plot(xs, ys, marker="s", label=f"TabImpute {model}", linestyle="--")
            ax.fill_between(xs, lo, hi, alpha=0.12, color=line.get_color())

    ax.set_xlabel("% withheld ratings")
    ax.set_ylabel("RMSE on withheld entries")
    ax.set_title(f"{dataset.capitalize()} {size} — paper reproduction + TabImpute")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    out = PLOTS_DIR / f"{dataset}_{size}_paper_plus_tabimpute.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    LOGGER.info("Wrote %s", out)
    return out


def run(args) -> int:
    paper_all_rows: List[Dict[str, str]] = []
    ti_all_rows: List[Dict[str, str]] = []
    datasets = args.datasets or [("movielens", "1m"), ("netflix", "prize")]

    for dataset, size in datasets:
        paper_csv = METRICS_DIR / f"{dataset}_{size}_paper.csv"
        ti_csv = METRICS_DIR / f"{dataset}_{size}_tabimpute.csv"
        paper_rows = _read_csv(paper_csv)
        ti_rows = _read_csv(ti_csv)
        paper_all_rows.extend(paper_rows)
        ti_all_rows.extend(ti_rows)

    paper_agg = _aggregate(paper_all_rows, ["dataset", "size", "method", "withhold"])
    ti_agg = _aggregate(
        ti_all_rows,
        ["dataset", "size", "model", "input", "eval_mode_requested", "eval_mode_used", "withhold"],
    )

    _write_agg(METRICS_DIR / "aggregate_paper.csv", paper_agg)
    _write_agg(METRICS_DIR / "aggregate_tabimpute.csv", ti_agg)

    for dataset, size in datasets:
        _plot_paper(dataset, size, paper_agg)
        _plot_tabimpute(dataset, size, ti_agg)
        _plot_combined(dataset, size, paper_agg, ti_agg)
    return 0


def _parse_datasets(s: str) -> List[Tuple[str, str]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "/" not in tok:
            raise ValueError(f"expected dataset/size, got {tok}")
        d, sz = tok.split("/", 1)
        out.append((d, sz))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Aggregate all known dataset/size pairs")
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=None,
        help="Comma-separated dataset/size pairs, e.g. movielens/1m,netflix/prize",
    )
    args = parser.parse_args()
    if not args.all and not args.datasets:
        args.all = True
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
