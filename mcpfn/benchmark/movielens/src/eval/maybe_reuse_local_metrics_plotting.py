"""Thin adapters that reuse the repo's existing plotting/metrics utilities.

This file is intentionally small — the real plotting lives in
``aggregation.py``, which imports ``benchmark/plot_options.py`` at runtime via
``sys.path`` so the color/font conventions stay consistent with the rest of
the repo. This module exists so a human (or an orchestrator) can grep for
"maybe_reuse_local_metrics_plotting" and see the integration surface.

Nothing in this file is re-implementing metrics: it only re-exports.
"""
from __future__ import annotations

import sys
from pathlib import Path


def load_repo_plot_options():
    bench_dir = Path("/root/tabular/mcpfn/benchmark")
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    try:
        import plot_options  # type: ignore
        return plot_options
    except Exception:
        return None
