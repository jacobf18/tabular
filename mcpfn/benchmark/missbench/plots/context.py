"""Inputs and options shared by all figure and table generators."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..methods import METHODS, PAPER_METHODS, specialist_key
from ..patterns import PATTERNS


class MissingInput(Exception):
    """Raised when a figure's input results have not been produced yet."""


@dataclass
class Context:
    results_dir: Path
    out_dir: Path
    # Baselines to show (default: the paper's methods that have results)
    methods: list[str] | None = None
    # Pair of methods for the head-to-head scatter plot
    compare: tuple[str, str] = ("tabimpute", "missforest")
    _cache: dict = field(default_factory=dict)

    def load(self, experiment: str) -> pd.DataFrame:
        if experiment not in self._cache:
            path = Path(self.results_dir) / f"{experiment}.csv"
            if not path.exists():
                raise MissingInput(f"{path} not found")
            self._cache[experiment] = pd.read_csv(path)
        return self._cache[experiment]

    def scores(self, experiment: str = "openml", metric: str = "nrmse") -> pd.DataFrame:
        """Rows with a value for `metric`."""
        df = self.load(experiment)
        return df[df[metric].notna()]

    def baselines(self, df: pd.DataFrame) -> list[str]:
        """Default TabImpute plus the selected baselines that have results in `df`."""
        wanted = self.methods or PAPER_METHODS
        present = set(df["method"])
        keys = [m for m in wanted if m in present]
        if not keys:
            raise MissingInput(f"none of the methods {wanted} have results")
        self._warn_coverage(df, keys)
        return keys

    def _warn_coverage(self, df: pd.DataFrame, methods: list[str]) -> None:
        """Warn (once) if methods were not all run on the same benchmark instances."""
        counts = df[df["method"].isin(methods)].groupby("method").size()
        incomplete = counts[counts < counts.max()]
        key = ("coverage", tuple(sorted(incomplete.index)))
        if len(incomplete) and key not in self._cache:
            self._cache[key] = True
            names = ", ".join(f"{m} ({n}/{counts.max()})" for m, n in incomplete.items())
            print(f"  Warning: some methods have results for fewer benchmark instances, so their "
                  f"averages are not comparable: {names}")

    def specialists(self, df: pd.DataFrame) -> dict[str, str]:
        """pattern key -> pattern-specific TabImpute method key, for those with results."""
        present = set(df["method"])
        return {p: specialist_key(p) for p in PATTERNS if p != "mcar" and specialist_key(p) in present}

    def out(self, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir / name


def require(df: pd.DataFrame, methods: list[str], what: str) -> None:
    missing = [m for m in methods if m not in set(df["method"])]
    if missing:
        names = ", ".join(METHODS[m].display if m in METHODS else m for m in missing)
        raise MissingInput(f"{what} needs results for: {names}")
