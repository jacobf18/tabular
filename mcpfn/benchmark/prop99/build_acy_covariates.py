"""Build state-level and year-level covariate files for the ACY paper's
Prop99 experiment (Section 5.2 of arxiv:2603.24833).

Reads the standard Abadie-Diamond-Hainmueller 2010 smoking dataset
(smoking_full.csv) and writes two files:

  - covariates_states.csv: (S, 5+1) state-level row context
  - covariates_years.csv : (T, 3)   year-level column context
  - panel_cigsale.csv    : (S, T)   the full cigsale matrix M

State covariates (5; mirror the paper as closely as the available data permits):
  1. mean retprice (pre-treatment 1970-1988)
  2. mean lnincome (where available, pre-treatment)
  3. mean age15to24 (pre-treatment 1970-1988)
  4. mean beer       (only 1984-1988 available; pre-treatment mean over that)
  5. pre-treatment mean cigsale (state-specific "tobacco preference proxy")

Year covariates (3):
  1. cross-state mean cigsale per year (national smoking trend)
  2. cross-state mean retprice per year (national price trend)
  3. Florida + Michigan mean cigsale per year (paper's preference-in-time proxy)

Pre-treatment window: 1970-1988 (treatment year = 1989).

Notes on what's NOT included (paper has these too, we couldn't source cheaply):
  - "% adults completing 4+ years of college" — would need US Census ACS data.
  - "Log per-capita real GDP" national — would need BEA data. We use cross-state
    mean cigsale as a substitute time-varying signal instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROP99_DIR = Path("/root/tabular/mcpfn/benchmark/prop99")
PRE_TREATMENT_YEARS = (1970, 1988)   # inclusive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=PROP99_DIR / "smoking_full.csv")
    ap.add_argument("--out_dir", type=Path, default=PROP99_DIR / "acy_paper_data")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["year"] = df["year"].astype(int)
    print(f"Loaded {df.shape[0]} rows; {df['state'].nunique()} states, "
          f"years {df['year'].min()}-{df['year'].max()}")

    # ---- Panel: states × years cigsale ----
    panel = df.pivot(index="state", columns="year", values="cigsale").sort_index()
    panel = panel.reindex(columns=range(1970, 2001))
    assert panel.isna().sum().sum() == 0, "cigsale should be complete"
    print(f"Panel shape: {panel.shape}  (S × T)")
    panel.to_csv(args.out_dir / "panel_cigsale.csv")

    # ---- State-level covariates (5) ----
    pre = df[df["year"].between(*PRE_TREATMENT_YEARS)]
    # For each covariate, average over years where it's not NaN.
    state_covs = pd.DataFrame(index=panel.index)
    state_covs["retprice_pre"]  = pre.groupby("state")["retprice"].mean()
    state_covs["lnincome_pre"]  = pre.groupby("state")["lnincome"].mean()
    state_covs["age15to24_pre"] = pre.groupby("state")["age15to24"].mean()
    state_covs["beer_pre"]      = pre.groupby("state")["beer"].mean()
    state_covs["cigsale_pre"]   = pre.groupby("state")["cigsale"].mean()
    # Fill any remaining NaNs with column mean (some states may lack beer/lnincome).
    state_covs = state_covs.fillna(state_covs.mean())
    print(f"State covariates shape: {state_covs.shape}  (S × d_row)")
    print(state_covs.head())
    state_covs.to_csv(args.out_dir / "covariates_states.csv")

    # ---- Year-level covariates (3) ----
    # Note: the paper's third covariate is Florida + Michigan avg cigarette
    # sales per year (as a tobacco-preference-in-time proxy). Those two
    # states are excluded from the ADH donor pool we have. We substitute
    # with the YEAR median cigsale across donor states as a robust proxy
    # for general tobacco-preference trend.
    year_covs = pd.DataFrame(index=panel.columns)   # year index
    year_covs.index.name = "year"
    year_covs["cigsale_mean_states"]  = df.groupby("year")["cigsale"].mean()
    year_covs["retprice_mean_states"] = df.groupby("year")["retprice"].mean()
    year_covs["cigsale_median_states"] = df.groupby("year")["cigsale"].median()
    print(f"Year covariates shape: {year_covs.shape}  (T × d_col)")
    print(year_covs.head())
    year_covs.to_csv(args.out_dir / "covariates_years.csv")

    print(f"\nWrote files to {args.out_dir}/")
    print(f"  panel_cigsale.csv         (S={panel.shape[0]} × T={panel.shape[1]})")
    print(f"  covariates_states.csv     (S × {state_covs.shape[1]})")
    print(f"  covariates_years.csv      (T × {year_covs.shape[1]})")


if __name__ == "__main__":
    main()
