#!/usr/bin/env python
"""Summarize SVT mixture/controller metrics from a training JSONL stream."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--last-n", type=int, default=500)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path, last_n: int) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if last_n > 0:
        rows = rows[-last_n:]
    return rows


def mean_finite(rows: list[dict[str, object]], key: str) -> float:
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            vals.append(float(value))
    return float(np.mean(vals)) if vals else float("nan")


def last_finite(rows: list[dict[str, object]], key: str) -> float:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return float("nan")


def step_weight_keys(row: dict[str, object]) -> list[tuple[int, str]]:
    out = []
    for key in row:
        prefix = "svt_mixture_weight_step_"
        if key.startswith(prefix):
            try:
                out.append((int(key[len(prefix) :]), key))
            except ValueError:
                continue
    return sorted(out)


def cv_rmse_keys(row: dict[str, object]) -> list[tuple[int, str]]:
    out = []
    for key in row:
        prefix = "svt_mixture_cv_rmse_step_"
        if key.startswith(prefix):
            try:
                out.append((int(key[len(prefix) :]), key))
            except ValueError:
                continue
    return sorted(out)


def summarize_group(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    latest = rows[-1]
    summary: dict[str, object] = {
        "group": name,
        "n": len(rows),
        "first_step": int(rows[0].get("step", -1)),
        "last_step": int(rows[-1].get("step", -1)),
        "loss_ema_last": last_finite(rows, "loss_missing_ema"),
        "loss_ema_mean": mean_finite(rows, "loss_missing_ema"),
        "cv_regret_last": last_finite(rows, "svt_mixture_cv_regret_loss"),
        "cv_regret_mean": mean_finite(rows, "svt_mixture_cv_regret_loss"),
        "rmse_mean_mean": mean_finite(rows, "rmse_missing_mean"),
        "expected_steps_mean": mean_finite(rows, "svt_mixture_expected_steps"),
        "oracle_step_mean": mean_finite(rows, "svt_mixture_depth_oracle_step"),
    }
    for step, key in step_weight_keys(latest):
        summary[f"weight_s{step}_mean"] = mean_finite(rows, key)
        summary[f"weight_s{step}_last"] = last_finite(rows, key)
    for step, key in cv_rmse_keys(latest):
        summary[f"cv_rmse_s{step}_mean"] = mean_finite(rows, key)
    if step_weight_keys(latest) and "svt_mixture_depth_oracle_step" in latest:
        agreements = []
        cv_agreements = []
        margins = []
        cv_margins = []
        for row in rows:
            weights = [(step, row.get(key)) for step, key in step_weight_keys(row)]
            weights = [
                (step, float(value))
                for step, value in weights
                if isinstance(value, (int, float)) and np.isfinite(float(value))
            ]
            oracle = row.get("svt_mixture_depth_oracle_step")
            if weights and isinstance(oracle, (int, float)):
                weights_sorted = sorted(weights, key=lambda item: item[1], reverse=True)
                agreements.append(float(weights_sorted[0][0] == int(float(oracle))))
                if len(weights_sorted) > 1:
                    margins.append(weights_sorted[0][1] - weights_sorted[1][1])
            cv_values = [(step, row.get(key)) for step, key in cv_rmse_keys(row)]
            cv_values = [
                (step, float(value))
                for step, value in cv_values
                if isinstance(value, (int, float)) and np.isfinite(float(value))
            ]
            if cv_values and isinstance(oracle, (int, float)):
                cv_sorted = sorted(cv_values, key=lambda item: item[1])
                cv_agreements.append(float(cv_sorted[0][0] == int(float(oracle))))
                if len(cv_sorted) > 1:
                    cv_margins.append(cv_sorted[1][1] - cv_sorted[0][1])
        if agreements:
            summary["top_weight_oracle_agreement"] = float(np.mean(agreements))
        if margins:
            summary["top_weight_margin_mean"] = float(np.mean(margins))
        if cv_agreements:
            summary["cv_top_oracle_agreement"] = float(np.mean(cv_agreements))
        if cv_margins:
            summary["cv_margin_mean"] = float(np.mean(cv_margins))
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "group",
        "n",
        "first_step",
        "last_step",
        "loss_ema_last",
        "loss_ema_mean",
        "cv_regret_last",
        "cv_regret_mean",
        "rmse_mean_mean",
        "expected_steps_mean",
        "oracle_step_mean",
        "top_weight_oracle_agreement",
        "top_weight_margin_mean",
        "cv_top_oracle_agreement",
        "cv_margin_mean",
    ]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics_jsonl, args.last_n)
    if not rows:
        raise SystemExit(f"No metrics rows found in {args.metrics_jsonl}")
    by_prior: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_prior[str(row.get("sampled_prior_name", "unknown"))].append(row)
    summary = [summarize_group("all", rows)]
    summary.extend(
        summarize_group(name, group)
        for name, group in sorted(by_prior.items())
    )
    if args.output_csv:
        write_csv(args.output_csv, summary)
    weight_fields = sorted(
        {
            key
            for row in summary
            for key in row
            if key.startswith("weight_s") and key.endswith("_last")
        },
        key=lambda key: int(key.split("_")[1][1:]),
    )
    fields = [
        "group",
        "n",
        "last_step",
        "loss_ema_last",
        "cv_regret_last",
        "expected_steps_mean",
        "oracle_step_mean",
        "top_weight_oracle_agreement",
        "top_weight_margin_mean",
        "cv_top_oracle_agreement",
        "cv_margin_mean",
        *weight_fields,
    ]
    print("\t".join(fields))
    for row in summary:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        print("\t".join(values))


if __name__ == "__main__":
    main()
