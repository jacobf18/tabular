#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("runs/objective_queues/rowcol_svt_adaptive_all_summary.tsv"),
    )
    return parser.parse_args()


def model_rmse(path: Path) -> str:
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    model_rows = [row for row in rows if row.get("method") != "SoftImpute"]
    if not model_rows:
        return ""
    return f"{float(model_rows[0]['rmse']):.6g}"


def softimpute_rmse(path: Path) -> str:
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    soft_rows = [row for row in rows if row.get("method") == "SoftImpute"]
    if not soft_rows:
        return ""
    return f"{float(soft_rows[0]['rmse']):.6g}"


def spectral_rmse(path: Path, size: int) -> str:
    if not path.exists():
        return ""
    for row in csv.DictReader(path.open()):
        if int(row["size"]) == size:
            return f"{float(row['rmse']):.6g}"
    return ""


def mnar_rmse(path: Path, missingness_type: str, rank: int, *, softimpute: bool) -> str:
    if not path.exists():
        return ""
    for row in csv.DictReader(path.open()):
        is_softimpute = row.get("method") == "SoftImpute"
        if (
            row.get("missingness_type") == missingness_type
            and int(row.get("rank", -1)) == rank
            and is_softimpute == softimpute
        ):
            return f"{float(row['rmse']):.6g}"
    return ""


def policy_summary(path: Path) -> str:
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    pieces = []
    for case in [
        "gaussian_rank10",
        "gaussian_rank80",
        "spectral_default",
        "spectral_harder",
    ]:
        values = [float(row["expected_steps"]) for row in rows if row["case"] == case]
        if values:
            pieces.append(f"{case}:{sum(values) / len(values):.1f}")
    return "; ".join(pieces)


def policy_metric_summary(path: Path, column: str, digits: int) -> str:
    if not path.exists():
        return ""
    rows = list(csv.DictReader(path.open()))
    if not rows or column not in rows[0]:
        return ""
    pieces = []
    for case in [
        "gaussian_rank10",
        "gaussian_rank80",
        "spectral_default",
        "spectral_harder",
    ]:
        values = [
            float(row[column])
            for row in rows
            if row["case"] == case and row.get(column) not in {"", None}
        ]
        if values:
            pieces.append(f"{case}:{sum(values) / len(values):.{digits}f}")
    return "; ".join(pieces)


def collect_prefixes(curve_dir: Path) -> list[tuple[str, str]]:
    prefixes = set()
    for path in curve_dir.glob("rowcol_svt_*_rank10_size300_summary.csv"):
        prefix = path.name.removesuffix("_rank10_size300_summary.csv")
        if prefix.startswith("rowcol_svt_adaptive_mixer_"):
            group = "adaptive_mixer"
            tag = prefix.removeprefix("rowcol_svt_adaptive_mixer_")
        elif prefix.startswith("rowcol_svt_spectral_mixer_"):
            group = "spectral_features"
            tag = prefix.removeprefix("rowcol_svt_spectral_mixer_")
        elif prefix.startswith("rowcol_svt_depth_distill_"):
            group = "depth_distill"
            tag = prefix.removeprefix("rowcol_svt_depth_distill_")
        elif prefix.startswith("rowcol_svt_spectrum_encoder_"):
            group = "spectrum_encoder"
            tag = prefix.removeprefix("rowcol_svt_spectrum_encoder_")
        elif prefix.startswith("rowcol_svt_adaptive_threshold_"):
            group = "adaptive_threshold"
            tag = prefix.removeprefix("rowcol_svt_adaptive_threshold_")
        elif prefix.startswith("rowcol_svt_learned_cv_"):
            group = "learned_cv"
            tag = prefix.removeprefix("rowcol_svt_learned_cv_")
        elif prefix.startswith("rowcol_svt_cv_mixer_"):
            group = "cv_policy"
            tag = prefix.removeprefix("rowcol_svt_cv_mixer_")
        else:
            continue
        prefixes.add((group, tag))
    return sorted(prefixes)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root
    curve_dir = runs_root / "curriculum_eval"
    spectral_dir = runs_root / "figures_by_setup/05_spectral_tails_rowcol_svt/attempts"
    queue_dir = runs_root / "objective_queues"
    rows = []
    for group, tag in collect_prefixes(curve_dir):
        curve_prefix = {
            "adaptive_mixer": "rowcol_svt_adaptive_mixer",
            "spectral_features": "rowcol_svt_spectral_mixer",
            "depth_distill": "rowcol_svt_depth_distill",
            "spectrum_encoder": "rowcol_svt_spectrum_encoder",
            "adaptive_threshold": "rowcol_svt_adaptive_threshold",
            "learned_cv": "rowcol_svt_learned_cv",
            "cv_policy": "rowcol_svt_cv_mixer",
        }[group]
        spectral_prefix = {
            "adaptive_mixer": "spectral_adaptive_mixer",
            "spectral_features": "spectral_svt_spectral_mixer",
            "depth_distill": "spectral_svt_depth_distill",
            "spectrum_encoder": "spectral_svt_spectrum_encoder",
            "adaptive_threshold": "spectral_svt_adaptive_threshold",
            "learned_cv": "spectral_svt_learned_cv",
            "cv_policy": "spectral_svt_cv_mixer",
        }[group]
        policy_prefix = {
            "adaptive_mixer": "rowcol_svt_adaptive_mixer",
            "spectral_features": "rowcol_svt_spectral_mixer",
            "depth_distill": "rowcol_svt_depth_distill",
            "spectrum_encoder": "rowcol_svt_spectrum_encoder",
            "adaptive_threshold": "rowcol_svt_adaptive_threshold",
            "learned_cv": "rowcol_svt_learned_cv",
            "cv_policy": "rowcol_svt_cv_mixer",
        }[group]
        rank10 = curve_dir / f"{curve_prefix}_{tag}_rank10_size300_summary.csv"
        rank80 = curve_dir / f"{curve_prefix}_{tag}_rank80_size300_summary.csv"
        default = spectral_dir / f"{spectral_prefix}_{tag}_default_300_500.summary.csv"
        harder = spectral_dir / f"{spectral_prefix}_{tag}_harder_300_500.summary.csv"
        policy = queue_dir / f"{policy_prefix}_{tag}_policy.csv"
        mnar = curve_dir / f"{curve_prefix}_{tag}_mnar_size300.summary.csv"
        rows.append(
            {
                "group": group,
                "tag": tag,
                "rank10_rmse": model_rmse(rank10),
                "rank10_softimpute": softimpute_rmse(rank10),
                "rank80_rmse": model_rmse(rank80),
                "rank80_softimpute": softimpute_rmse(rank80),
                "spectral300": spectral_rmse(default, 300),
                "spectral500": spectral_rmse(default, 500),
                "harder300": spectral_rmse(harder, 300),
                "harder500": spectral_rmse(harder, 500),
                "mnar_latent_r10": mnar_rmse(
                    mnar,
                    "mnar_latent_factor",
                    10,
                    softimpute=False,
                ),
                "mnar_latent_r10_softimpute": mnar_rmse(
                    mnar,
                    "mnar_latent_factor",
                    10,
                    softimpute=True,
                ),
                "mnar_latent_r80": mnar_rmse(
                    mnar,
                    "mnar_latent_factor",
                    80,
                    softimpute=False,
                ),
                "mnar_latent_r80_softimpute": mnar_rmse(
                    mnar,
                    "mnar_latent_factor",
                    80,
                    softimpute=True,
                ),
                "mnar_spatial_r80": mnar_rmse(
                    mnar,
                    "mnar_spatial_block",
                    80,
                    softimpute=False,
                ),
                "mnar_spatial_r80_softimpute": mnar_rmse(
                    mnar,
                    "mnar_spatial_block",
                    80,
                    softimpute=True,
                ),
                "mnar_censor_r10": mnar_rmse(
                    mnar,
                    "mnar_censoring",
                    10,
                    softimpute=False,
                ),
                "mnar_censor_r10_softimpute": mnar_rmse(
                    mnar,
                    "mnar_censoring",
                    10,
                    softimpute=True,
                ),
                "mnar_cluster_r80": mnar_rmse(
                    mnar,
                    "mnar_cluster_level",
                    80,
                    softimpute=False,
                ),
                "mnar_cluster_r80_softimpute": mnar_rmse(
                    mnar,
                    "mnar_cluster_level",
                    80,
                    softimpute=True,
                ),
                "expected_steps": policy_summary(policy),
                "threshold_scale": policy_metric_summary(
                    policy,
                    "threshold_scale",
                    3,
                ),
                "oracle_steps": policy_metric_summary(policy, "oracle_step", 1),
                "oracle_rmse_norm": policy_metric_summary(
                    policy,
                    "oracle_rmse_norm",
                    5,
                ),
                "mixture_oracle_gap_norm": policy_metric_summary(
                    policy,
                    "mixture_oracle_gap_norm",
                    5,
                ),
                "tau_oracle_scale": policy_metric_summary(
                    policy,
                    "tau_oracle_scale",
                    3,
                ),
                "tau_oracle_steps": policy_metric_summary(
                    policy,
                    "tau_oracle_step",
                    1,
                ),
                "tau_oracle_rmse_norm": policy_metric_summary(
                    policy,
                    "tau_oracle_rmse_norm",
                    5,
                ),
                "mixture_tau_oracle_gap_norm": policy_metric_summary(
                    policy,
                    "mixture_tau_oracle_gap_norm",
                    5,
                ),
            }
        )

    fields = [
        "group",
        "tag",
        "rank10_rmse",
        "rank10_softimpute",
        "rank80_rmse",
        "rank80_softimpute",
        "spectral300",
        "spectral500",
        "harder300",
        "harder500",
        "mnar_latent_r10",
        "mnar_latent_r10_softimpute",
        "mnar_latent_r80",
        "mnar_latent_r80_softimpute",
        "mnar_spatial_r80",
        "mnar_spatial_r80_softimpute",
        "mnar_censor_r10",
        "mnar_censor_r10_softimpute",
        "mnar_cluster_r80",
        "mnar_cluster_r80_softimpute",
        "expected_steps",
        "threshold_scale",
        "oracle_steps",
        "oracle_rmse_norm",
        "mixture_oracle_gap_norm",
        "tau_oracle_scale",
        "tau_oracle_steps",
        "tau_oracle_rmse_norm",
        "mixture_tau_oracle_gap_norm",
    ]
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with args.output_tsv.open("r", encoding="utf-8") as file:
        print(file.read(), end="")


if __name__ == "__main__":
    main()
