#!/usr/bin/env python3
"""Batch driver for the 5 paper-canonical CPS placebo combos.

Pipeline per combo: create benchmark -> eval imputers -> eval SDID/SC/DiD
                    -> summarize -> plot.

For combos with no R-generated source folder, reports a clear 'missing' line
so you know which `export_sdid_cps_draws.R` runs are still required.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAPER_COMBOS: list[tuple[str, str]] = [
    ("log_wage", "min_wage"),
    ("log_wage", "open_carry"),
    ("log_wage", "abort_ban"),
    ("hours", "min_wage"),
    ("urate", "min_wage"),
]

STAGES_ALL = ("create", "eval", "sdid", "summarize", "plot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combos", type=str, default="",
                   help="Comma-separated 'outcome:assignment' pairs (default: all 5).")
    p.add_argument("--skip", type=str, default="",
                   help=f"Comma-separated stages to skip. Choose from: {','.join(STAGES_ALL)}.")
    p.add_argument("--eval-methods", type=str,
                   default="knn,softimpute,mice,tabimpute_all,"
                           "round2_t2-rank_1_30,round2_t2-rank_10_10,"
                           "round2_t2-rank_15_15_from_10_10",
                   help="Methods passed to eval_cps_causal.py.")
    p.add_argument("--heavy-methods", type=str, default="",
                   help="Heavy methods (runs separately with --limit heavy-limit).")
    p.add_argument("--heavy-limit", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def source_dir(outcome: str, assignment: str) -> Path:
    return HERE / f"sdid_cps_{outcome}_{assignment}_200"


def benchmark_dir(outcome: str, assignment: str) -> Path:
    return HERE / "causal_eval" / f"{outcome}_{assignment}"


def figures_dir(outcome: str, assignment: str) -> Path:
    return HERE / "figures" / f"{outcome}_{assignment}"


def run(cmd: list[str], label: str) -> int:
    print(f"\n[{label}] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=HERE).returncode


def do_combo(
    outcome: str,
    assignment: str,
    stages: set[str],
    args: argparse.Namespace,
) -> bool:
    tag = f"{outcome}_{assignment}"
    src = source_dir(outcome, assignment)
    bench = benchmark_dir(outcome, assignment)
    figs = figures_dir(outcome, assignment)

    print(f"\n========== combo: {tag} ==========")
    if not src.exists():
        print(f"  MISSING source folder: {src.name}  "
              f"-> run export_sdid_cps_draws.R with outcome_name='{outcome}', "
              f"label_name='{assignment}' and rerun this driver.")
        return False

    if "create" in stages:
        rc = run([
            sys.executable, "create_cps_causal_benchmark.py",
            "--input-dir", str(src),
            "--output-dir", str(bench),
            "--force",
        ], label=f"{tag}/create")
        if rc != 0:
            return False

    if "eval" in stages:
        cmd = [
            sys.executable, "eval_cps_causal.py",
            "--benchmark-dir", str(bench),
            "--methods", args.eval_methods,
            "--device", args.device,
        ]
        if args.force_rerun:
            cmd.append("--force-rerun")
        rc = run(cmd, label=f"{tag}/eval")
        if rc != 0:
            return False

        if args.heavy_methods:
            cmd_heavy = [
                sys.executable, "eval_cps_causal.py",
                "--benchmark-dir", str(bench),
                "--methods", args.heavy_methods,
                "--device", args.device,
                "--limit", str(args.heavy_limit),
            ]
            if args.force_rerun:
                cmd_heavy.append("--force-rerun")
            rc = run(cmd_heavy, label=f"{tag}/eval-heavy")
            if rc != 0:
                return False

    if "sdid" in stages:
        cmd = [
            sys.executable, "run_sdid_baselines.py",
            "--benchmark-dir", str(bench),
        ]
        if args.force_rerun:
            cmd.append("--force-rerun")
        rc = run(cmd, label=f"{tag}/sdid")
        if rc != 0:
            return False

    if "summarize" in stages:
        rc = run([
            sys.executable, "summarize_cps_causal.py",
            "--results-dir", str(bench / "results"),
        ], label=f"{tag}/summarize")
        if rc != 0:
            return False

    if "plot" in stages:
        rc = run([
            sys.executable, "plot_cps.py",
            "--results-dir", str(bench / "results"),
            "--figures-dir", str(figs),
        ], label=f"{tag}/plot")
        if rc != 0:
            return False

    return True


def main() -> None:
    args = parse_args()

    if args.combos.strip():
        combos = []
        for tok in args.combos.split(","):
            if ":" not in tok:
                raise SystemExit(f"Bad combo '{tok}', expected 'outcome:assignment'.")
            o, a = tok.split(":", 1)
            combos.append((o.strip(), a.strip()))
    else:
        combos = list(PAPER_COMBOS)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    bad_skip = skip - set(STAGES_ALL)
    if bad_skip:
        raise SystemExit(f"Unknown skip stages: {bad_skip}. Known: {STAGES_ALL}")
    stages = set(STAGES_ALL) - skip

    done: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    for outcome, assignment in combos:
        ok = do_combo(outcome, assignment, stages, args)
        tag = f"{outcome}:{assignment}"
        if ok:
            done.append(tag)
        elif not source_dir(outcome, assignment).exists():
            missing.append(tag)
        else:
            failed.append(tag)
            if args.stop_on_error:
                break

    print("\n\n========== batch summary ==========")
    print(f" done:    {done}")
    print(f" missing: {missing}  (need R: export_sdid_cps_draws.R)")
    print(f" failed:  {failed}")


if __name__ == "__main__":
    main()
