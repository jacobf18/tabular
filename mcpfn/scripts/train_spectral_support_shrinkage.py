"""Isolated training-only augmentation experiment; public inference stays unchanged."""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import train_direct_rowcol_svt_hybrid_random_rank_curriculum as trainer
from spectral_support_normalization import apply_shrunk_support_mask


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--support-variance-df", type=float, required=True)
    args, rest = parser.parse_known_args()
    if args.support_variance_df <= 0:
        raise ValueError("support-variance-df must be positive")
    metadata_parser = argparse.ArgumentParser(add_help=False)
    metadata_parser.add_argument("--work-dir", type=Path)
    metadata_parser.add_argument("--run-name", required=True)
    metadata, _ = metadata_parser.parse_known_args(rest)
    work = metadata.work_dir or (Path(__file__).resolve().parents[1] /
                                 "src/tabimpute/workdir" / metadata.run_name)
    work.mkdir(parents=True, exist_ok=True)
    (work / "support_normalization.json").write_text(json.dumps({
        "variance_df": args.support_variance_df, "scope": "training sparse-support augmentation only",
        "validation_and_interface": "unchanged", "statistics": "remaining observed entries only"}, indent=2))
    trainer.base.apply_support_mask = partial(apply_shrunk_support_mask, variance_df=args.support_variance_df)
    sys.argv = [sys.argv[0], *rest]
    trainer.base.main()


if __name__ == "__main__":
    main()
