"""Causal-benchmark MatFormer + xRFM-residual boost — corrected version.

The v1 script used pre-computed `dwip-v2-*.npy` files as the MatFormer base.
That broke because those files store the FINAL imputed matrix (observed cells
preserved exactly), so MatFormer-vs-observed residuals at observed cells were
all zero — xRFM trained on zeros and predicted zeros.

This v2 uses BoostMatFormerImputer directly: it loads a backbone checkpoint
and uses ImputePFN.impute(return_full=True) to get MatFormer's RAW predictions
at every cell (not preserved). Same approach as the UCI boost result.

Usage:
    python causal_boost_v2.py --benchmark all
    python causal_boost_v2.py --benchmark CPS --limit 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/root/tabular/mcpfn/src')
from tabimpute.interface_rfm import BoostMatFormerImputer  # noqa: E402


# Architecture (same as dwip_v2-trial_002, which is the UCI MatFormer base).
ROUND2_T2_CONFIG = {'config': {
    'embedding_size': 768, 'num_attention_heads': 16, 'mlp_hidden_size': 1536,
    'num_layers': 8, 'num_outputs': 5000, 'num_cls': 12, 'use_rope': True,
    'rope_base': 10000.0, 'rope_fraction': 0.5,
    'use_absolute_positional_embeddings': False, 'positional_damping_factor': 0.1,
    'attention_dropout': 0.0, 'ffn_dropout': 0.05, 'drop_path_rate': 0.01,
    'residual_scale_init': 0.5, 'embedding_dropout': 0.0, 'rms_norm_eps': 1e-6,
}}
CHECKPOINT_PATH = '/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth'

BENCHMARK_DIR = Path("/root/tabular/mcpfn/benchmark")


def find_draws(benchmark: str) -> list[Path]:
    base = BENCHMARK_DIR / benchmark / "causal_eval"
    return sorted([p for p in base.rglob("draw_*") if p.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["CPS", "kansas", "prop99", "all"],
                    default="all")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out_basename", type=str,
                    default="boost_matformer_dwip_v2_t002_a0p1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=50)
    ap.add_argument("--max_cols", type=int, default=50)
    ap.add_argument("--adaptive_alpha", action="store_true",
                    help="Per-matrix α tuning (overrides --alpha for selection).")
    ap.add_argument("--alpha_val_frac", type=float, default=0.1)
    ap.add_argument("--adaptive_alpha_threshold", type=float, default=0.05)
    ap.add_argument("--adaptive_alpha_method", choices=["grid", "bias_variance"],
                    default="grid")
    ap.add_argument("--alpha_shrinkage", type=float, default=0.5,
                    help="Multiply α* by this before applying. <1 = robust to "
                         "val→test distribution shift.")
    ap.add_argument("--alpha_max_cap", type=float, default=0.1,
                    help="Clip α to this maximum after shrinkage.")
    ap.add_argument("--benchmarks_filter", nargs="*", default=None,
                    help="Optional whitelist of subdir names under causal_eval/ "
                         "(e.g. hours_min_wage log_wage_open_carry).")
    args = ap.parse_args()

    print(
        f"Loading BoostMatFormerImputer "
        f"{'(adaptive α)' if args.adaptive_alpha else f'(α={args.alpha})'}, "
        f"checkpoint={CHECKPOINT_PATH}", flush=True
    )
    imputer = BoostMatFormerImputer(
        device='cuda',
        entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG,
        checkpoint_path=CHECKPOINT_PATH,
        max_num_rows=args.max_rows,
        max_num_cols=args.max_cols,
        blend_alpha=args.alpha,
        adaptive_alpha=args.adaptive_alpha,
        alpha_val_frac=args.alpha_val_frac,
        adaptive_alpha_threshold=args.adaptive_alpha_threshold,
        adaptive_alpha_method=args.adaptive_alpha_method,
        alpha_shrinkage=args.alpha_shrinkage,
        alpha_max_cap=args.alpha_max_cap,
    )

    benchmarks = ["CPS", "kansas", "prop99"] if args.benchmark == "all" else [args.benchmark]
    total_ok = total_skip = total_err = 0

    for bm in benchmarks:
        draws = find_draws(bm)
        if args.benchmarks_filter:
            wl = set(args.benchmarks_filter)
            draws = [d for d in draws if any(part in wl for part in d.parts)]
        if args.limit > 0:
            draws = draws[: args.limit]
        print(f"\n[{bm}] {len(draws)} draws", flush=True)

        t0 = time.time()
        ok = sk = er = 0
        for i, d in enumerate(draws, 1):
            observed_p = d / "observed.npy"
            out_p = d / f"{args.out_basename}.npy"
            if not observed_p.is_file():
                sk += 1
                continue
            if out_p.exists() and not args.force:
                sk += 1
                continue

            X = np.load(observed_p).astype(np.float32)
            try:
                X_hat = imputer.impute(X)
            except Exception as e:
                er += 1
                print(f"  [{i}/{len(draws)}] {d.name}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                continue
            np.save(out_p, X_hat.astype(np.float32))
            ok += 1

            if i % 50 == 0:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                eta = (len(draws) - i) / rate if rate > 0 else 0
                print(f"  [{bm}] {i}/{len(draws)} ok={ok} skip={sk} err={er}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)
        dt = time.time() - t0
        print(f"[{bm}] DONE in {dt/60:.1f} min  ok={ok} skip={sk} err={er}",
              flush=True)
        total_ok += ok; total_skip += sk; total_err += er

    print(f"\n[total] ok={total_ok}  skip={total_skip}  err={total_err}")


if __name__ == "__main__":
    main()
