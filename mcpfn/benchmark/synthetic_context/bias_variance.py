"""Bias/variance decomposition: is the boost debiasing MatFormer or denoising it?

For each (setting, seed):
  - Run MatFormer alone:    pred_mf[r, c]
  - Run boost:              pred_boost[r, c] = pred_mf + alpha * r_pred
  - True target:            M[r, c]  (clean signal, NOT Y)

Over all missing cells (pooled across seeds within a setting):
  e_mf       = pred_mf    - M       (signed error of MatFormer)
  e_boost    = pred_boost - M
  r_pred     = (pred_boost - pred_mf) / alpha   (the xRFM residual prediction)

Compute:
  bias       = mean(e)            global signed bias
  var        = var(e)             spread of signed error
  mse        = mean(e^2) = bias^2 + var
  cov(e_mf, r_pred)               negative => boost moves opposite to mf error
  corr(e_mf, r_pred)
  rmse       = sqrt(mse)

Decomposition of MSE delta:
  MSE_boost - MSE_mf =  2*alpha*cov(e_mf, r_pred) + alpha^2*E[r_pred^2]
                        \_______ debias _______/   \____ variance ____/
  Negative cov => debias term reduces MSE.

Also: optimal alpha* per setting = -cov(e_mf, r_pred) / E[r_pred^2].
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/root/tabular/mcpfn/src')
sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from generate import ALPHA_SETTINGS, make_draw, observed_matrix  # noqa: E402
from tabimpute.interface_rfm import BoostMatFormerImputer  # noqa: E402

ROUND2_T2_CONFIG = {'config': {
    'embedding_size': 768, 'num_attention_heads': 16, 'mlp_hidden_size': 1536,
    'num_layers': 8, 'num_outputs': 5000, 'num_cls': 12, 'use_rope': True,
    'rope_base': 10000.0, 'rope_fraction': 0.5,
    'use_absolute_positional_embeddings': False, 'positional_damping_factor': 0.1,
    'attention_dropout': 0.0, 'ffn_dropout': 0.05, 'drop_path_rate': 0.01,
    'residual_scale_init': 0.5, 'embedding_dropout': 0.0, 'rms_norm_eps': 1e-6,
}}
CHECKPOINT_PATH = '/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth'


def main():
    import torch
    from tabimpute.interface import ImputePFN

    N = T = 200
    n_seeds = 5
    alpha = 0.1
    settings = list(ALPHA_SETTINGS.keys())

    print(f"Loading BoostMatFormerImputer (alpha={alpha}) ...", flush=True)
    imputer = BoostMatFormerImputer(
        device='cuda',
        entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG,
        checkpoint_path=CHECKPOINT_PATH,
        max_num_rows=N, max_num_cols=T,
        blend_alpha=alpha,
    )

    rows = []
    t_start = time.time()

    for setting in settings:
        alphas = ALPHA_SETTINGS[setting]
        e_mf_pool = []
        e_boost_pool = []
        r_pred_pool = []

        for seed in range(n_seeds):
            draw = make_draw(N=N, T=T, alphas=alphas, seed=seed)
            Xobs = observed_matrix(draw)
            M_true = draw.M
            mask_missing = draw.mask

            with torch.no_grad():
                X_mf, _ = ImputePFN.impute(imputer, Xobs, return_full=True)
            X_boost = imputer.impute(Xobs)

            X_mf = np.asarray(X_mf, dtype=np.float64)
            X_boost = np.asarray(X_boost, dtype=np.float64)
            M = M_true.astype(np.float64)

            e_mf = (X_mf - M)[mask_missing]
            e_boost = (X_boost - M)[mask_missing]
            r_pred = ((X_boost - X_mf) / alpha)[mask_missing]

            e_mf_pool.append(e_mf)
            e_boost_pool.append(e_boost)
            r_pred_pool.append(r_pred)

            print(f"[{setting} s{seed}] "
                  f"mf rmse={float(np.sqrt(np.mean(e_mf**2))):.4f} "
                  f"boost rmse={float(np.sqrt(np.mean(e_boost**2))):.4f} "
                  f"({time.time()-t_start:.0f}s)",
                  flush=True)

        e_mf = np.concatenate(e_mf_pool)
        e_boost = np.concatenate(e_boost_pool)
        r_pred = np.concatenate(r_pred_pool)

        bias_mf = float(np.mean(e_mf))
        var_mf = float(np.var(e_mf))
        mse_mf = float(np.mean(e_mf**2))

        bias_boost = float(np.mean(e_boost))
        var_boost = float(np.var(e_boost))
        mse_boost = float(np.mean(e_boost**2))

        cov_e_r = float(np.cov(e_mf, r_pred, ddof=0)[0, 1])
        corr_e_r = float(np.corrcoef(e_mf, r_pred)[0, 1])
        E_r2 = float(np.mean(r_pred**2))

        debias_term = 2 * alpha * cov_e_r
        variance_term = alpha**2 * E_r2
        mse_delta_predicted = debias_term + variance_term
        mse_delta_observed = mse_boost - mse_mf

        alpha_star = -cov_e_r / E_r2 if E_r2 > 0 else float('nan')
        mse_at_alpha_star = mse_mf - cov_e_r**2 / E_r2 if E_r2 > 0 else mse_mf

        rows.append({
            'setting': setting,
            'rmse_mf': float(np.sqrt(mse_mf)),
            'rmse_boost': float(np.sqrt(mse_boost)),
            'bias_mf': bias_mf,
            'bias_boost': bias_boost,
            'var_mf': var_mf,
            'var_boost': var_boost,
            'cov_e_r': cov_e_r,
            'corr_e_r': corr_e_r,
            'E_r_pred_sq': E_r2,
            'debias_contribution': debias_term,
            'variance_contribution': variance_term,
            'mse_delta_predicted': mse_delta_predicted,
            'mse_delta_observed': mse_delta_observed,
            'alpha_star': alpha_star,
            'rmse_at_alpha_star': float(np.sqrt(max(0.0, mse_at_alpha_star))),
        })

    df = pd.DataFrame(rows)
    out = Path('/root/tabular/mcpfn/benchmark/synthetic_context/results') / 'bias_variance.csv'
    df.to_csv(out, index=False)
    print()
    print(f"Wrote {out}")
    print()
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
