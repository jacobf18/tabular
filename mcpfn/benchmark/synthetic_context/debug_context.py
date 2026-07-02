"""Quick check: does the context kwarg actually flow through to the residual
learner, by switching residual_learner to 'linear' (ridge) where context
absolutely should help on row_only setting?"""
import sys
import numpy as np

sys.path.insert(0, '/root/tabular/mcpfn/src')
sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from tabimpute.interface_rfm import BoostMatFormerImputer  # noqa: E402
from generate import make_draw, observed_matrix, ALPHA_SETTINGS  # noqa: E402

ROUND2_T2_CONFIG = {'config': {
    'embedding_size': 768, 'num_attention_heads': 16, 'mlp_hidden_size': 1536,
    'num_layers': 8, 'num_outputs': 5000, 'num_cls': 12, 'use_rope': True,
    'rope_base': 10000.0, 'rope_fraction': 0.5,
    'use_absolute_positional_embeddings': False, 'positional_damping_factor': 0.1,
    'attention_dropout': 0.0, 'ffn_dropout': 0.05, 'drop_path_rate': 0.01,
    'residual_scale_init': 0.5, 'embedding_dropout': 0.0, 'rms_norm_eps': 1e-6,
}}
CKPT = '/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth'


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def make_imputer(residual_learner='linear', alpha=0.1):
    return BoostMatFormerImputer(
        device='cuda',
        entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG,
        checkpoint_path=CKPT,
        max_num_rows=200, max_num_cols=200,
        blend_alpha=alpha,
        residual_learner=residual_learner,
    )


print("Loading imputer (linear ridge)...", flush=True)
imp = make_imputer('linear', alpha=1.0)  # alpha=1 so any context signal is undamped

d = make_draw(N=200, T=200, alphas=ALPHA_SETTINGS['row_only'], seed=0)
Xobs = observed_matrix(d)
miss = d.mask
print(f"row_only: x_r drives matrix; missing block is rows {d.N0}-{d.M.shape[0]} x cols {d.T0}-{d.M.shape[1]}")

X_nc = imp.impute(Xobs)
X_rc = imp.impute(Xobs, row_context=d.X_row)
print(f"linear ridge / α=1.0  / row_only:")
print(f"  no context:    miss-RMSE = {rmse(X_nc[miss], d.M[miss]):.4f}")
print(f"  + row context: miss-RMSE = {rmse(X_rc[miss], d.M[miss]):.4f}")
print(f"  raw diff:      {np.abs(X_nc - X_rc).max():.4e}")

print("\nNow same test with xRFM:", flush=True)
imp2 = make_imputer('xrfm', alpha=1.0)
X_nc2 = imp2.impute(Xobs)
X_rc2 = imp2.impute(Xobs, row_context=d.X_row)
# Shuffled context (control: should hurt if context is genuinely used)
rng = np.random.default_rng(123)
X_row_shuffled = d.X_row[rng.permutation(d.X_row.shape[0])]
X_sc2 = imp2.impute(Xobs, row_context=X_row_shuffled)
print(f"  no context:        miss-RMSE = {rmse(X_nc2[miss], d.M[miss]):.4f}")
print(f"  + row context:     miss-RMSE = {rmse(X_rc2[miss], d.M[miss]):.4f}")
print(f"  + shuffled ctx:    miss-RMSE = {rmse(X_sc2[miss], d.M[miss]):.4f}")
print(f"  raw_ctx vs no:     {np.abs(X_nc2 - X_rc2).max():.4e}")
print(f"  shuffled vs no:    {np.abs(X_nc2 - X_sc2).max():.4e}")

# Larger contextual effect — try smaller matrix with fewer raw features.
print("\nSmaller matrix (N=T=50, fewer raw features):", flush=True)
imp3 = make_imputer('xrfm', alpha=1.0)
d2 = make_draw(N=50, T=50, alphas=ALPHA_SETTINGS['row_only'], seed=0,
               noise_std=0.5)
Xobs2 = observed_matrix(d2)
miss2 = d2.mask
X_nc3 = imp3.impute(Xobs2)
X_rc3 = imp3.impute(Xobs2, row_context=d2.X_row)
rng2 = np.random.default_rng(123)
X_row_sh = d2.X_row[rng2.permutation(d2.X_row.shape[0])]
X_sc3 = imp3.impute(Xobs2, row_context=X_row_sh)
print(f"  no context:        miss-RMSE = {rmse(X_nc3[miss2], d2.M[miss2]):.4f}")
print(f"  + row context:     miss-RMSE = {rmse(X_rc3[miss2], d2.M[miss2]):.4f}")
print(f"  + shuffled ctx:    miss-RMSE = {rmse(X_sc3[miss2], d2.M[miss2]):.4f}")
