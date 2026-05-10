import os
import numpy as np

base_path = "../datasets/openml"
datasets = sorted(os.listdir(base_path))

PATTERNS = ["MCAR", "MAR", "MNAR"]
P = "0.4"

per_dataset = []
for dataset in datasets:
    rows_for_dataset = []
    for pattern in PATTERNS:
        cfg_dir = f"{base_path}/{dataset}/{pattern}_{P}"
        diff_path = f"{cfg_dir}/diffputer.npy"
        miss_path = f"{cfg_dir}/missing.npy"
        true_path = f"{cfg_dir}/true.npy"
        if not (os.path.isfile(diff_path) and os.path.isfile(miss_path) and os.path.isfile(true_path)):
            continue
        X_imp = np.load(diff_path)
        X_miss = np.load(miss_path)
        X_true = np.load(true_path)
        mask = np.isnan(X_miss)

        n_cols = X_true.shape[1]
        col_stats = []
        for j in range(n_cols):
            col_mask = mask[:, j]
            if not np.any(col_mask):
                continue
            obs_mean = np.nanmean(X_miss[:, j])
            obs_std = np.nanstd(X_miss[:, j])
            imp_vals = X_imp[col_mask, j]
            imp_mean = np.mean(imp_vals)
            imp_std = np.std(imp_vals)

            denom = obs_std if obs_std > 1e-9 else 1.0
            mean_diff_norm = abs(imp_mean - obs_mean) / denom
            std_ratio = imp_std / denom
            col_stats.append((mean_diff_norm, std_ratio))

        if not col_stats:
            continue
        mean_diff_norm_avg = float(np.mean([c[0] for c in col_stats]))
        std_ratio_avg = float(np.mean([c[1] for c in col_stats]))
        rows_for_dataset.append((pattern, mean_diff_norm_avg, std_ratio_avg, len(col_stats)))

    if rows_for_dataset:
        per_dataset.append((dataset, rows_for_dataset))

# Summary: flag datasets that look degenerate.
# Heuristic: imputed std < 5% of observed-col std AND imputed mean ~ observed-col mean.
print(f"{'dataset':40s}  {'pat':>5s}  {'|imp_mean - obs_mean|/obs_std':>30s}  {'imp_std/obs_std':>16s}  {'ncols':>5s}  flag")
print("-" * 110)
n_degen = 0
n_total = 0
for dataset, rows in per_dataset:
    for pattern, m, s, k in rows:
        n_total += 1
        flag = "DEGENERATE" if (s < 0.05 and m < 0.10) else ""
        if flag:
            n_degen += 1
        print(f"{dataset:40.40s}  {pattern:>5s}  {m:>30.4f}  {s:>16.4f}  {k:>5d}  {flag}")

print()
print(f"Degenerate (collapsed to col mean): {n_degen}/{n_total}  ({(100.0*n_degen/n_total) if n_total else 0:.1f}%)")
