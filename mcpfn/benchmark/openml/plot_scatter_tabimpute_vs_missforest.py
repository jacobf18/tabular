import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_options import (
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    NEUTRAL_COLOR_DARK,
    FIGURE_SIZES,
    compute_normalized_rmse_columnwise,
)

matplotlib.rcParams["font.family"] = "serif"

base_path = "../datasets/openml"
figure_base_path = "./figures"

patterns = {
    "MCAR",
    "MAR",
    "MNAR",
    "MAR_Neural",
    "MAR_BlockNeural",
    "MNARPolarizationPattern",
    "MNARSoftPolarizationPattern",
    "MNARLatentFactorPattern",
    "MNARClusterLevelPattern",
    "MNARTwoPhaseSubsetPattern",
    "MNARCensoringPattern",
}

method_files = {
    "TabImpute": "masters_mcar",
    "MissForest": "missforest",
}

datasets = os.listdir(base_path)

records = []
for dataset in datasets:
    dataset_dir = f"{base_path}/{dataset}"
    if not os.path.isdir(dataset_dir):
        continue
    for config in os.listdir(dataset_dir):
        if "repeats" in config:
            continue
        parts = config.split("_")
        p = parts[-1]
        if p != str(0.4):
            continue
        pattern_name = "_".join(parts[:-1])
        if pattern_name not in patterns:
            continue

        config_dir = f"{dataset_dir}/{pattern_name}_{p}"
        try:
            X_missing = np.load(f"{config_dir}/missing.npy")
            X_true = np.load(f"{config_dir}/true.npy")
        except FileNotFoundError:
            continue
        mask = np.isnan(X_missing)

        rmse_per_method = {}
        ok = True
        for label, fname in method_files.items():
            path = f"{config_dir}/{fname}.npy"
            if not os.path.exists(path):
                ok = False
                break
            X_imputed = np.load(path)
            rmse_per_method[label] = compute_normalized_rmse_columnwise(
                X_true, X_imputed, mask
            )
        if not ok:
            continue

        records.append(
            {
                "dataset": dataset,
                "pattern": pattern_name,
                "TabImpute": rmse_per_method["TabImpute"],
                "MissForest": rmse_per_method["MissForest"],
            }
        )

df = pd.DataFrame.from_records(records)

x = df["TabImpute"].to_numpy()
y = df["MissForest"].to_numpy()
finite = np.isfinite(x) & np.isfinite(y)
df = df[finite].reset_index(drop=True)
x, y = x[finite], y[finite]

tabimpute_wins = int((x < y).sum())
total = len(x)
win_pct = 100.0 * tabimpute_wins / total if total else 0.0

try:
    _w_stat, p_wilcoxon = stats.wilcoxon(
        x, y, alternative="less", zero_method="pratt"
    )
except ValueError:
    p_wilcoxon = float("nan")

if np.isnan(p_wilcoxon):
    p_str = "p=n/a"
elif p_wilcoxon < 1e-4:
    p_str = "p<1e-4"
else:
    p_str = f"p={p_wilcoxon:.2e}"

lo = 0.0
hi = 20.0

fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])

ax.fill_between(
    [lo, hi],
    [lo, hi],
    [hi, hi],
    color=HIGHLIGHT_COLOR,
    alpha=0.10,
    linewidth=0,
    zorder=0,
)
# ax.text(
#     lo + 0.05 * (hi - lo),
#     hi - 0.05 * (hi - lo),
#     "TabImpute better",
#     color=HIGHLIGHT_COLOR,
#     fontsize=12,
#     ha="left",
#     va="top",
#     style="italic",
#     zorder=1,
# )

# _arrow_tail = (hi - 0.10 * (hi - lo), lo + 0.10 * (hi - lo))
# _arrow_head = (lo + 0.10 * (hi - lo), hi - 0.10 * (hi - lo))
# ax.annotate(
#     "",
#     xy=_arrow_head,
#     xytext=_arrow_tail,
#     arrowprops=dict(
#         arrowstyle="->",
#         color=HIGHLIGHT_COLOR,
#         lw=2.0,
#         shrinkA=0,
#         shrinkB=0,
#     ),
#     zorder=2,
# )

ax.scatter(
    x,
    y,
    s=40,
    color=HIGHLIGHT_COLOR,
    marker="x",
    linewidth=1.2,
    alpha=0.8,
    zorder=3,
)
ax.plot([lo, hi], [lo, hi], color=NEUTRAL_COLOR, linestyle="--", linewidth=1.2, zorder=1)

_pattern_short = {
    "MCAR": "MCAR",
    "MAR": "Col-MAR",
    "MNAR": "Self-MNAR",
    "MAR_Neural": "NN-MNAR",
    "MAR_BlockNeural": "Block-MNAR",
    "MNARPolarizationPattern": "Polar",
    "MNARSoftPolarizationPattern": "Soft-Pol",
    "MNARLatentFactorPattern": "Latent-MNAR",
    "MNARClusterLevelPattern": "Clustering",
    "MNARTwoPhaseSubsetPattern": "Two-Phase",
    "MNARCensoringPattern": "Censoring",
}

_visible = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
_gap = np.where(_visible, np.abs(y - x), -np.inf)
_outlier_idx = np.argsort(_gap)[-6:][::-1]
for _i in _outlier_idx:
    if not np.isfinite(_gap[_i]) or _gap[_i] <= 0:
        continue
    _pat = df.iloc[int(_i)]["pattern"]
    _label = _pattern_short.get(_pat, _pat)
    ax.annotate(
        _label,
        xy=(x[_i], y[_i]),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=10,
        color=NEUTRAL_COLOR_DARK,
        family="monospace",
        zorder=4,
    )

# Turn off right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("TabImpute (RMSE)", fontsize=18)
ax.set_ylabel("MissForest (RMSE)", fontsize=18)
ax.set_title(
    f"TabImpute vs. MissForest\nTabImpute better in {tabimpute_wins}/{total} = {win_pct:.1f}%; Wilcoxon {p_str}",
    fontsize=15,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=12)

fig.tight_layout()
fig.savefig(
    f"{figure_base_path}/scatter_tabimpute_vs_missforest.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(
    f"Scatter saved: {figure_base_path}/scatter_tabimpute_vs_missforest.pdf "
    f"({tabimpute_wins}/{total} TabImpute wins; {win_pct:.1f}%)"
)
