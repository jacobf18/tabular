from collections import defaultdict
from typing import Any

import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _is_categorical_column(df: pd.DataFrame, col_idx: int) -> bool:
    """Heuristic: categorical = object/category dtype or non-numeric."""
    col = df.iloc[:, col_idx]
    return df.dtypes.iloc[col_idx] in ["object", "category"] or not pd.api.types.is_numeric_dtype(
        col
    )


def _n_classes_ground_truth(df: pd.DataFrame, col_idx: int) -> int:
    """Distinct non-null label count in the full reference frame (same as class-counts script)."""
    s = pd.Series(df.iloc[:, col_idx]).astype(object)
    return int(s.dropna().nunique())


# Only aggregate metrics over categorical columns with fewer than this many classes.
MAX_CLASSES_EXCLUSIVE = 5


def _masked_column_as_object(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    col_idx: int,
    col_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract masked values as numpy object arrays.

    Comparing two pandas ``Categorical`` arrays with ``==`` requires identical
    ``categories``; ground truth and imputed frames often differ after IO. Casting
    through ``object`` compares the actual level values.
    """
    y_true = (
        pd.Series(df_true.iloc[:, col_idx].values[col_mask]).astype(object).to_numpy()
    )
    y_pred = (
        pd.Series(df_pred.iloc[:, col_idx].values[col_mask]).astype(object).to_numpy()
    )
    return y_true, y_pred


base_path = "../datasets/openml_categorical"

datasets = os.listdir(base_path)

methods = [
    # "softimpute", 
    # "column_mean", 
    "hyperimpute",
    # "ot_sinkhorn",
    "missforest",
    # "ice",
    # "mice",
    # "gain",
    # "miwae",
    "mcpfn",
    "tabimpute_round2_t2",
    "tabimpute_round2_t2_adapter",
    "mode",
    # "tabpfn",
    # "tabpfn_impute",
    # "knn",
    # "forestdiffusion",
    # "diffputer",
    # "remasker",
]

method_names = {
    "mixed_nonlinear": "TabImpute (Nonlinear FM)",
    "mcpfn_ensemble": "TabImpute+",
    "mcpfn_mixed_fixed": "TabImpute (Fixed)",
    "masters_mcar": "TabImpute",
    "masters_mar": "TabImpute (MAR)",
    "masters_mnar": "TabImpute (Self-Masking-MNAR)",
    "masters_mcar_nonlinear": "TabImpute (Nonlinear)",
    "tabimpute_ensemble": "TabImpute Ensemble",
    "tabimpute_ensemble_router": "TabImpute Router",
    "mcpfn_mixed_adaptive": "TabImpute",
    "mcpfn_mar_linear": "TabImpute (MCAR then MAR)",
    "mixed_more_heads": "TabImpute (More Heads)",
    "tabpfn_no_proprocessing": "TabPFN Fine-Tuned No Preprocessing",
    # "mixed_perm_both_row_col": "TabImpute",
    "tabpfn_impute": "TabPFN",
    "tabpfn": "EWF-TabPFN",
    "column_mean": "Col Mean",
    "hyperimpute": "HyperImpute",
    "ot_sinkhorn": "OT",
    "missforest": "MissForest",
    "softimpute": "SoftImpute",
    "ice": "ICE",
    "mice": "MICE",
    "gain": "GAIN",
    "miwae": "MIWAE",
    "forestdiffusion": "ForestDiffusion",
    "knn": "K-Nearest Neighbors",
    "diffputer": "DiffPuter",
    "remasker": "ReMasker",
}

# Define consistent color mapping for methods (using display names as they appear in the DataFrame)
method_colors = {
    "TabImpute+": "#2f88a8",  # Blue
    "TabImpute": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Ensemble": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Self-Masking-MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Fixed)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MCAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Nonlinear)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Router": "#2f88a8",  # Sea Green (distinct from GPU)
    "EWF-TabPFN": "#3e3b6e",  # 
    "HyperImpute": "#ff7f0e",  # Orange
    "MissForest": "#2ca02c",   # Green
    "OT": "#591942",           # Red
    "Col Mean": "#9467bd",     # Purple
    "SoftImpute": "#8c564b",   # Brown
    "ICE": "#a14d88",          # Pink
    "MICE": "#7f7f7f",         # Gray
    "GAIN": "#286b33",         # Dark Green
    "MIWAE": "#17becf",        # Cyan
    "TabPFN": "#3e3b6e",       # Blue
    "K-Nearest Neighbors": "#a36424",  # Orange
    "ForestDiffusion": "#52b980",      # Medium Green
    "MCPFN": "#ff9896",        # Light Red
    "MCPFN (Linear Permuted)": "#c5b0d5",  # Light Purple
    "MCPFN (Nonlinear Permuted)": "#c49c94",  # Light Brown
    "DiffPuter": "#d62728",  # Red
    "ReMasker": "#f58231",  # Dark Orange
}

# Overall metrics (all datasets, categorical columns, imputed cells only).
# Columns are included only if ground truth has strictly fewer than MAX_CLASSES_EXCLUSIVE
# distinct non-null classes (default: < 5 classes).
# - accuracy: pooled micro accuracy (correct / total over all such cells).
# - precision / recall / f1: mean of sklearn macro P/R/F1 per (dataset, column),
#   skipping columns where ground truth has only one class among imputed rows.

TABLE_METHODS = [
    "hyperimpute",
    "missforest",
    "mcpfn",
    "tabimpute_round2_t2",
    "tabimpute_round2_t2_adapter",
    "mode",
]

method_stats: defaultdict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "correct": 0,
        "total": 0,
        "macro_precisions": [],
        "macro_recalls": [],
        "macro_f1s": [],
    }
)

for dataset in datasets:
    df_missing = pd.read_pickle(f"{base_path}/{dataset}/dataframe_missing.pkl")
    df = pd.read_pickle(f"{base_path}/{dataset}/dataframe.pkl")
    mask = pd.isnull(df_missing).values

    for method in methods:
        imputed_path = f"{base_path}/{dataset}/dataframe_imputed_{method}.pkl"
        if not os.path.exists(imputed_path):
            continue
        df_imputed = pd.read_pickle(imputed_path)

        for col_idx in range(df.shape[1]):
            if not _is_categorical_column(df, col_idx):
                continue
            if _n_classes_ground_truth(df, col_idx) >= MAX_CLASSES_EXCLUSIVE:
                continue
            col_mask = mask[:, col_idx]
            if not np.any(col_mask):
                continue
            y_true_col, y_pred_col = _masked_column_as_object(
                df, df_imputed, col_idx, col_mask
            )
            s_true = pd.Series(y_true_col)
            s_pred = pd.Series(y_pred_col)
            valid = ~(pd.isna(s_true) | pd.isna(s_pred))
            if not valid.any():
                continue
            yt = s_true[valid].to_numpy()
            yp = s_pred[valid].to_numpy()

            stats_m = method_stats[method]
            stats_m["correct"] += int(np.sum(yt == yp))
            stats_m["total"] += int(len(yt))

            if len(np.unique(yt)) >= 2:
                stats_m["macro_precisions"].append(
                    precision_score(yt, yp, average="macro", zero_division=0)
                )
                stats_m["macro_recalls"].append(
                    recall_score(yt, yp, average="macro", zero_division=0)
                )
                stats_m["macro_f1s"].append(
                    f1_score(yt, yp, average="macro", zero_division=0)
                )

summary_rows: list[dict[str, Any]] = []
for method in methods:
    if method not in method_stats:
        continue
    st = method_stats[method]
    n_tot = st["total"]
    acc = float(st["correct"] / n_tot) if n_tot else float("nan")
    mp, mr, mf = st["macro_precisions"], st["macro_recalls"], st["macro_f1s"]
    summary_rows.append(
        {
            "method": method,
            "f1": float(np.mean(mf)) if mf else float("nan"),
            "accuracy": acc,
            "recall": float(np.mean(mr)) if mr else float("nan"),
            "precision": float(np.mean(mp)) if mp else float("nan"),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_table = summary_df[summary_df["method"].isin(TABLE_METHODS)].copy()
summary_table = summary_table.sort_values("f1", ascending=False, na_position="last")

display_cols = ["method", "f1", "accuracy", "recall", "precision"]
print(
    f"\nOverall categorical imputation "
    f"(imputed cells only; columns with < {MAX_CLASSES_EXCLUSIVE} classes in ground truth), "
    "sorted by F1:"
)
print(
    summary_table[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

latex_df = summary_table[display_cols].set_index("method")
latex_table = latex_df.to_latex(
    na_rep="--",
    float_format="{:.4f}".format,
    escape=False,
)
output_file = "../figures/categorical_imputation_metrics.tex"
with open(output_file, "w") as f:
    f.write(latex_table)

print(f"\nLaTeX table written to {output_file}")