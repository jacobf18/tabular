import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = [
    # "softimpute", 
    "column_mean", 
    "hyperimpute",
    # # "ot_sinkhorn",
    "missforest",
    # "ice",
    # "mice",
    # "gain",
    # "miwae",
    # "mcpfn_ensemble",
    "masters_mcar",
    # # "tabpfn_impute",
]

method_names = {
    "masters_mcar": "TabImpute",
    "tabpfn_impute": "TabPFN",
    "column_mean": "Col Mean",
    "hyperimpute": "HyperImpute",
    "ot_sinkhorn": "OT",
    "missforest": "MissForest",
    "softimpute": "SoftImpute",
    "ice": "ICE",
    "mice": "MICE",
    "gain": "GAIN",
    "miwae": "MIWAE",
}

highlight_color = "#2A6FBB"
neutral_color = "#B8B8B8"

# Create slightly different neutral colors for each method
neutral_colors = {
    "HyperImpute": "#A8A8A8",  # Slightly darker gray
    "Col Mean": "#B3B3B3",      # Medium gray
    "MissForest": "#B0B0B0",   # Medium-dark gray
}

# Only focus on MCAR pattern
pattern_name = "MCAR"

# p values from 0.1 to 0.9
p_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

negative_rmse = {}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

def compute_percent_correct(X_true, X_imputed, mask):
    return np.mean(np.isclose(X_true[mask], X_imputed[mask], atol=0.01)) * 100

# Collect data for all p values and MCAR pattern
for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        config_split = config.split("_")
        p = config_split[-1]
        
        # Check if this is one of our target p values
        if float(p) not in p_values:
            continue
        
        # remove p from config_split
        config_split = config_split[:-1]
        current_pattern = "_".join(config_split)
        
        # Only process MCAR pattern
        if current_pattern != pattern_name:
            continue
        
        X_missing = np.load(f"{base_path}/{dataset}/{current_pattern}_{p}/missing.npy")
        X_true = np.load(f"{base_path}/{dataset}/{current_pattern}_{p}/true.npy")
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            try:
                X_imputed = np.load(f"{base_path}/{dataset}/{current_pattern}_{p}/{method}.npy")
                name = method_names[method]
                negative_rmse[(dataset, float(p), name)] = compute_negative_rmse(X_true, X_imputed, mask)
                # percent_correct[(dataset, float(p), name)] = compute_percent_correct(X_true, X_imputed, mask)
            except FileNotFoundError:
                print(f"Warning: File not found for {dataset}/{current_pattern}_{p}/{method}.npy")
                continue

data = []
for (dataset, p, method), value in negative_rmse.items():
    data.append({
        'dataset': dataset,
        'p': p,
        'method': method,
        'value': value
    })
df = pd.DataFrame(data)
df.sort_values(by="p", inplace=True)

# # Remove outliers from the value column
# df["value"] = df.groupby(["dataset", "p"])["value"].transform(
#     lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95))
# )

# Normalize the value column within each dataset and p value, but leave the method column
df["value_norm"] = df.groupby(["dataset"])["value"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
print(df)
include_methods = ["TabImpute", "HyperImpute", "Col Mean", "MissForest"]

# Create color palette: TabImpute gets highlight color, others get slightly different neutral colors
palette = {method: highlight_color if method == "TabImpute" else neutral_colors.get(method, neutral_color) for method in include_methods}

# Plot the values with a separate line for each method
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="p", y="value_norm", hue="method", data=df[df["method"].isin(include_methods)], hue_order=include_methods, palette=palette)
plt.ylabel("1 - Normalized RMSE (0–1)")
plt.xlabel("Fraction of Missing Values")

# Remove legend and add labels at the end of each line
ax.legend_.remove()

# Make TabImpute's line bold
lines = ax.get_lines()
for i, method in enumerate(include_methods):
    if i < len(lines) and method == "TabImpute":
        lines[i].set_linewidth(2.5)  # Make TabImpute line bold

# Get the maximum x value to place labels at the end
max_p = df["p"].max()

# Get lines from the plot and add labels at the end
lines = ax.get_lines()
label_positions = []  # Store (method, x_end, y_end) for overlap detection

# First pass: collect all label positions
for i, method in enumerate(include_methods):
    if i < len(lines):
        line = lines[i]
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if len(x_data) > 0 and len(y_data) > 0:
            x_end = x_data[-1]
            y_end = y_data[-1]
            label_positions.append((method, x_end, y_end))

# Adjust positions to prevent overlap (minimum spacing)
min_spacing = 0.03  # Minimum vertical spacing between labels (in data coordinates)
label_positions.sort(key=lambda x: x[2])  # Sort by y position

# Adjust overlapping labels
adjusted_positions = []
for i, (method, x_end, y_end) in enumerate(label_positions):
    adjusted_y = y_end
    # Check if too close to previous label
    if i > 0:
        prev_y = adjusted_positions[-1][2]
        if abs(adjusted_y - prev_y) < min_spacing:
            # Move this label up or down to create space
            if adjusted_y > prev_y:
                adjusted_y = prev_y + min_spacing
            else:
                adjusted_y = prev_y - min_spacing
    adjusted_positions.append((method, x_end, adjusted_y))

# Second pass: add labels with adjusted positions
for method, x_end, y_end in adjusted_positions:
    line_idx = include_methods.index(method)
    if line_idx < len(lines):
        line = lines[line_idx]
        line_color = line.get_color()
        ax.text(x_end, y_end, f"  {method}", 
               color=line_color, va='center', fontsize=10, fontweight='bold' if method == "TabImpute" else 'normal')

plt.savefig('figures/mcar_normalized_performance_vs_p.pdf', dpi=300)
plt.close()

print(f"Saved figure to figures/mcar_normalized_performance_vs_p.pdf")