# --- Plotting ---
# sns.set(style="whitegrid")

# for (dataset_name, pattern_name), imputer_errors in results.items():
#     imputer_names = list(imputer_errors.keys())
#     error_data = [imputer_errors[imp] for imp in imputer_names]

#     plt.figure(figsize=(8, 5))
#     ax = sns.boxplot(data=error_data)
#     ax.set_xticks(range(len(imputer_names)))
#     ax.set_xticklabels(imputer_names)
#     ax.set_ylabel("Absolute Error")
#     ax.set_title(f"Dataset: {dataset_name} | Pattern: {pattern_name}")
#     ax.set_ylim(bottom=0.0)
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.savefig(f"boxplot_{dataset_name}_{pattern_name}.png")
#     plt.close()
    
#     plt.figure(figsize=(8, 5))
#     ax = sns.violinplot(data=error_data, cut=0)
#     ax.set_xticks(range(len(imputer_names)))
#     ax.set_xticklabels(imputer_names)
#     ax.set_ylabel("Absolute Error")
#     ax.set_title(f"Dataset: {dataset_name} | Pattern: {pattern_name}")
#     ax.set_ylim(bottom=0.0)
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.savefig(f"violinplot_{dataset_name}_{pattern_name}.png")
#     plt.close()