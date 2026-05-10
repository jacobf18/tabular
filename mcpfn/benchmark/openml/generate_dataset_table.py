#!/usr/bin/env python3
"""
Generate a LaTeX table of all datasets in the openml directory
"""

import os

def get_dataset_names():
    """Get all dataset names from the openml directory"""
    base_path = "datasets/openml"
    datasets = []
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            datasets.append(item)
    
    return sorted(datasets)

def create_latex_table(datasets, cols=3):
    """Create a LaTeX table with the dataset names"""
    
    # Calculate number of rows needed
    rows = (len(datasets) + cols - 1) // cols
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Datasets used in the benchmark evaluation}\n"
    latex += "\\label{tab:datasets}\n"
    latex += "\\begin{tabular}{" + "l" * cols + "}\n"
    latex += "\\toprule\n"
    
    # Create header row
    header = " & ".join([f"Dataset {i+1}" for i in range(cols)])
    latex += header + " \\\\\n"
    latex += "\\midrule\n"
    
    # Fill the table
    for row in range(rows):
        row_data = []
        for col in range(cols):
            idx = row * cols + col
            if idx < len(datasets):
                # Escape underscores and other special LaTeX characters
                dataset_name = datasets[idx].replace("_", "\\_")
                row_data.append(dataset_name)
            else:
                row_data.append("")
        
        latex += " & ".join(row_data) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def main():
    datasets = get_dataset_names()
    
    print(f"Found {len(datasets)} datasets:")
    print("\n".join(f"{i+1:2d}. {dataset}" for i, dataset in enumerate(datasets)))
    
    print("\n" + "="*60)
    print("LATEX TABLE (1 column):")
    print("="*60)
    latex_table = create_latex_table(datasets, cols=1)
    print(latex_table)

if __name__ == "__main__":
    main()
