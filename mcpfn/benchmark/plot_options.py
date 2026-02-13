"""
Centralized plotting configuration with LaTeX font settings.

This module contains all common plot options used across plotting scripts.
Import this module to use consistent plotting styles with LaTeX fonts.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

# Configure matplotlib to use LaTeX for text rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16

# Set seaborn style
sns.set(style="whitegrid")

# Common figure sizes
FIGURE_SIZES = {
    'standard': (6.5, 5.5),
    'wide': (10, 6),
    'tall': (6.5, 4.5),
    'square': (6.5, 4),
    'large': (12, 6),
}

# Method name mappings (consolidated from all plot files)
METHOD_NAMES = {
    "tabimpute_large_mcar": "TabImpute (New Model)",
    "tabimpute_large_mcar_mnar": "TabImpute (MNAR)",
    "tabimpute_large_mcar_rank_1_11": "TabImpute (New)",
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
    "cacti": "CACTI",
    "mode": "Mode",
    "mixed_adaptive": "MCPFN",
    "mixed_perm_both_row_col": "MCPFN (Linear Permuted)",
    "tabpfn_impute": "Col-TabPFN",
    "tabpfn": "EWF-TabPFN",
}

# Neutral and highlight colors
NEUTRAL_COLOR = "#B8B8B8"
HIGHLIGHT_COLOR = "#2A6FBB"

# Method color mappings (consolidated from all plot files)
METHOD_COLORS = {
    "TabImpute+": HIGHLIGHT_COLOR,
    "TabImpute": HIGHLIGHT_COLOR,
    "TabImpute Ensemble": HIGHLIGHT_COLOR,
    "TabImpute (Self-Masking-MNAR)": HIGHLIGHT_COLOR,
    "TabImpute (Fixed)": HIGHLIGHT_COLOR,
    "TabImpute (MCAR)": HIGHLIGHT_COLOR,
    "TabImpute (MAR)": HIGHLIGHT_COLOR,
    "TabImpute (MNAR)": HIGHLIGHT_COLOR,
    "TabImpute (Nonlinear)": HIGHLIGHT_COLOR,
    "TabImpute Router": HIGHLIGHT_COLOR,
    "TabImpute (New Model)": HIGHLIGHT_COLOR,
    "TabImpute (MNAR)": HIGHLIGHT_COLOR,
    "TabImpute (New)": HIGHLIGHT_COLOR,
    "EWF-TabPFN": NEUTRAL_COLOR,
    "HyperImpute": NEUTRAL_COLOR,
    "MissForest": NEUTRAL_COLOR,
    "OT": NEUTRAL_COLOR,
    "Col Mean": NEUTRAL_COLOR,
    "SoftImpute": NEUTRAL_COLOR,
    "ICE": NEUTRAL_COLOR,
    "MICE": NEUTRAL_COLOR,
    "GAIN": NEUTRAL_COLOR,
    "MIWAE": NEUTRAL_COLOR,
    "TabPFN": NEUTRAL_COLOR,
    "Col-TabPFN": NEUTRAL_COLOR,
    "K-Nearest Neighbors": NEUTRAL_COLOR,
    "ForestDiffusion": NEUTRAL_COLOR,
    "MCPFN": NEUTRAL_COLOR,
    "MCPFN (Linear Permuted)": NEUTRAL_COLOR,
    "MCPFN (Nonlinear Permuted)": NEUTRAL_COLOR,
    "DiffPuter": NEUTRAL_COLOR,
    "ReMasker": NEUTRAL_COLOR,
    "CACTI": NEUTRAL_COLOR,
    "Mode": NEUTRAL_COLOR,
}

# Neutral colors for different methods
NEUTRAL_COLORS = {
    "HyperImpute": "#A8A8A8",
    "Col Mean": "#B3B3B3",
    "MissForest": "#B0B0B0",
}

# Missingness patterns
PATTERNS = {
    "MCAR",
    "MAR",
    "MNAR",
    "MAR_Neural",
    "MAR_BlockNeural",
    "MAR_Sequential",
    "MNARPanelPattern",
    "MNARPolarizationPattern",
    "MNARSoftPolarizationPattern",
    "MNARLatentFactorPattern",
    "MNARClusterLevelPattern",
    "MNARTwoPhaseSubsetPattern",
    "MNARCensoringPattern",
}

# LaTeX pattern names
PATTERN_LATEX_NAMES = {
    "MCAR": "\\mcar",
    "MAR_Neural": "\\nnmar",
    "MNAR": "\\mnarself",
    "MAR": "\\colmar",
    "MAR_BlockNeural": "\\marblockneural",
    "MAR_Sequential": "\\seqmar",
    "MNARPanelPattern": "\\panelmnar",
    "MNARPolarizationPattern": "\\polarmnar",
    "MNARSoftPolarizationPattern": "\\softpolarmnar",
    "MNARLatentFactorPattern": "\\latentmnar",
    "MNARClusterLevelPattern": "\\clustermnar",
    "MNARTwoPhaseSubsetPattern": "\\twophasemnar",
    "MNARCensoringPattern": "\\censormnar",
    "Overall": "Overall"
}

# Common plotting parameters
PLOT_PARAMS = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'edgecolor': 'none',
}

# Barplot styling
BARPLOT_STYLE = {
    'capsize': 0.2,
    'err_kws': {"color": "#999999"},
    'edgecolor': "#6E6E6E",
}

# Line plot styling
LINEPLOT_STYLE = {
    'linewidth': 2.0,
    'markersize': 6,
    'markeredgewidth': 1.5,
}

def setup_latex_fonts():
    """
    Configure matplotlib to use LaTeX fonts.
    Call this function at the beginning of plotting scripts.
    """
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
    sns.set(style="whitegrid")

def get_method_color(method_name, default=None):
    """
    Get color for a method name.
    
    Args:
        method_name: Display name of the method
        default: Default color if method not found
        
    Returns:
        Color hex code
    """
    return METHOD_COLORS.get(method_name, default or NEUTRAL_COLOR)

def get_method_name(method_key):
    """
    Get display name for a method key.
    
    Args:
        method_key: Internal method identifier
        
    Returns:
        Display name for the method
    """
    return METHOD_NAMES.get(method_key, method_key)

def apply_latex_labels(ax, xlabel=None, ylabel=None, title=None):
    """
    Apply LaTeX-formatted labels to axes.
    
    Args:
        ax: Matplotlib axes object
        xlabel: X-axis label (will be wrapped in LaTeX)
        ylabel: Y-axis label (will be wrapped in LaTeX)
        title: Plot title (will be wrapped in LaTeX)
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
