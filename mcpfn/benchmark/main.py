import torch
import numpy as np
from download_openml import fetch_clean_openml_datasets
from mcpfn.prior.training_set_generation import MCARPattern
from mcpfn.interface import ImputePFN  # your imputer


def compute_mad(X_true: torch.Tensor, X_missing: torch.Tensor, X_imputed: np.ndarray) -> float:
    """
    Compute Mean Absolute Deviation (MAD) between imputed and ground truth,
    restricted to only the entries that were originally missing.

    Args:
        X_true: full torch tensor with no missing values
        X_missing: tensor with NaNs in some entries
        X_imputed: full imputed matrix (numpy array of same shape)

    Returns:
        Mean Absolute Deviation (float)
    """
    mask = torch.isnan(X_missing)
    true_vals = X_true[mask]
    imputed_vals = torch.tensor(X_imputed)[mask]

    mad = torch.mean(torch.abs(true_vals - imputed_vals)).item()
    return mad


# Step 1: Download datasets
datasets = fetch_clean_openml_datasets(num_datasets=3)

# Step 2: Define missingness pattern
missingness = MCARPattern(config={"p_missing": 0.5})

print(type(datasets[0][0]), missingness)

# Step 3: Load imputer
imputer = ImputePFN(
    device='cpu',
    encoder_path='/Users/dwaipayansaha/researchProjects/tabular/mcpfn/src/mcpfn/model/encoder.pth',
    borders_path='/Users/dwaipayansaha/researchProjects/tabular/mcpfn/borders.pt',
    checkpoint_path='./test.ckpt'
)

for X, name, did in datasets:
    X_missing = missingness._induce_missingness(X.clone())
    # print(X_missing.shape, type(X_missing))
    X_filled = imputer.impute(X_missing.numpy())  # assuming numpy input
    print(f"✔️ Completed imputation for {name} (ID: {did})")

    mad = compute_mad(X, X_missing, X_filled)  # compute MAD for each dataset
    print(f"MAD for {name} (ID: {did}): {mad}")
