from mar_missingness import MAR_missingness
import matplotlib.pyplot as plt
import numpy as np
import torch

# Create MAR simulation configuration
mar_config = {
    "num_layers_upper": 3,
    "hidden_lower": 1,
    "hidden_upper": 10,
    "activation": "relu",
    "N": 20, # Row size of X (reduced for testing)
    "T": 20, # Column size of X (reduced for testing)
    "row_neighbor_upper": 5, # Upper bound of row neighbor (reduced for testing)
    "col_neighbor_upper": 5, # Upper bound of column neighbor (reduced for testing)
    "seed": 42,
    "neighbor_type": "random"
}

N = mar_config['N']
T = mar_config['T']
X = torch.randn(N, T)

model = MAR_missingness(mar_config)
propensities = model(X)

masks = []
num_examples = 4
for i in range(num_examples):
    mask = torch.bernoulli(propensities)
    masks.append(mask)

titles = ['Mask 1', 'Mask 2', 'Mask 3', 'Mask 4']

# Create subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

# Plot each mask
for i, (mask, title) in enumerate(zip(masks, titles)):
    axes[i].imshow(mask, cmap='binary', aspect='auto')
    axes[i].set_title(title)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Samples')

plt.tight_layout()
plt.savefig('outputs/data_check/multiple_masks.png', dpi=300, bbox_inches='tight')
plt.close()