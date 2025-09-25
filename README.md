# TabImpute

This is the official repository of TabImpute, a transformer-based architecture for missing data imputation on tabular data.

## Installation

```bash
pip install -e .
```

## Usage

```python
from tabimpute.interface import ImputePFN
import numpy as np

imputer = ImputePFN(device='cpu', # 'cuda' if you have a GPU
                    encoder_path='./src/mcpfn/model/encoder.pth', # Path to the encoder model
                    borders_path='./borders.pt', # Path to the borders tensor
                    checkpoint_path='./stage1/checkpoint/test.ckpt') # Path to the checkpoint file

X = np.random.rand(5, 5)
print(X)
X[np.random.rand(*X.shape) < 0.1] = np.nan
print(X)

out = imputer.impute(X)
print(out)
```

## Generate data

Run the `generate_data.sh` script to generate the data.

## Train

Run the `train.sh` script to train the model.