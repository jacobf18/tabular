import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mar_missing import MAR_missingness

mar_block_config = {
    "N": 100,
    "T": 50,
    "row_blocks": 10,
    "col_blocks": 10,
    "convolution_type": "mean"
}

class MAR_block_missingness(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.N = config['N']
        self.T = config['T']
        self.row_blocks = config['row_blocks']
        self.col_blocks = config['col_blocks']
        self.convolution_type = config['convolution_type']
        mar_small_config = {
            "num_layers_upper": 3,
            "hidden_lower": 1,
            "hidden_upper": 100,
            "activation": "relu",
            "N": self.row_blocks,
            "T": self.col_blocks,
            "row_neighbor_upper": self.row_blocks//2,
            "col_neighbor_upper": self.col_blocks//2,
            "seed": 42,
            "neighbor_type": "random"
        }
        self.mar_generator = MAR_missingness(mar_small_config)
    def forward(self, X):
        assert X.shape[0] == self.N and X.shape[1] == self.T
        row_partition = np.random.multinomial(self.N, [1/self.row_blocks]*self.row_blocks)
        col_partition = np.random.multinomial(self.T, [1/self.col_blocks]*self.col_blocks)
        row_cumsum = np.concatenate([[0], np.cumsum(row_partition)])
        col_cumsum = np.concatenate([[0], np.cumsum(col_partition)])
        X_small = np.zeros((self.row_blocks, self.col_blocks))
        for i in range(self.row_blocks):
            for t in range(self.col_blocks):
                block = X[row_cumsum[i]:row_cumsum[i+1], col_cumsum[t]:col_cumsum[t+1]]
                if self.convolution_type == "mean":
                    X_small[i, t] = block.mean().item()
                if self.convolution_type == "max":
                    X_small[i, t] = block.max().item()
        X_small = torch.from_numpy(X_small).to(dtype=torch.float32)
        propensity_small = self.mar_generator(X_small)

        return propensity_small, row_cumsum, col_cumsum
