import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mar_onesided_config = {
    "num_layers_upper": 2,
    "hidden_lower": 2,
    "hidden_upper": 5,
    "activation": "relu",
    "N": 10,
    "T": 8,
    "row_neighbor_upper": 3,
    "col_neighbor_upper": 3,
    "seed": 42,
    "neighbor_type": "random",
    "extreme_neighbor_mode": "zero",
    "extreme_axis": "row",
}


class MAR_onesided_missingness(nn.Module):
    """
    Neural Net approach for constructing MAR propensities with extreme neighbor modes
    Args:
        X: (n_rows, n_cols) numeric matrix; fully observed
        config: dict; configuration for the neural net
    Returns:
        prop: (n_rows, n_cols) numeric matrix; missingness propensities
    """

    def _layer_activation_mixer(self, i, t):
        """Helper function to interleave layers and activations"""
        mixed = []
        # Add input layer first
        mixed.append(self.input_layers[i][t])
        mixed.append(self.activation_fn)

        # Add hidden layers with activations
        for j, layer in enumerate(self.layers[i][t]):
            mixed.append(layer)
            if j < len(self.layers[i][t]) - 1:
                mixed.append(self.activation_fn)

        # Add output layer
        mixed.append(self.output_layers[i][t])

        return nn.Sequential(*mixed)

    def __init__(self, config):
        super().__init__()
        self.N = config["N"]
        self.T = config["T"]
        self.hid_low = config["hidden_lower"]
        self.hid_up = config["hidden_upper"]
        self.activation = config["activation"]
        self.neighbor_type = config["neighbor_type"]

        # New parameters for extreme neighbor modes
        self.extreme_neighbor_mode = config.get(
            "extreme_neighbor_mode", "zero"
        )  # 'zero' or 'all'
        self.extreme_axis = config.get("extreme_axis", "row")  # 'row' or 'column'

        torch.manual_seed(config["seed"])

        self.num_layers = torch.stack(
            [
                torch.stack(
                    [
                        torch.randint(1, config["num_layers_upper"] + 1, size=(1,))
                        for t in range(self.T)
                    ]
                )
                for i in range(self.N)
            ]
        )

        # For the non-extreme axis, use normal random neighbor selection
        if self.extreme_axis == "row":
            self.col_neighbor = torch.stack(
                [
                    torch.stack(
                        [
                            torch.randint(
                                1, config["col_neighbor_upper"] + 1, size=(1,)
                            )
                            for t in range(self.T)
                        ]
                    )
                    for i in range(self.N)
                ]
            )
            # Row neighbors will be determined by extreme mode
            self.row_neighbor = None
        else:  # extreme_axis == 'column'
            self.row_neighbor = torch.stack(
                [
                    torch.stack(
                        [
                            torch.randint(
                                1, config["row_neighbor_upper"] + 1, size=(1,)
                            )
                            for t in range(self.T)
                        ]
                    )
                    for i in range(self.N)
                ]
            )
            # Column neighbors will be determined by extreme mode
            self.col_neighbor = None

        # Calculate input size based on extreme modes
        if self.extreme_axis == "row":
            if self.extreme_neighbor_mode == "zero":
                # Only self row, normal column neighbors
                self.input_size = torch.stack(
                    [
                        torch.stack(
                            [1 * (self.col_neighbor[i][t] + 1) for t in range(self.T)]
                        )
                        for i in range(self.N)
                    ]
                )
            else:  # 'all'
                # All rows, normal column neighbors
                self.input_size = torch.stack(
                    [
                        torch.stack(
                            [
                                self.N * (self.col_neighbor[i][t] + 1)
                                for t in range(self.T)
                            ]
                        )
                        for i in range(self.N)
                    ]
                )
        else:  # extreme_axis == 'column'
            if self.extreme_neighbor_mode == "zero":
                # Normal row neighbors, only self column
                self.input_size = torch.stack(
                    [
                        torch.stack(
                            [(self.row_neighbor[i][t] + 1) * 1 for t in range(self.T)]
                        )
                        for i in range(self.N)
                    ]
                )
            else:  # 'all'
                # Normal row neighbors, all columns
                self.input_size = torch.stack(
                    [
                        torch.stack(
                            [
                                (self.row_neighbor[i][t] + 1) * self.T
                                for t in range(self.T)
                            ]
                        )
                        for i in range(self.N)
                    ]
                )

        self.hid_seq = [
            [
                torch.randint(
                    self.hid_low, self.hid_up + 1, size=(self.num_layers[i][t],)
                )
                for t in range(self.T)
            ]
            for i in range(self.N)
        ]

        # Create input layers (input_size -> first_hidden)
        self.input_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.input_size[i][t], self.hid_seq[i][t][0])
                        for t in range(self.T)
                    ]
                )
                for i in range(self.N)
            ]
        )

        # Create hidden layers (hidden -> hidden)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                nn.Linear(
                                    self.hid_seq[i][t][k], self.hid_seq[i][t][k + 1]
                                )
                                for k in range(self.num_layers[i][t] - 1)
                            ]
                        )
                        for t in range(self.T)
                    ]
                )
                for i in range(self.N)
            ]
        )

        # Create output layers (last_hidden -> 1)
        self.output_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.hid_seq[i][t][-1], 1) for t in range(self.T)]
                )
                for i in range(self.N)
            ]
        )

        # Initialize all layers
        for i in range(self.N):
            for t in range(self.T):
                # Initialize input layer
                nn.init.normal_(self.input_layers[i][t].weight, mean=0.0, std=1)
                nn.init.normal_(self.input_layers[i][t].bias, mean=0.0, std=1)
                self.input_layers[i][t].weight.requires_grad = False
                self.input_layers[i][t].bias.requires_grad = False

                # Initialize hidden layers
                for layer in self.layers[i][t]:
                    nn.init.normal_(layer.weight, mean=0.0, std=1)
                    nn.init.normal_(layer.bias, mean=0.0, std=1)
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False

                # Initialize output layer
                nn.init.normal_(self.output_layers[i][t].weight, mean=0.0, std=1)
                nn.init.normal_(self.output_layers[i][t].bias, mean=0.0, std=1)
                self.output_layers[i][t].weight.requires_grad = False
                self.output_layers[i][t].bias.requires_grad = False

        if self.activation == "relu":
            self.activation_fn = nn.ReLU()
        elif self.activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif self.activation == "tanh":
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

        self.final_net = nn.ModuleList(
            [
                nn.ModuleList(
                    [self._layer_activation_mixer(i, t) for t in range(self.T)]
                )
                for i in range(self.N)
            ]
        )

    def get_final_net(self, i, t):
        return self.final_net[i][t]

    def forward(self, X):
        n_rows, n_cols = X.shape
        assert n_rows == self.N and n_cols == self.T
        propensity_matrix = torch.zeros(n_rows, n_cols)

        for i in range(self.N):
            for t in range(self.T):
                input_size = self.input_size[i][t].item()

                # Determine neighbor indices based on extreme modes
                if self.extreme_axis == "row":
                    # Row neighbors: extreme mode
                    if self.extreme_neighbor_mode == "zero":
                        row_neighbor_idx = torch.tensor([i])  # Only self
                    else:  # 'all'
                        row_neighbor_idx = torch.arange(n_rows)  # All rows

                    # Column neighbors: normal random selection
                    col_neighbor = self.col_neighbor[i][t].item()
                    col_neighbor = min(col_neighbor, n_cols - 1)
                    col_neighbor = max(col_neighbor, 1)

                    col_neighbor_idx = torch.randint(0, n_cols, size=(col_neighbor,))

                    # Remove current position if it exists and replace with a different neighbor
                    if t in col_neighbor_idx:
                        available_cols = torch.tensor(
                            [
                                c
                                for c in range(n_cols)
                                if c != t and c not in col_neighbor_idx
                            ]
                        )
                        if len(available_cols) > 0:
                            replacement = available_cols[
                                torch.randint(0, len(available_cols), (1,))
                            ]
                            col_neighbor_idx[col_neighbor_idx == t] = replacement

                    # Add self neighbor
                    col_neighbor_idx = torch.cat([col_neighbor_idx, torch.tensor([t])])

                else:  # extreme_axis == 'column'
                    # Row neighbors: normal random selection
                    row_neighbor = self.row_neighbor[i][t].item()
                    row_neighbor = min(row_neighbor, n_rows - 1)
                    row_neighbor = max(row_neighbor, 1)

                    row_neighbor_idx = torch.randint(0, n_rows, size=(row_neighbor,))

                    # Remove current position if it exists and replace with a different neighbor
                    if i in row_neighbor_idx:
                        available_rows = torch.tensor(
                            [
                                r
                                for r in range(n_rows)
                                if r != i and r not in row_neighbor_idx
                            ]
                        )
                        if len(available_rows) > 0:
                            replacement = available_rows[
                                torch.randint(0, len(available_rows), (1,))
                            ]
                            row_neighbor_idx[row_neighbor_idx == i] = replacement

                    # Add self neighbor
                    row_neighbor_idx = torch.cat([row_neighbor_idx, torch.tensor([i])])

                    # Column neighbors: extreme mode
                    if self.extreme_neighbor_mode == "zero":
                        col_neighbor_idx = torch.tensor([t])  # Only self
                    else:  # 'all'
                        col_neighbor_idx = torch.arange(n_cols)  # All columns

                # Use proper PyTorch advanced indexing
                X_neighbor = X[
                    row_neighbor_idx.unsqueeze(1), col_neighbor_idx.unsqueeze(0)
                ]

                # Flatten and ensure correct input size
                X_neighbor = X_neighbor.flatten()

                # Add batch dimension for neural network
                X_neighbor = X_neighbor.unsqueeze(0)  # Shape: [1, input_size]

                # Process through network
                final_net = self.get_final_net(i, t)
                propensity = torch.sigmoid(final_net(X_neighbor))

                # Propensity matrix
                propensity_matrix[i, t] = propensity.squeeze()

        return propensity_matrix
