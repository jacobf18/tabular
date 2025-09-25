#!/usr/bin/env python3
"""
Training Data Generator using MCPFN MissingnessPrior

This script generates synthetic tabular datasets with controlled missingness patterns
for training diffusion models on tabular data imputation tasks.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# example of a config
mar_config = {
    "num_layers_upper": 3,
    "hidden_lower": 1,
    "hidden_upper": 100,
    "activation": "relu",
    "N": 100, # Row size of X (reduced for testing)
    "T": 50, # Column size of X (reduced for testing)
    "row_neighbor_upper": 5, # Upper bound of row neighbor (reduced for testing)
    "col_neighbor_upper": 5, # Upper bound of column neighbor (reduced for testing)
    "seed": 42,
    "neighbor_type": "random"
}

class MAR_missingness(nn.Module):
    """
    Neural Net approach for constructing MAR propensities
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
        self.N = config['N']
        self.T = config['T']
        self.hid_low = config['hidden_lower']
        self.hid_up = config['hidden_upper']
        self.activation = config['activation']
        self.neighbor_type = config['neighbor_type']
        torch.manual_seed(config['seed'])
        
        self.num_layers = torch.stack([
            torch.stack([torch.randint(1, config['num_layers_upper']+ 1, size = (1, )) for t in range(self.T)])
            for i in range(self.N)
        ])
        self.row_neighbor = torch.stack([
            torch.stack([torch.randint(1, config['row_neighbor_upper']+1, size = (1, )) for t in range(self.T)])
            for i in range(self.N)
        ])
        self.col_neighbor = torch.stack([
            torch.stack([torch.randint(1, config['col_neighbor_upper']+1, size = (1, )) for t in range(self.T)])
            for i in range(self.N)
        ])
        self.input_size = torch.stack([
            torch.stack([(self.row_neighbor[i][t]+1)*(self.col_neighbor[i][t]+1) for t in range(self.T)])
            for i in range(self.N)
        ])
        self.hid_seq = [
            [torch.randint(self.hid_low, self.hid_up+1, size = (self.num_layers[i][t], )) for t in range(self.T)]
            for i in range(self.N)
        ]

        # Create input layers (input_size -> first_hidden)
        self.input_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.input_size[i][t], self.hid_seq[i][t][0])
                for t in range(self.T)
            ])
            for i in range(self.N)
        ])

        # Create hidden layers (hidden -> hidden)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(self.hid_seq[i][t][k], self.hid_seq[i][t][k+1]) for k in range(self.num_layers[i][t]-1)
                ])
                for t in range(self.T)
            ])
            for i in range(self.N)
        ])

        # Create output layers (last_hidden -> 1)
        self.output_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.hid_seq[i][t][-1], 1)
                for t in range(self.T)
            ])
            for i in range(self.N)
        ])
        
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

        self.final_net = nn.ModuleList([
            nn.ModuleList([
                self._layer_activation_mixer(i, t)
                for t in range(self.T)
            ])
            for i in range(self.N)
        ])
    
    def get_final_net(self, i, t):
        return self.final_net[i][t]
    
    # def get_output_layers(self, i, t):
        # return self.output_layers[i][t]
    
    def forward(self, X):
        n_rows, n_cols = X.shape
        assert n_rows == self.N and n_cols == self.T
        propensity_matrix = torch.zeros(n_rows, n_cols)

        for i in range(self.N):
            for t in range(self.T):
                input_size = self.input_size[i][t].item()
                row_neighbor = self.row_neighbor[i][t].item()
                col_neighbor = self.col_neighbor[i][t].item()
                
                # Add bounds checking for neighbor sizes
                row_neighbor = min(row_neighbor, n_rows - 1)
                col_neighbor = min(col_neighbor, n_cols - 1)
                
                # Ensure minimum neighbor sizes
                row_neighbor = max(row_neighbor, 1)
                col_neighbor = max(col_neighbor, 1)
                
                final_net = self.get_final_net(i, t)
                row_neighbor_idx = torch.randint(0, n_rows, size=(row_neighbor,))
                col_neighbor_idx = torch.randint(0, n_cols, size=(col_neighbor,))
                
                # Remove current position if it exists and replace with a different neighbor to maintain count
                if i in row_neighbor_idx:
                    # Find a replacement that's not i and not already in the list
                    available_rows = torch.tensor([r for r in range(n_rows) if r != i and r not in row_neighbor_idx])
                    if len(available_rows) > 0:
                        replacement = available_rows[torch.randint(0, len(available_rows), (1,))]
                        row_neighbor_idx[row_neighbor_idx == i] = replacement
                
                if t in col_neighbor_idx:
                    # Find a replacement that's not t and not already in the list
                    available_cols = torch.tensor([c for c in range(n_cols) if c != t and c not in col_neighbor_idx])
                    if len(available_cols) > 0:
                        replacement = available_cols[torch.randint(0, len(available_cols), (1,))]
                        col_neighbor_idx[col_neighbor_idx == t] = replacement
                    
                # Add self neighbor
                row_neighbor_idx = torch.cat([row_neighbor_idx, torch.tensor([i])]) # Self neighbor
                col_neighbor_idx = torch.cat([col_neighbor_idx, torch.tensor([t])]) # Self neighbor
                
                # Use proper PyTorch advanced indexing
                X_neighbor = X[row_neighbor_idx.unsqueeze(1), col_neighbor_idx.unsqueeze(0)]
                
                # Flatten and ensure correct input size
                X_neighbor = X_neighbor.flatten()
                
                # Add batch dimension for neural network
                X_neighbor = X_neighbor.unsqueeze(0)  # Shape: [1, input_size]
                
                # Process through network (final_net already includes output layer via _layer_activation_mixer)
                propensity = torch.sigmoid(final_net(X_neighbor))

                # Propensity matrix
                propensity_matrix[i, t] = propensity.squeeze()
        
        return propensity_matrix

def test_mar_missingness():
    """Test function to verify MAR_missingness implementation"""
    print("Testing MAR_missingness implementation...")
    
    # Create test configuration
    test_config = {
        "num_layers_upper": 2,
        "hidden_lower": 2,
        "hidden_upper": 5,
        "activation": "relu",
        "N": 10,
        "T": 8,
        "row_neighbor_upper": 3,
        "col_neighbor_upper": 3,
        "seed": 42,
        "neighbor_type": "random"
    }
    
    # Create model and test data
    model = MAR_missingness(test_config)
    X_test = torch.randn(test_config['N'], test_config['T'])
    
    print(f"Input shape: {X_test.shape}")
    
    # Test forward pass
    try:
        propensities = model(X_test)
        print(f"Output shape: {propensities.shape}")
        print(f"Propensity range: [{propensities.min():.4f}, {propensities.max():.4f}]")
        print(f"Mean propensity: {propensities.mean():.4f}")
        print("✓ Forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main function with basic testing"""
    print("MAR Missingness Pattern Generator")
    print("=" * 50)
    
    success = test_mar_missingness()
    
    if success:
        print("\n" + "=" * 50)
        print("Example usage:")
        print("model = MAR_missingness(mar_config)")
        print("propensities = model(X)")
        print("missing_mask = torch.bernoulli(propensities)")
    else:
        print("Tests failed - please check implementation")

if __name__ == "__main__":
    main()