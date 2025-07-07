#!/usr/bin/env python3
"""
Visualize L-Shaped Matrix Examples
Show examples of the training data before running the full experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def generate_l_shaped_matrix(rows: int, cols: int, l_row: int, l_col: int) -> np.ndarray:
    """
    Generate an L-shaped binary matrix.
    
    Args:
        rows: Total number of rows
        cols: Total number of columns  
        l_row: Row threshold (0-indexed) - rows 0 to l_row-1 are "upper" part
        l_col: Column threshold (0-indexed) - columns 0 to l_col-1 are "left" part
    
    Returns:
        Binary matrix where:
        - Upper-right quadrant (rows 0:l_row, cols l_col:) = 0
        - Everything else = 1
    """
    matrix = np.ones((rows, cols), dtype=np.float32)
    
    # Set the upper-right quadrant to zero
    # This is the region: rows 0 to l_row-1, columns l_col to end
    matrix[:l_row, l_col:] = 0.0
    
    return matrix

def visualize_l_shaped_examples():
    """Generate and visualize various L-shaped matrix examples."""
    
    matrix_size = 10
    num_examples = 20
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate examples
    examples = []
    l_positions = []
    
    for i in range(num_examples):
        # Randomly choose L position (avoiding edges to ensure L is visible)
        l_row = random.randint(1, matrix_size - 2)  # Not first or last row
        l_col = random.randint(1, matrix_size - 2)  # Not first or last column
        
        # Generate L-shaped matrix
        l_matrix = generate_l_shaped_matrix(matrix_size, matrix_size, l_row, l_col)
        examples.append(l_matrix)
        l_positions.append((l_row, l_col))
    
    # Create visualization
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(num_examples):
        matrix = examples[i]
        l_row, l_col = l_positions[i]
        
        # Calculate the zero region dimensions
        zero_rows = l_row  # Number of rows in the zero region
        zero_cols = matrix_size - l_col  # Number of columns in the zero region
        
        axes[i].imshow(matrix, cmap='binary', vmin=0, vmax=1)
        axes[i].set_title(f'Threshold({l_row},{l_col})\nZero region: {zero_rows}×{zero_cols}')
        axes[i].axis('off')
        
        # Add grid to show matrix structure
        axes[i].set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
        axes[i].grid(True, which='minor', color='gray', linewidth=0.5, alpha=0.3)
    
    plt.suptitle(f'L-Shaped Matrix Examples ({matrix_size}x{matrix_size})\nWhite = 1, Black = 0 (upper-right quadrant)', fontsize=16)
    plt.tight_layout()
    plt.savefig('l_shaped_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"Generated {num_examples} L-shaped {matrix_size}x{matrix_size} matrices")
    print("\nL-position statistics:")
    l_rows = [pos[0] for pos in l_positions]
    l_cols = [pos[1] for pos in l_positions]
    print(f"Row positions: min={min(l_rows)}, max={max(l_rows)}, mean={np.mean(l_rows):.1f}")
    print(f"Col positions: min={min(l_cols)}, max={max(l_cols)}, mean={np.mean(l_cols):.1f}")
    
    # Show some specific examples with their properties
    print("\nSample matrices with properties:")
    for i in range(min(5, num_examples)):
        l_row, l_col = l_positions[i]
        zero_rows = l_row
        zero_cols = matrix_size - l_col
        total_ones = np.sum(examples[i])
        
        print(f"  Matrix {i+1}: Threshold({l_row},{l_col}), "
              f"Zero region: {zero_rows}×{zero_cols}, "
              f"Total 1s: {total_ones}")

def visualize_l_shape_variations():
    """Show how L-shape thickness and position can vary."""
    
    matrix_size = 10
    
    # Create different L-shape variations
    variations = [
        # Different positions
        (1, 1, "Top-left corner"),
        (3, 3, "Middle"),
        (7, 7, "Bottom-right corner"),
        (2, 6, "Top-right area"),
        (6, 2, "Bottom-left area"),
        (4, 5, "Center-right"),
        (5, 4, "Center-bottom"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (l_row, l_col, description) in enumerate(variations):
        matrix = generate_l_shaped_matrix(matrix_size, matrix_size, l_row, l_col)
        
        axes[i].imshow(matrix, cmap='binary', vmin=0, vmax=1)
        axes[i].set_title(f'{description}\nL({l_row},{l_col})')
        axes[i].axis('off')
        
        # Add grid
        axes[i].set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
        axes[i].grid(True, which='minor', color='gray', linewidth=0.5, alpha=0.3)
    
    # Hide the last subplot if we have odd number of variations
    if len(variations) < 8:
        axes[-1].axis('off')
    
    plt.suptitle('L-Shape Position Variations\nWhite = 1, Black = 0 (upper-right quadrant)', fontsize=16)
    plt.tight_layout()
    plt.savefig('l_shaped_variations.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== L-Shaped Matrix Examples ===")
    print("Matrix size: 10x10 (fixed)")
    print("White pixels = 1 (main region)")
    print("Black pixels = 0 (upper-right quadrant)")
    print("Pattern: Choose row and column thresholds, upper-right quadrant becomes zero")
    print()
    
    # Show random examples
    visualize_l_shaped_examples()
    
    print("\n" + "="*50)
    print()
    
    # Show position variations
    visualize_l_shape_variations()
    
    print("\nThese examples show the training data that will be used for the diffusion model.")
    print("The model will learn to generate L-shaped patterns with varying positions.") 