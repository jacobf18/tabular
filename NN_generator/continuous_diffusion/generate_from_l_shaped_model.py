#!/usr/bin/env python3
"""
Generate L-shaped matrices from the trained diffusion model.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List
import sys
import os

# Add the current directory to the path so we can import diffusion_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion_model import MinimalDiffusionModel, MinimalDiffusionScheduler

def load_trained_model(model_path: str = 'l_shaped_diffusion_model.pth', device: str = 'cpu'):
    """Load the trained L-shaped diffusion model."""
    print(f"Loading trained model from {model_path}...")
    
    # Initialize model and scheduler
    model = MinimalDiffusionModel(max_size=10).to(device)
    scheduler = MinimalDiffusionScheduler(num_timesteps=20)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    # Check if checkpoint is wrapped in a dict or direct state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    return model, scheduler

def generate_matrices_from_model(
    model,
    scheduler,
    num_samples: int = 16,
    matrix_size: int = 10,
    device: str = 'cpu',
    threshold: float = 0.5
) -> List[np.ndarray]:
    """Generate L-shaped matrices from the trained model."""
    
    model.eval()
    generated_matrices = []
    
    print(f"Generating {num_samples} matrices from trained model...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Start with random noise
            x = torch.randn(1, 1, matrix_size, matrix_size).to(device)
            
            # Denoising process
            for t in reversed(range(scheduler.num_timesteps)):
                # Get noise prediction
                noise_pred = model(x, torch.tensor([t]).to(device))
                
                # Apply scheduler step
                x = scheduler.step(noise_pred, t, x)
            
            # Convert to numpy and apply threshold
            matrix = x.squeeze().cpu().numpy()
            binary_matrix = (matrix > threshold).astype(np.float32)
            
            generated_matrices.append(binary_matrix)
            
            if (i + 1) % 4 == 0:
                print(f"Generated {i + 1}/{num_samples} matrices")
    
    return generated_matrices

def visualize_generated_matrices(matrices: List[np.ndarray], save_path: str = '../figures/generated_l_shaped_matrices.png'):
    """Visualize the generated matrices."""
    num_samples = len(matrices)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, matrix in enumerate(matrices):
        row = i // cols
        col = i % cols
        
        axes[row, col].imshow(matrix, cmap='binary', vmin=0, vmax=1)
        axes[row, col].set_title(f'Generated {i+1}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Generated matrices saved to {save_path}")

def analyze_generated_matrices(matrices: List[np.ndarray]) -> dict:
    """Analyze the generated matrices to see if they maintain L-shape properties."""
    analysis = {
        'total_matrices': len(matrices),
        'matrix_size': matrices[0].shape[0] if matrices else 0,
        'l_shaped_count': 0,
        'l_positions': [],
        'density_stats': []
    }
    
    for i, matrix in enumerate(matrices):
        # Calculate density
        density = np.mean(matrix)
        analysis['density_stats'].append(density)
        
        # Check if it's L-shaped (simplified check)
        # Look for the characteristic pattern: upper-right quadrant should be mostly 0
        rows, cols = matrix.shape
        mid_row, mid_col = rows // 2, cols // 2
        
        upper_right = matrix[:mid_row, mid_col:]
        rest = np.concatenate([matrix[:mid_row, :mid_col].flatten(), 
                              matrix[mid_row:, :].flatten()])
        
        # If upper-right is mostly 0 and rest is mostly 1, it's L-shaped
        if np.mean(upper_right) < 0.3 and np.mean(rest) > 0.7:
            analysis['l_shaped_count'] += 1
            analysis['l_positions'].append(i)
    
    analysis['l_shaped_percentage'] = (analysis['l_shaped_count'] / analysis['total_matrices']) * 100
    analysis['avg_density'] = np.mean(analysis['density_stats'])
    analysis['density_std'] = np.std(analysis['density_stats'])
    
    return analysis

def main():
    """Main function to generate and analyze matrices from the trained model."""
    
    # Check if model file exists
    model_path = '../trained_model/l_shaped_diffusion_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model, scheduler = load_trained_model(model_path, device)
    
    # Generate matrices
    num_samples = 20  # Generate 20 samples
    generated_matrices = generate_matrices_from_model(
        model, scheduler, num_samples=num_samples, device=device
    )
    
    # Visualize results
    visualize_generated_matrices(generated_matrices)
    
    # Analyze results
    analysis = analyze_generated_matrices(generated_matrices)
    
    print("\n=== Analysis Results ===")
    print(f"Total matrices generated: {analysis['total_matrices']}")
    print(f"L-shaped matrices: {analysis['l_shaped_count']} ({analysis['l_shaped_percentage']:.1f}%)")
    print(f"Average density: {analysis['avg_density']:.3f} ± {analysis['density_std']:.3f}")
    
    if analysis['l_positions']:
        print(f"L-shaped matrices found at indices: {analysis['l_positions']}")
    
    # Save some matrices as numpy arrays for further analysis
    np.save('../figures/generated_l_shaped_matrices.npy', np.array(generated_matrices))
    print("Generated matrices saved as '../figures/generated_l_shaped_matrices.npy'")

if __name__ == "__main__":
    main() 