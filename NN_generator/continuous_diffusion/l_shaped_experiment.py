#!/usr/bin/env python3
"""
L-Shaped Matrix Experiment for Diffusion Model
Generate stylized L-shaped binary matrices and train diffusion model to reproduce them.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from tqdm import tqdm
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

def generate_l_shaped_dataset(
    num_samples: int = 10000,
    matrix_size: int = 10,
    random_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate a large dataset of L-shaped matrices.
    
    Args:
        num_samples: Number of L-shaped matrices to generate
        matrix_size: Size of square matrices (e.g., 10 for 10x10)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (data_matrices, mask_matrices) where both are L-shaped
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    data_matrices = []
    mask_matrices = []
    
    print(f"Generating {num_samples} L-shaped {matrix_size}x{matrix_size} matrices...")
    
    for i in tqdm(range(num_samples)):
        # Randomly choose L position (avoiding edges to ensure L is visible)
        l_row = random.randint(1, matrix_size - 2)  # Not first or last row
        l_col = random.randint(1, matrix_size - 2)  # Not first or last column
        
        # Generate L-shaped matrix
        l_matrix = generate_l_shaped_matrix(matrix_size, matrix_size, l_row, l_col)
        
        # For this experiment, both data and mask are the same L-shaped matrix
        # This simplifies the learning task - model just needs to reproduce the L-shape
        data_matrices.append(l_matrix)
        mask_matrices.append(l_matrix.copy())
    
    print(f"Generated {len(data_matrices)} L-shaped matrices")
    print(f"Sample L-shape parameters: row={l_row}, col={l_col}")
    
    return data_matrices, mask_matrices

def visualize_l_shaped_samples(data_matrices: List[np.ndarray], num_samples: int = 16):
    """Visualize sample L-shaped matrices."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(data_matrices))):
        matrix = data_matrices[i]
        axes[i].imshow(matrix, cmap='binary', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('l_shaped_training_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_l_shaped_patterns(matrices: List[np.ndarray]) -> dict:
    """Analyze the L-shaped patterns in the dataset."""
    analysis = {
        'total_matrices': len(matrices),
        'matrix_size': matrices[0].shape[0] if matrices else 0,
        'l_positions': [],
        'l_sizes': []
    }
    
    for matrix in matrices:
        # Find the L position by looking for the corner
        rows, cols = matrix.shape
        l_row, l_col = None, None
        
        # Find the top-left corner of the L
        for r in range(rows):
            for c in range(cols):
                if matrix[r, c] == 1:
                    # Check if this is the corner (has 1s to the right and below)
                    if (c < cols - 1 and matrix[r, c+1] == 1 and 
                        r < rows - 1 and matrix[r+1, c] == 1):
                        l_row, l_col = r, c
                        break
            if l_row is not None:
                break
        
        if l_row is not None:
            analysis['l_positions'].append((l_row, l_col))
            
            # Calculate L dimensions
            # Vertical part length
            vert_length = 0
            for r in range(l_row, rows):
                if matrix[r, l_col] == 1:
                    vert_length += 1
                else:
                    break
            
            # Horizontal part length  
            horiz_length = 0
            for c in range(l_col, cols):
                if matrix[l_row, c] == 1:
                    horiz_length += 1
                else:
                    break
            
            analysis['l_sizes'].append((vert_length, horiz_length))
    
    # Calculate statistics
    if analysis['l_positions']:
        l_rows = [pos[0] for pos in analysis['l_positions']]
        l_cols = [pos[1] for pos in analysis['l_positions']]
        analysis['row_stats'] = {
            'mean': np.mean(l_rows),
            'std': np.std(l_rows),
            'min': np.min(l_rows),
            'max': np.max(l_rows)
        }
        analysis['col_stats'] = {
            'mean': np.mean(l_cols),
            'std': np.std(l_cols),
            'min': np.min(l_cols),
            'max': np.max(l_cols)
        }
    
    return analysis

def train_l_shaped_diffusion_model(
    data_matrices: List[np.ndarray],
    masking_matrices: List[np.ndarray],
    num_epochs: int = 200,
    batch_size: int = 32,
    max_size: int = 10,
    device: str = 'cpu',
    learning_rate: float = 1e-3,
    save_path: str = 'l_shaped_diffusion_model.pth'
):
    """Train diffusion model on L-shaped matrices."""
    
    # Import the diffusion model components
    from diffusion_model import MinimalDiffusionModel, MinimalDiffusionScheduler, MinimalMatrixDataset
    
    print(f"Training diffusion model on {len(data_matrices)} L-shaped matrices...")
    print(f"Matrix size: {max_size}x{max_size}")
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    # Create dataset
    dataset = MinimalMatrixDataset(data_matrices, masking_matrices, max_size=max_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and scheduler
    model = MinimalDiffusionModel(max_size=max_size).to(device)
    scheduler = MinimalDiffusionScheduler(num_timesteps=20)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x, size_info) in enumerate(dataloader):
            x = x.to(device)  # [B, 1, H, W] - only L-shaped matrix
            size_info = size_info.to(device)
            
            # Random timestep
            t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
            
            # Add noise
            x_noisy, noise = scheduler.add_noise(x, t)
            
            # Predict noise - convert t to float for the model
            t_float = t.float().unsqueeze(1) / scheduler.num_timesteps
            predicted = model(x_noisy, t_float, size_info)
            
            # Loss (MSE on noise prediction)
            loss = F.mse_loss(predicted, x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss for L-Shaped Diffusion Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('l_shaped_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, scheduler, losses

def sample_l_shaped_matrices(
    model,
    scheduler,
    num_samples: int = 16,
    matrix_size: int = 10,
    device: str = 'cpu',
    threshold: float = 0.5
) -> List[np.ndarray]:
    """Sample L-shaped matrices from the trained model."""
    
    from diffusion_model import MinimalDiffusionModel
    
    model.eval()
    samples = []
    
    print(f"Sampling {num_samples} L-shaped matrices...")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            # Start with random noise
            x = torch.randn(1, 1, matrix_size, matrix_size, device=device)
            size_info = torch.tensor([[matrix_size, matrix_size]], dtype=torch.float32, device=device)
            
            # Denoising process
            for t in reversed(range(scheduler.num_timesteps)):
                t_tensor = torch.tensor([t], device=device)
                t_float = t_tensor.float().unsqueeze(1) / scheduler.num_timesteps
                predicted = model(x, t_float, size_info)
                x = scheduler.remove_noise(x, t_tensor, predicted)
            
            # Output is [1, 1, H, W], apply sigmoid and threshold
            out = torch.sigmoid(predicted)
            out = (out > threshold).float()
            sample = out.squeeze().cpu().numpy()
            samples.append(sample)
    
    return samples

def evaluate_l_shaped_samples(original_samples: List[np.ndarray], generated_samples: List[np.ndarray]) -> dict:
    """Evaluate how well the generated samples match L-shaped patterns."""
    
    def is_l_shaped(matrix: np.ndarray) -> bool:
        """Check if a matrix is L-shaped."""
        rows, cols = matrix.shape
        
        # Find the corner of potential L
        l_row, l_col = None, None
        for r in range(rows):
            for c in range(cols):
                if matrix[r, c] == 1:
                    # Check if this could be the corner
                    if (c < cols - 1 and matrix[r, c+1] == 1 and 
                        r < rows - 1 and matrix[r+1, c] == 1):
                        l_row, l_col = r, c
                        break
            if l_row is not None:
                break
        
        if l_row is None:
            return False
        
        # Check if it forms a proper L
        # Vertical part should be continuous
        for r in range(l_row, rows):
            if matrix[r, l_col] != 1:
                return False
        
        # Horizontal part should be continuous
        for c in range(l_col, cols):
            if matrix[l_row, c] != 1:
                return False
        
        # Check that there are no other 1s outside the L
        for r in range(rows):
            for c in range(cols):
                if matrix[r, c] == 1:
                    if not (r >= l_row or c >= l_col):
                        return False
        
        return True
    
    # Analyze original samples
    original_l_count = sum(1 for matrix in original_samples if is_l_shaped(matrix))
    original_l_rate = original_l_count / len(original_samples)
    
    # Analyze generated samples
    generated_l_count = sum(1 for matrix in generated_samples if is_l_shaped(matrix))
    generated_l_rate = generated_l_count / len(generated_samples)
    
    # Calculate other metrics
    def calculate_metrics(matrices: List[np.ndarray]) -> dict:
        total_ones = sum(np.sum(matrix) for matrix in matrices)
        avg_ones = total_ones / len(matrices)
        
        # Calculate connectivity (how many 1s are adjacent to other 1s)
        connectivity_scores = []
        for matrix in matrices:
            connected = 0
            total = 0
            rows, cols = matrix.shape
            
            for r in range(rows):
                for c in range(cols):
                    if matrix[r, c] == 1:
                        total += 1
                        # Check neighbors
                        neighbors = 0
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < rows and 0 <= nc < cols and 
                                matrix[nr, nc] == 1):
                                neighbors += 1
                        if neighbors > 0:
                            connected += 1
            
            if total > 0:
                connectivity_scores.append(connected / total)
            else:
                connectivity_scores.append(0)
        
        return {
            'avg_ones': avg_ones,
            'avg_connectivity': np.mean(connectivity_scores),
            'connectivity_std': np.std(connectivity_scores)
        }
    
    original_metrics = calculate_metrics(original_samples)
    generated_metrics = calculate_metrics(generated_samples)
    
    return {
        'original_l_rate': original_l_rate,
        'generated_l_rate': generated_l_rate,
        'original_metrics': original_metrics,
        'generated_metrics': generated_metrics,
        'l_shape_accuracy': generated_l_rate
    }

def visualize_comparison(original_samples: List[np.ndarray], generated_samples: List[np.ndarray], num_samples: int = 8):
    """Visualize original vs generated L-shaped matrices."""
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        if i < len(original_samples):
            axes[0, i].imshow(original_samples[i], cmap='binary', vmin=0, vmax=1)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
        
        if i < len(generated_samples):
            axes[1, i].imshow(generated_samples[i], cmap='binary', vmin=0, vmax=1)
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Generated', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('l_shaped_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main experiment function."""
    print("=== L-Shaped Matrix Diffusion Experiment ===")
    
    # Parameters - LARGER EXPERIMENT
    num_samples = 5000  # Increased from 500
    matrix_size = 10
    num_epochs = 100   # Increased from 50
    batch_size = 32    # Increased from 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Training samples: {num_samples}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Generate L-shaped dataset
    data_matrices, mask_matrices = generate_l_shaped_dataset(
        num_samples=num_samples,
        matrix_size=matrix_size,
        random_seed=42
    )
    
    # Analyze the dataset
    analysis = analyze_l_shaped_patterns(data_matrices)
    print("\nDataset Analysis:")
    print(f"Total matrices: {analysis['total_matrices']}")
    print(f"Matrix size: {analysis['matrix_size']}")
    print(f"L position stats - Row: {analysis['row_stats']}")
    print(f"L position stats - Col: {analysis['col_stats']}")
    
    # Visualize sample matrices
    visualize_l_shaped_samples(data_matrices[:16])
    
    # Train diffusion model
    model, scheduler, losses = train_l_shaped_diffusion_model(
        data_matrices=data_matrices,
        masking_matrices=mask_matrices,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_size=matrix_size,
        device=device,
        learning_rate=1e-3,
        save_path='../trained_model/l_shaped_diffusion_model.pth'
    )
    
    # Sample from trained model
    generated_samples = sample_l_shaped_matrices(
        model=model,
        scheduler=scheduler,
        num_samples=100,  # Increased for more robust evaluation
        matrix_size=matrix_size,
        device=device,
        threshold=0.5
    )
    
    # Evaluate results
    evaluation = evaluate_l_shaped_samples(data_matrices[:100], generated_samples)
    print("\nEvaluation Results:")
    print(f"Original L-shape rate: {evaluation['original_l_rate']:.3f}")
    print(f"Generated L-shape rate: {evaluation['generated_l_rate']:.3f}")
    print(f"L-shape accuracy: {evaluation['l_shape_accuracy']:.3f}")
    print(f"Original avg ones: {evaluation['original_metrics']['avg_ones']:.2f}")
    print(f"Generated avg ones: {evaluation['generated_metrics']['avg_ones']:.2f}")
    print(f"Original connectivity: {evaluation['original_metrics']['avg_connectivity']:.3f}")
    print(f"Generated connectivity: {evaluation['generated_metrics']['avg_connectivity']:.3f}")
    
    # Visualize comparison
    visualize_comparison(data_matrices[:8], generated_samples[:8])
    
    print("\nExperiment completed!")
    print(f"Model saved as: ../trained_model/l_shaped_diffusion_model.pth")
    print(f"Visualizations saved as: l_shaped_*.png")

if __name__ == "__main__":
    main() 