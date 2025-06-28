#!/usr/bin/env python3
"""
Minimal Diffusion Model for Matrix Generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from tqdm import tqdm

class MinimalDiffusionModel(nn.Module):
    """Complex diffusion model with more capacity for learning patterns."""
    
    def __init__(self, max_size: int = 32):
        super().__init__()
        self.max_size = max_size
        
        # Time embedding - more complex
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Size embedding - more complex
        self.size_embed = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Much more complex CNN architecture
        # Initial convolution
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Multiple convolutional blocks with residual connections
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Final convolution
        self.conv_final = nn.Conv2d(64, 2, 3, padding=1)
        
        self.relu = nn.ReLU()
        
        # Downsample and upsample for multi-scale processing
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x, t=None, size_info=None):
        # Time and size embeddings
        if t is not None:
            t_emb = self.time_embed(t)  # [B, 64]
            t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
            t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])  # [B, 64, H, W]
        else:
            t_emb = torch.zeros(x.shape[0], 64, x.shape[2], x.shape[3], device=x.device)
            
        if size_info is not None:
            size_emb = self.size_embed(size_info)  # [B, 64]
            size_emb = size_emb.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
            size_emb = size_emb.expand(-1, -1, x.shape[2], x.shape[3])  # [B, 64, H, W]
        else:
            size_emb = torch.zeros(x.shape[0], 64, x.shape[2], x.shape[3], device=x.device)
        
        # Initial convolution
        h = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H, W]
        
        # Add embeddings
        h = h + t_emb + size_emb
        
        # First conv block with residual connection
        residual = h
        h = self.conv_block1(h)
        h = self.relu(h + residual)  # Residual connection
        
        # Second conv block with downsampling
        h_down = self.downsample(h)  # [B, 64, H/2, W/2]
        h_down = self.conv_block2(h_down)  # [B, 128, H/2, W/2]
        h_down = self.relu(h_down)
        
        # Third conv block
        h_down = self.conv_block3(h_down)  # [B, 128, H/2, W/2]
        h_down = self.relu(h_down)
        
        # Fourth conv block with upsampling
        h_up = self.conv_block4(h_down)  # [B, 64, H/2, W/2]
        h_up = self.relu(h_up)
        h_up = self.upsample(h_up)  # [B, 64, H, W]
        
        # Combine with original features
        h = h + h_up  # Skip connection
        
        # Final convolution
        h = self.conv_final(h)  # [B, 2, H, W]
        
        # Split into data and mask channels
        data_channel = h[:, 0:1, :, :]  # [B, 1, H, W] - continuous data
        mask_channel = torch.sigmoid(h[:, 1:2, :, :])  # [B, 1, H, W] - binary-like mask (0-1)
        
        # Combine back
        output = torch.cat([data_channel, mask_channel], dim=1)  # [B, 2, H, W]
        
        return output

class MinimalDiffusionScheduler:
    """Minimal noise scheduler."""
    
    def __init__(self, num_timesteps: int = 20):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise
    
    def remove_noise(self, x_noisy, t, predicted_noise):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        x_start = (x_noisy - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        return x_start

class MinimalMatrixDataset(Dataset):
    """Minimal dataset for matrix pairs."""
    
    def __init__(self, data_matrices, masking_matrices, max_size=32):
        self.data_matrices = data_matrices
        self.masking_matrices = masking_matrices
        self.max_size = max_size
        
        # Pad matrices
        self.padded_data = []
        self.padded_masks = []
        self.size_info = []
        
        for data, mask in zip(data_matrices, masking_matrices):
            padded_data = self.pad_matrix(data, max_size)
            padded_mask = self.pad_matrix(mask, max_size)
            
            self.padded_data.append(padded_data)
            self.padded_masks.append(padded_mask)
            self.size_info.append([data.shape[0], data.shape[1]])
    
    def pad_matrix(self, matrix, max_size):
        rows, cols = matrix.shape
        padded = np.zeros((max_size, max_size))
        padded[:rows, :cols] = matrix
        return padded
    
    def __len__(self):
        return len(self.data_matrices)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.padded_data[idx])
        mask = torch.FloatTensor(self.padded_masks[idx])
        size_info = torch.FloatTensor(self.size_info[idx])
        
        x = torch.stack([data, mask], dim=0)
        return x, size_info

def generate_staggered_adoption_data(
    num_samples: int = 50,
    size_range: Tuple[int, int] = (8, 12),
    random_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate training data with fully heterogeneous staggered adoption patterns.
    For each row, the transition from 1s to 0s is at a random column.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data_matrices = []
    masking_matrices = []
    
    for i in range(num_samples):
        n_rows = np.random.randint(size_range[0], size_range[1] + 1)
        n_cols = np.random.randint(size_range[0], size_range[1] + 1)
        data = np.random.randn(n_rows, n_cols)
        mask = np.ones((n_rows, n_cols), dtype=int)
        for r in range(n_rows):
            transition = np.random.randint(0, n_cols + 1)  # can be 0 (all missing) to n_cols (all observed)
            mask[r, transition:] = 0
        data_matrices.append(data)
        masking_matrices.append(mask)
    return data_matrices, masking_matrices

def generate_training_data(
    num_samples: int = 50,
    size_range: Tuple[int, int] = (8, 12),
    random_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate minimal synthetic training data with binary masks."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data_matrices = []
    masking_matrices = []
    
    for i in range(num_samples):
        # Generate different sizes for rows and columns
        num_rows = np.random.randint(*size_range)
        num_cols = np.random.randint(*size_range)
        
        # Generate random data matrix (rectangular)
        data_matrix = np.random.randn(num_rows, num_cols)
        
        # Generate binary mask matrix with some structure
        # Create sparse patterns with some clustering
        mask_matrix = np.zeros((num_rows, num_cols))
        
        # Add some random clusters of 1s
        num_clusters = np.random.randint(2, 6)
        for _ in range(num_clusters):
            # Random cluster center
            center_row = np.random.randint(1, num_rows-1)
            center_col = np.random.randint(1, num_cols-1)
            
            # Random cluster size
            cluster_size = np.random.randint(1, min(4, min(num_rows, num_cols)//2))
            
            # Fill cluster
            for dr in range(-cluster_size, cluster_size+1):
                for dc in range(-cluster_size, cluster_size+1):
                    r, c = center_row + dr, center_col + dc
                    if 0 <= r < num_rows and 0 <= c < num_cols:
                        if np.random.random() < 0.7:  # 70% chance to fill
                            mask_matrix[r, c] = 1
        
        # Ensure mask is truly binary
        mask_matrix = (mask_matrix > 0).astype(np.float32)
        
        data_matrices.append(data_matrix)
        masking_matrices.append(mask_matrix)
    
    return data_matrices, masking_matrices

def train_minimal_diffusion_model(
    data_matrices,
    masking_matrices,
    num_epochs=100,  # More epochs for complex model
    batch_size=16,  # Larger batch size for complex model
    max_size=32,
    device='cpu',
    learning_rate=5e-4,  # Lower learning rate for complex model
    save_path='complex_diffusion_model.pth'
):
    """Train the complex diffusion model."""
    print(f"Training complex diffusion model on {len(data_matrices)} samples")
    
    # Create dataset and dataloader
    dataset = MinimalMatrixDataset(data_matrices, masking_matrices, max_size=max_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model and scheduler
    model = MinimalDiffusionModel(max_size=max_size).to(device)
    scheduler = MinimalDiffusionScheduler(num_timesteps=20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    
    # Training loop
    model.train()
    losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for x, size_info in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device)
            size_info = size_info.to(device)
            
            batch_size = x.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            
            # Add noise
            x_noisy, noise = scheduler.add_noise(x, t)
            
            # Predict noise
            t_float = t.float().unsqueeze(1) / scheduler.num_timesteps
            predicted_noise = model(x_noisy, t_float, size_info)
            
            # Compute loss
            loss = criterion(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}")
    
    print(f"Training completed! Best loss: {best_loss:.6f}")
    print(f"Model saved to {save_path}")
    
    return model

def sample_from_minimal_diffusion(
    model,
    scheduler,
    size_info,
    num_steps=10,
    device='cpu',
    threshold=0.5
):
    """Sample from the minimal diffusion model."""
    model.eval()
    size_info = size_info.to(device)
    
    # Ensure size_info has the right shape [B, 2] - if it's [2], add batch dim
    if size_info.dim() == 1:
        size_info = size_info.unsqueeze(0)
    
    # Start from pure noise
    x = torch.randn(1, 2, model.max_size, model.max_size, device=device)
    
    # Denoising process
    step_size = scheduler.num_timesteps // num_steps
    
    with torch.no_grad():
        for i in range(num_steps - 1, -1, -1):
            t = torch.tensor([i * step_size], device=device)
            t_float = t.float() / scheduler.num_timesteps
            
            # Predict noise - size_info already has batch dimension
            predicted_noise = model(x, t_float.unsqueeze(0), size_info)
            
            # Remove noise
            x = scheduler.remove_noise(x, t, predicted_noise)
            
            # Add noise for next step (except last step)
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(scheduler.betas[t]) * noise
    
    # Threshold the mask channel to get binary values
    data_channel = x[:, 0:1, :, :]  # Keep data as continuous
    mask_channel = (x[:, 1:2, :, :] > threshold).float()  # Threshold to binary
    
    # Combine back
    x = torch.cat([data_channel, mask_channel], dim=1)
    
    return x

def visualize_staggered_patterns(original_masks, generated_masks, num_samples=5):
    """Visualize and compare original vs generated staggered adoption patterns."""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        if i < len(original_masks) and i < len(generated_masks):
            # Original pattern
            axes[0, i].imshow(original_masks[i], cmap='RdYlBu_r', aspect='auto')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Units')
            
            # Generated pattern
            axes[1, i].imshow(generated_masks[i], cmap='RdYlBu_r', aspect='auto')
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Units')
    
    plt.tight_layout()
    plt.show()

def analyze_staggered_patterns(masks):
    """Analyze staggered adoption patterns in masks."""
    patterns = []
    
    for mask in masks:
        # Handle 4D tensors from diffusion model [B, C, H, W]
        if isinstance(mask, torch.Tensor) and mask.dim() == 4:
            # Extract mask channel and convert to numpy
            mask = mask[0, 1, :, :].cpu().numpy()  # [H, W]
        elif isinstance(mask, torch.Tensor) and mask.dim() == 3:
            # Handle [C, H, W] tensors
            mask = mask[1, :, :].cpu().numpy()  # [H, W]
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            # Handle [H, W] tensors
            mask = mask.cpu().numpy()
        
        rows, cols = mask.shape
        dropout_points = []
        
        for unit in range(rows):
            # Find when this unit starts missing data
            missing_start = None
            for t in range(cols):
                if mask[unit, t] == 0:
                    missing_start = t
                    break
            
            if missing_start is not None:
                dropout_points.append(missing_start)
            else:
                dropout_points.append(cols)  # Never drops out
        
        patterns.append(dropout_points)
    
    return patterns

# Alias functions for compatibility
DiffusionUNet = MinimalDiffusionModel
DiffusionScheduler = MinimalDiffusionScheduler
MatrixDataset = MinimalMatrixDataset
DiffusionTrainer = None  # Not used in minimal version
train_diffusion_model = train_minimal_diffusion_model
sample_from_diffusion = sample_from_minimal_diffusion 