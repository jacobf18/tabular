#!/usr/bin/env python3
"""
Discrete Diffusion Model for Staggered Adoption Patterns
Based on MDLM (Masked Diffusion Language Models) principles
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from tqdm import tqdm
import math

class DiscreteDiffusionScheduler:
    """Discrete diffusion scheduler using masking as the noise process."""
    
    def __init__(self, num_timesteps: int = 100):
        self.num_timesteps = num_timesteps
        
        # Linear schedule for alpha_t (similar to MDLM)
        # alpha_t decreases from ~1 to ~0 over timesteps
        self.alphas = torch.linspace(0.99, 0.01, num_timesteps)
        
        # Cumulative alpha for efficient computation
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Time derivative for loss computation
        self.alpha_derivatives = torch.diff(self.alphas, prepend=torch.tensor([1.0]))
    
    def add_noise(self, x_start, t):
        """
        Add discrete noise (masking) to the input.
        x_start: [batch, seq_len] with values 0 (missing) or 1 (observed)
        t: [batch] timesteps
        """
        batch_size, seq_len = x_start.shape
        device = x_start.device
        
        # Get alpha values for this timestep
        alpha_t = self.alphas_cumprod[t].reshape(-1, 1)  # [batch, 1]
        
        # Create mask token (value 2 represents [MASK])
        mask_token = torch.full_like(x_start, 2)
        
        # Sample from categorical distribution
        # P(z_t = x_start) = alpha_t, P(z_t = mask) = 1 - alpha_t
        mask_prob = 1.0 - alpha_t
        mask_indicator = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Apply masking
        x_noisy = torch.where(mask_indicator, mask_token, x_start)
        
        return x_noisy, mask_indicator
    
    def get_loss_weight(self, t):
        """Get loss weight for timestep t (alpha_t' / (1 - alpha_t))."""
        alpha_t = self.alphas_cumprod[t]
        alpha_derivative = self.alpha_derivatives[t]
        return alpha_derivative / (1.0 - alpha_t)

class TransformerEncoder(nn.Module):
    """Simple transformer encoder for discrete diffusion."""
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Time embedding (like MDLM)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Size embedding for variable matrix sizes
        self.size_embed = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, t, size_info=None):
        """
        x: [batch, seq_len] token indices
        t: [batch] timesteps (normalized to [0, 1])
        size_info: [batch, 2] matrix dimensions [rows, cols]
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token embeddings
        x_emb = self.token_embed(x)  # [batch, seq_len, d_model]
        
        # Positional embeddings (handle variable sequence length)
        if seq_len <= self.max_seq_len:
            pos_emb = self.pos_embed[:, :seq_len, :]  # [1, seq_len, d_model]
        else:
            # If sequence is longer than max_seq_len, truncate or use interpolation
            pos_emb = self.pos_embed[:, :self.max_seq_len, :]  # [1, max_seq_len, d_model]
            # Pad or truncate to match sequence length
            if seq_len > self.max_seq_len:
                x_emb = x_emb[:, :self.max_seq_len, :]
                seq_len = self.max_seq_len
        
        x_emb = x_emb + pos_emb
        
        # Time embeddings
        t_emb = self.time_embed(t.unsqueeze(-1).float())  # [batch, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
        x_emb = x_emb + t_emb
        
        # Size embeddings (if provided)
        if size_info is not None:
            size_emb = self.size_embed(size_info)  # [batch, d_model]
            size_emb = size_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
            x_emb = x_emb + size_emb
        
        # Transformer encoding
        x_emb = self.transformer(x_emb)
        
        # Output projection
        logits = self.output_proj(x_emb)  # [batch, seq_len, vocab_size]
        
        return logits

class DiscreteDiffusionModel(nn.Module):
    """Discrete diffusion model with SUBS parameterization (like MDLM)."""
    
    def __init__(self, vocab_size=3, d_model=128, nhead=8, num_layers=6, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.transformer = TransformerEncoder(vocab_size, d_model, nhead, num_layers, max_seq_len)
        
    def forward(self, x, t, size_info=None):
        """
        Forward pass with SUBS parameterization.
        x: [batch, seq_len] with values 0, 1, 2 (missing, observed, mask)
        """
        batch_size, seq_len = x.shape
        
        # Get logits from transformer
        logits = self.transformer(x, t, size_info)  # [batch, seq_len, vocab_size]
        
        # Apply SUBS parameterization (like MDLM):
        # 1. Zero masking probabilities for [MASK] token (index 2)
        logits[:, :, 2] = float('-inf')
        
        # 2. Carry-over unmasking for observed tokens (index 1)
        # If input is observed (1), force output to be observed
        observed_mask = (x == 1).unsqueeze(-1)  # [batch, seq_len, 1]
        logits = torch.where(observed_mask, 
                           torch.tensor([float('-inf'), 0.0, float('-inf')], device=logits.device),
                           logits)
        
        return logits

class MatrixSequenceDataset(Dataset):
    """Dataset that converts matrices to sequences for discrete diffusion."""
    
    def __init__(self, data_matrices, masking_matrices, max_size=32):
        self.data_matrices = data_matrices
        self.masking_matrices = masking_matrices
        self.max_size = max_size
        
        # Convert matrices to sequences
        self.sequences = []
        self.size_info = []
        
        for data, mask in zip(data_matrices, masking_matrices):
            # For staggered adoption, we only care about the mask pattern
            # Convert mask matrix to sequence: [unit1_t1, unit1_t2, ..., unitM_tN]
            rows, cols = mask.shape
            sequence = mask.flatten().astype(np.int64)  # 0=missing, 1=observed
            
            # Pad sequence to max length
            max_seq_len = max_size * max_size
            if len(sequence) < max_seq_len:
                sequence = np.pad(sequence, (0, max_seq_len - len(sequence)), 
                                mode='constant', constant_values=0)
            else:
                sequence = sequence[:max_seq_len]
            
            self.sequences.append(sequence)
            self.size_info.append([rows, cols])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.LongTensor(self.sequences[idx])
        size_info = torch.FloatTensor(self.size_info[idx])
        return sequence, size_info

def generate_staggered_adoption_data(
    num_samples: int = 100,
    size_range: Tuple[int, int] = (8, 16),
    random_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate training data with staggered adoption missing patterns.
    Optimized for discrete diffusion training.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data_matrices = []
    masking_matrices = []
    
    for i in range(num_samples):
        # Generate different sizes for rows (units) and columns (time points)
        num_units = np.random.randint(*size_range)
        num_timepoints = np.random.randint(*size_range)
        
        # Generate random data matrix (rectangular)
        data_matrix = np.random.randn(num_units, num_timepoints)
        
        # Generate staggered adoption mask matrix
        mask_matrix = np.ones((num_units, num_timepoints))  # Start with all 1s (no missing data)
        
        # Create varied staggered adoption patterns
        pattern_type = np.random.choice(['random', 'early', 'late', 'mixed'])
        
        if pattern_type == 'random':
            # Random staggered adoption - each unit has random missing start point
            for unit in range(num_units):
                if np.random.random() < 0.7:  # 70% chance unit has missing data
                    start_missing = np.random.randint(1, num_timepoints)
                    mask_matrix[unit, start_missing:] = 0
                    
        elif pattern_type == 'early':
            # Early adoption - most units start missing data early
            for unit in range(num_units):
                if np.random.random() < 0.8:  # 80% chance unit has missing data
                    start_missing = np.random.randint(1, max(2, num_timepoints // 3))
                    mask_matrix[unit, start_missing:] = 0
                    
        elif pattern_type == 'late':
            # Late adoption - most units start missing data late
            for unit in range(num_units):
                if np.random.random() < 0.8:  # 80% chance unit has missing data
                    start_missing = np.random.randint(num_timepoints // 2, num_timepoints)
                    mask_matrix[unit, start_missing:] = 0
                    
        else:  # mixed
            # Mixed pattern - some early, some late, some random
            for unit in range(num_units):
                if np.random.random() < 0.6:  # 60% chance unit has missing data
                    pattern = np.random.choice(['early', 'late', 'random'])
                    if pattern == 'early':
                        start_missing = np.random.randint(1, max(2, num_timepoints // 3))
                    elif pattern == 'late':
                        start_missing = np.random.randint(num_timepoints // 2, num_timepoints)
                    else:  # random
                        start_missing = np.random.randint(1, num_timepoints)
                    mask_matrix[unit, start_missing:] = 0
        
        # Ensure mask is truly binary
        mask_matrix = mask_matrix.astype(np.float32)
        
        data_matrices.append(data_matrix)
        masking_matrices.append(mask_matrix)
    
    return data_matrices, masking_matrices

def train_discrete_diffusion_model(
    data_matrices,
    masking_matrices,
    num_epochs=200,
    batch_size=32,
    max_size=32,
    device='cpu',
    learning_rate=1e-4,
    save_path='discrete_diffusion_model.pth'
):
    """Train the discrete diffusion model."""
    print(f"Training discrete diffusion model on {len(data_matrices)} samples")
    
    # Create dataset and dataloader
    dataset = MatrixSequenceDataset(data_matrices, masking_matrices, max_size=max_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model and scheduler
    model = DiscreteDiffusionModel(vocab_size=3, d_model=128, nhead=8, num_layers=6).to(device)
    scheduler = DiscreteDiffusionScheduler(num_timesteps=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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
            
            # Add discrete noise (masking)
            x_noisy, mask_indicator = scheduler.add_noise(x, t)
            
            # Get loss weights
            loss_weights = scheduler.get_loss_weight(t)  # [batch]
            
            # Predict logits
            logits = model(x_noisy, t, size_info)  # [batch, seq_len, vocab_size]
            
            # Compute loss only on masked positions (like MDLM)
            # Create target: original values at masked positions
            targets = x.clone()  # [batch, seq_len]
            
            # Only compute loss on masked positions
            masked_logits = logits[mask_indicator]  # [num_masked, vocab_size]
            masked_targets = targets[mask_indicator]  # [num_masked]
            
            if len(masked_logits) > 0:
                # Cross entropy loss on masked positions
                loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
                
                # Apply loss weights
                masked_weights = loss_weights.repeat_interleave(mask_indicator.sum(dim=1))
                loss = (loss * masked_weights).mean()
            else:
                loss = torch.tensor(0.0, device=device)
            
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
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}")
    
    print(f"Training completed! Best loss: {best_loss:.6f}")
    print(f"Model saved to {save_path}")
    
    return model

def sample_from_discrete_diffusion(
    model,
    scheduler,
    size_info,
    num_steps=50,
    device='cpu'
):
    """Sample from the discrete diffusion model."""
    model.eval()
    size_info = size_info.to(device)
    
    # Get sequence length from size info
    rows, cols = size_info[0].cpu().numpy()
    seq_len = int(rows * cols)
    
    # Start from all masked tokens
    x = torch.full((1, seq_len), 2, dtype=torch.long, device=device)  # All [MASK] tokens
    
    # Denoising process
    step_size = scheduler.num_timesteps // num_steps
    
    with torch.no_grad():
        for i in range(num_steps - 1, -1, -1):
            t = torch.tensor([i * step_size], device=device)
            
            # Predict logits
            logits = model(x, t, size_info.unsqueeze(0))  # [1, seq_len, vocab_size]
            
            # Sample from categorical distribution
            probs = F.softmax(logits, dim=-1)
            
            # Only sample for masked positions
            masked_positions = (x[0] == 2)
            if masked_positions.any():
                # Sample new tokens for masked positions
                masked_probs = probs[0, masked_positions]  # [num_masked, vocab_size]
                sampled_tokens = torch.multinomial(masked_probs, 1).squeeze(-1)  # [num_masked]
                
                # Update masked positions
                x[0, masked_positions] = sampled_tokens
    
    # Convert back to matrix
    sequence = x[0].cpu().numpy()[:seq_len]  # Remove padding
    mask_matrix = sequence.reshape(int(rows), int(cols))
    
    return mask_matrix

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