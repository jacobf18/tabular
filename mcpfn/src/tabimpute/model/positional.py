import torch
import math
from torch import nn

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=100, damping_factor=0.1, dimension='row'):
        """
        Sinusoidal positional embedding for rows or columns.
        
        Args:
            embedding_size: Size of the embedding vector
            max_len: Initial maximum length for pre-computed embeddings
            damping_factor: Scaling factor for the positional embeddings
            dimension: Either 'row' or 'column' to specify which dimension to apply embeddings to
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.damping_factor = damping_factor
        self.dimension = dimension
        
        if dimension not in ['row', 'column']:
            raise ValueError(f"dimension must be 'row' or 'column', got {dimension}")
        
        # 1. Store the frequency term (div_term) as a buffer.
        # We MUST use the original max_len to define the curve's 'slope'.
        # If we change this later, the embeddings for pos 0-max_len would change, 
        # confusing the model.
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float() * (-math.log(max_len) / embedding_size)
        )
        self.register_buffer('div_term', div_term)
        
        # 2. Pre-compute the initial PE cache
        # We verify if `pe` exists in forward, but initializing it here is good practice.
        self.register_buffer('pe', self._generate_pe(max_len))

    def _generate_pe(self, length):
        """
        Generates positional embeddings for positions [0, length).
        Uses the stored self.div_term to ensure consistency.
        """
        # Ensure we generate on the correct device/dtype
        pe = torch.zeros(length, self.embedding_size, device=self.div_term.device, dtype=self.div_term.dtype)
        position = torch.arange(0, length, dtype=self.div_term.dtype, device=self.div_term.device).unsqueeze(1)
        
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, num_rows, num_cols, embedding_size)
        """
        if self.dimension == 'row':
            _, num_positions, _, _ = x.shape
            # Shape: (1, num_rows, 1, embedding_size) for broadcasting
            unsqueeze_dims = (0, 2)
        else:  # dimension == 'column'
            _, _, num_positions, _ = x.shape
            # Shape: (1, 1, num_cols, embedding_size) for broadcasting
            unsqueeze_dims = (0, 1)
        
        current_max_len = self.pe.size(0)

        # 3. Dynamic Extrapolation
        # If input is longer than our cache, extend the cache
        if num_positions > current_max_len:
            # Generate ONLY the new needed positions (e.g., from 100 to 150)
            # This is more efficient than regenerating the whole matrix
            new_positions = torch.arange(
                current_max_len, num_positions, 
                dtype=torch.float, 
                device=self.pe.device
            ).unsqueeze(1)
            
            new_pe = torch.zeros(
                num_positions - current_max_len, 
                self.embedding_size, 
                device=self.pe.device
            )
            
            new_pe[:, 0::2] = torch.sin(new_positions * self.div_term)
            new_pe[:, 1::2] = torch.cos(new_positions * self.div_term)
            
            # Concatenate and update the buffer so next time it's fast
            self.pe = torch.cat([self.pe, new_pe], dim=0)

        # Slice the embeddings to the current number of positions
        pos_embeddings = self.pe[:num_positions, :].unsqueeze(unsqueeze_dims[0]).unsqueeze(unsqueeze_dims[1]).to(x.device).to(x.dtype)
        
        return x + pos_embeddings * self.damping_factor

class LinearPositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=100, damping_factor=0.1, dimension='row'):
        """
        Learnable linear positional embedding for rows or columns.
        
        Args:
            embedding_size: Size of the embedding vector
            max_len: Initial maximum length for positional embeddings (will be extended if needed)
            damping_factor: Scaling factor for the positional embeddings
            dimension: Either 'row' or 'column' to specify which dimension to apply embeddings to
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.damping_factor = damping_factor
        self.dimension = dimension
        self._initial_max_len = max_len
        
        if dimension not in ['row', 'column']:
            raise ValueError(f"dimension must be 'row' or 'column', got {dimension}")
        
        # Use nn.Embedding for learnable positional embeddings
        # Start with initial max_len, but we'll extend dynamically if needed
        self.pos_embedding = nn.Embedding(max_len, embedding_size)

    def _extend_embeddings(self, new_max_len):
        """Extend the embedding layer to support more positions."""
        old_embedding = self.pos_embedding.weight
        old_max_len = old_embedding.size(0)
        
        # Create new embedding with larger capacity
        new_embedding = nn.Embedding(new_max_len, self.embedding_size, device=old_embedding.device, dtype=old_embedding.dtype)
        
        # Copy existing embeddings
        with torch.no_grad():
            new_embedding.weight[:old_max_len] = old_embedding
            # Initialize new positions with small random values
            nn.init.normal_(new_embedding.weight[old_max_len:], std=0.02)
        
        # Replace the embedding layer
        self.pos_embedding = new_embedding

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, num_rows, num_cols, embedding_size)
        """
        if self.dimension == 'row':
            _, num_positions, _, _ = x.shape
            # Shape: (1, num_rows, 1, embedding_size) for broadcasting
            unsqueeze_dims = (0, 2)
        else:  # dimension == 'column'
            _, _, num_positions, _ = x.shape
            # Shape: (1, 1, num_cols, embedding_size) for broadcasting
            unsqueeze_dims = (0, 1)
        
        # If we need more positions than current capacity, extend the embedding layer
        current_max_len = self.pos_embedding.num_embeddings
        if num_positions > current_max_len:
            self._extend_embeddings(num_positions)
        
        # Get position indices
        positions = torch.arange(num_positions, device=x.device, dtype=torch.long)
        
        # Get embeddings and reshape for broadcasting
        pos_embeddings = self.pos_embedding(positions).unsqueeze(unsqueeze_dims[0]).unsqueeze(unsqueeze_dims[1])
        
        return x + pos_embeddings * self.damping_factor

# Backward compatibility aliases
class SinusoidalRowEmbedding(SinusoidalPositionalEmbedding):
    def __init__(self, embedding_size, max_len=100, damping_factor=0.1):
        super().__init__(embedding_size, max_len, damping_factor, dimension='row')

class SinusoidalColumnEmbedding(SinusoidalPositionalEmbedding):
    def __init__(self, embedding_size, max_len=100, damping_factor=0.1):
        super().__init__(embedding_size, max_len, damping_factor, dimension='column')