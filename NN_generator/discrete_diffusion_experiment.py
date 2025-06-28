#!/usr/bin/env python3
"""
Discrete Diffusion Experiment for Staggered Adoption Patterns
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from discrete_diffusion_model import (
    generate_staggered_adoption_data,
    train_discrete_diffusion_model,
    sample_from_discrete_diffusion,
    DiscreteDiffusionScheduler,
    DiscreteDiffusionModel,
    visualize_staggered_patterns,
    analyze_staggered_patterns
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simple_discrete_test():
    """Run a fast simple test of discrete diffusion with reasonable quality."""
    
    print("=== Simple Discrete Diffusion Test ===")
    
    # Set device and seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate less sparse training data (more observed entries)
    def less_sparse_staggered_adoption_data(num_samples=40, size_range=(6, 8), random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        data_matrices = []
        masking_matrices = []
        for i in range(num_samples):
            num_units = np.random.randint(*size_range)
            num_timepoints = np.random.randint(*size_range)
            data_matrix = np.random.randn(num_units, num_timepoints)
            mask_matrix = np.ones((num_units, num_timepoints))
            for unit in range(num_units):
                if np.random.random() < 0.4:
                    start_missing = np.random.randint(num_timepoints//2, num_timepoints)
                    mask_matrix[unit, start_missing:] = 0
            mask_matrix = mask_matrix.astype(np.float32)
            data_matrices.append(data_matrix)
            masking_matrices.append(mask_matrix)
        return data_matrices, masking_matrices
    
    print("1. Generating less sparse training dataset...")
    data_matrices, masking_matrices = less_sparse_staggered_adoption_data(
        num_samples=40,
        size_range=(6, 8),
        random_seed=42
    )
    print(f"   Generated {len(data_matrices)} samples")
    print(f"   Matrix sizes: {[m.shape for m in masking_matrices[:5]]}")
    
    # Visualize a few training masks
    print("   Visualizing a few training masks...")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(masking_matrices[i], cmap='RdYlBu_r', aspect='auto')
        axes[i].set_title(f'Train Mask {i+1}')
    plt.tight_layout()
    plt.savefig('simple_discrete_train_masks.png', dpi=120, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Analyze a few patterns
    patterns = analyze_staggered_patterns(masking_matrices[:5])
    print("   Sample patterns:")
    for i, pattern in enumerate(patterns):
        print(f"     Matrix {i+1}: {pattern}")
    
    # Convert data to sequences
    sequences = []
    size_infos = []
    for mask in masking_matrices:
        rows, cols = mask.shape
        sequence = mask.flatten().astype(np.int64)
        if len(sequence) < 64:
            sequence = np.pad(sequence, (0, 64 - len(sequence)), constant_values=0)
        else:
            sequence = sequence[:64]
        sequences.append(sequence)
        size_infos.append([rows, cols])
    x_data = torch.LongTensor(np.array(sequences)).to(device)
    size_data = torch.FloatTensor(np.array(size_infos)).to(device)
    print(f"   Training data shape: {x_data.shape}")
    print(f"   Data statistics: 0s={torch.sum(x_data == 0).item()}, 1s={torch.sum(x_data == 1).item()}")
    
    # Compute class weights for loss
    num_zeros = torch.sum(x_data == 0).item()
    num_ones = torch.sum(x_data == 1).item()
    total = num_zeros + num_ones
    weight_0 = total / (2 * num_zeros) if num_zeros > 0 else 1.0
    weight_1 = total / (2 * num_ones) if num_ones > 0 else 1.0
    class_weights = torch.tensor([weight_0, weight_1, 1.0], dtype=torch.float32, device=device)
    print(f"   Class weights: {class_weights.tolist()}")
    
    # Create model
    print("\n2. Creating model...")
    model = DiscreteDiffusionModel(
        vocab_size=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_seq_len=64
    ).to(device)
    scheduler = DiscreteDiffusionScheduler(num_timesteps=20)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n3. Running training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    losses = []
    for epoch in range(15):
        total_loss = 0.0
        num_batches = 0
        batch_size = 8
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i+batch_size]
            batch_size_info = size_data[i:i+batch_size]
            if len(batch_x) == 0:
                continue
            t = torch.randint(0, scheduler.num_timesteps, (len(batch_x),), device=device)
            x_noisy, mask_indicator = scheduler.add_noise(batch_x, t)
            loss_weights = scheduler.get_loss_weight(t)
            logits = model(x_noisy, t, batch_size_info)
            targets = batch_x.clone()
            masked_logits = logits[mask_indicator]
            masked_targets = targets[mask_indicator]
            if len(masked_logits) > 0:
                loss = F.cross_entropy(masked_logits, masked_targets, weight=class_weights, reduction='none')
                masked_weights = loss_weights.repeat_interleave(mask_indicator.sum(dim=1))
                loss = (loss * masked_weights).mean()
            else:
                loss = torch.tensor(0.0, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/15: Loss = {avg_loss:.6f}")
    print(f"   Final loss: {losses[-1]:.6f}")
    
    # Test generation
    print("\n4. Testing generation...")
    model.eval()
    test_sizes = [(6, 6), (7, 6), (6, 7)]
    generated_samples = []
    for rows, cols in test_sizes:
        print(f"   Generating {rows}x{cols} sample...")
        size_info = torch.tensor([[rows, cols]], dtype=torch.float32, device=device)
        sample = sample_simple(model, scheduler, size_info, device, num_steps=10)
        generated_samples.append(sample)
        pattern = analyze_staggered_patterns([sample])[0]
        print(f"     Generated pattern: {pattern}")
        print(f"     Sample matrix (first few rows):")
        print(f"     {sample[:3, :]}")
    # Visualize results
    print("\n5. Creating simple visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        if i < len(masking_matrices):
            axes[0, i].imshow(masking_matrices[i], cmap='RdYlBu_r', aspect='auto')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Units')
    for i in range(3):
        if i < len(generated_samples):
            axes[1, i].imshow(generated_samples[i], cmap='RdYlBu_r', aspect='auto')
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Units')
    plt.tight_layout()
    plt.savefig('simple_discrete_test.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # Basic analysis
    print("\n6. Basic analysis...")
    original_patterns = analyze_staggered_patterns(masking_matrices[:3])
    generated_patterns = analyze_staggered_patterns(generated_samples)
    print("   Original patterns:")
    for i, pattern in enumerate(original_patterns):
        print(f"     {pattern}")
    print("   Generated patterns:")
    for i, pattern in enumerate(generated_patterns):
        print(f"     {pattern}")
    def check_staggered_properties(patterns):
        results = []
        for pattern in patterns:
            unique_points = len(set(pattern))
            has_variation = unique_points > 1
            early_dropout = any(p < max(pattern) * 0.5 for p in pattern)
            late_dropout = any(p > max(pattern) * 0.5 for p in pattern)
            results.append({
                'has_variation': has_variation,
                'early_dropout': early_dropout,
                'late_dropout': late_dropout,
                'unique_points': unique_points
            })
        return results
    original_props = check_staggered_properties(original_patterns)
    generated_props = check_staggered_properties(generated_patterns)
    print("   Original pattern properties:")
    for i, props in enumerate(original_props):
        print(f"     Pattern {i+1}: variation={props['has_variation']}, "
              f"early={props['early_dropout']}, late={props['late_dropout']}, "
              f"unique={props['unique_points']}")
    print("   Generated pattern properties:")
    for i, props in enumerate(generated_props):
        print(f"     Pattern {i+1}: variation={props['has_variation']}, "
              f"early={props['early_dropout']}, late={props['late_dropout']}, "
              f"unique={props['unique_points']}")
    print("\n7. Debugging analysis...")
    print("   Testing model predictions on a simple case...")
    test_x = torch.full((1, 36), 2, dtype=torch.long, device=device)
    test_t = torch.tensor([0], device=device)
    test_size = torch.tensor([[6, 6]], dtype=torch.float32, device=device)
    with torch.no_grad():
        test_logits = model(test_x, test_t, test_size)
        test_probs = F.softmax(test_logits, dim=-1)
        prob_0 = test_probs[0, :, 0].mean().item()
        prob_1 = test_probs[0, :, 1].mean().item()
        print(f"     Average P(missing) = {prob_0:.3f}")
        print(f"     Average P(observed) = {prob_1:.3f}")
        if prob_0 > 0.8:
            print("     ⚠️  Model seems to strongly prefer missing data")
        elif prob_1 > 0.8:
            print("     ⚠️  Model seems to strongly prefer observed data")
        else:
            print("     ✅ Model shows balanced predictions")
    print("\n=== Simple test completed! ===")
    print("Check 'simple_discrete_test.png' and 'simple_discrete_train_masks.png' for visual results.")
    return {
        'model': model,
        'scheduler': scheduler,
        'losses': losses,
        'original_patterns': original_patterns,
        'generated_patterns': generated_patterns,
        'original_props': original_props,
        'generated_props': generated_props
    }

def sample_simple(model, scheduler, size_info, device, num_steps=10):
    """Simple sampling function for testing."""
    
    # Get sequence length
    rows, cols = size_info[0].cpu().numpy()
    seq_len = int(rows * cols)
    
    # Start from all masked tokens
    x = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
    
    # Simple denoising
    step_size = scheduler.num_timesteps // num_steps
    
    with torch.no_grad():
        for i in range(num_steps - 1, -1, -1):
            t = torch.tensor([i * step_size], device=device)
            
            # Predict - fix size_info shape
            # size_info should be [batch, 2] not [1, 1, 2]
            if size_info.dim() == 3:
                size_info = size_info.squeeze(0)  # Remove extra dimension
            
            logits = model(x, t, size_info)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            
            # Only sample for masked positions
            masked_positions = (x[0] == 2)
            if masked_positions.any():
                masked_probs = probs[0, masked_positions]
                sampled_tokens = torch.multinomial(masked_probs, 1).squeeze(-1)
                x[0, masked_positions] = sampled_tokens
    
    # Convert back to matrix
    sequence = x[0].cpu().numpy()[:seq_len]
    mask_matrix = sequence.reshape(int(rows), int(cols))
    
    return mask_matrix

def run_discrete_diffusion_experiment():
    """Run the complete discrete diffusion experiment."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate training data
    print("Generating staggered adoption training data...")
    data_matrices, masking_matrices = generate_staggered_adoption_data(
        num_samples=200,  # More samples for better learning
        size_range=(8, 16),
        random_seed=42
    )
    
    print(f"Generated {len(data_matrices)} training samples")
    print(f"Matrix sizes range: {min([m.shape for m in masking_matrices])} to {max([m.shape for m in masking_matrices])}")
    
    # Analyze training patterns
    print("\nAnalyzing training patterns...")
    training_patterns = analyze_staggered_patterns(masking_matrices[:10])
    print("Sample dropout points (first 10 matrices):")
    for i, pattern in enumerate(training_patterns):
        print(f"Matrix {i+1}: {pattern}")
    
    # Train the discrete diffusion model
    print("\nTraining discrete diffusion model...")
    model = train_discrete_diffusion_model(
        data_matrices=data_matrices,
        masking_matrices=masking_matrices,
        num_epochs=150,  # More epochs for better learning
        batch_size=32,
        max_size=32,
        device=device,
        learning_rate=1e-4,
        save_path='discrete_diffusion_model.pth'
    )
    
    # Load the best model
    model.load_state_dict(torch.load('discrete_diffusion_model.pth', map_location=device))
    model.eval()
    
    # Initialize scheduler
    scheduler = DiscreteDiffusionScheduler(num_timesteps=100)
    
    # Generate test samples
    print("\nGenerating test samples...")
    test_sizes = [
        (10, 12), (12, 10), (8, 15), (15, 8), (11, 11),
        (9, 13), (13, 9), (14, 10), (10, 14), (12, 12)
    ]
    
    generated_masks = []
    for i, (rows, cols) in enumerate(test_sizes):
        print(f"Generating sample {i+1}/{len(test_sizes)}: {rows}x{cols}")
        size_info = torch.tensor([[rows, cols]], dtype=torch.float32, device=device)
        
        # Generate multiple samples for each size
        samples_for_size = []
        for j in range(3):  # Generate 3 samples per size
            sample = sample_from_discrete_diffusion(
                model, scheduler, size_info, num_steps=50, device=device
            )
            samples_for_size.append(sample)
        
        generated_masks.extend(samples_for_size)
    
    # Analyze generated patterns
    print("\nAnalyzing generated patterns...")
    generated_patterns = analyze_staggered_patterns(generated_masks)
    
    # Create some original patterns for comparison
    print("\nGenerating comparison original patterns...")
    _, original_masks = generate_staggered_adoption_data(
        num_samples=len(generated_masks),
        size_range=(8, 16),
        random_seed=123  # Different seed for comparison
    )
    original_patterns = analyze_staggered_patterns(original_masks)
    
    # Visualize results
    print("\nCreating visualizations...")
    
    # 1. Pattern comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original patterns heatmap
    original_dropout_matrix = np.array(original_patterns)
    im1 = axes[0, 0].imshow(original_dropout_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Original Staggered Adoption Patterns\n(Dropout Points)')
    axes[0, 0].set_xlabel('Units')
    axes[0, 0].set_ylabel('Matrix Index')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Generated patterns heatmap
    generated_dropout_matrix = np.array(generated_patterns)
    im2 = axes[0, 1].imshow(generated_dropout_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Generated Staggered Adoption Patterns\n(Dropout Points)')
    axes[0, 1].set_xlabel('Units')
    axes[0, 1].set_ylabel('Matrix Index')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Dropout point distributions
    axes[1, 0].hist([p for pattern in original_patterns for p in pattern], 
                   bins=20, alpha=0.7, label='Original', color='blue')
    axes[1, 0].hist([p for pattern in generated_patterns for p in pattern], 
                   bins=20, alpha=0.7, label='Generated', color='red')
    axes[1, 0].set_title('Distribution of Dropout Points')
    axes[1, 0].set_xlabel('Dropout Time Point')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Pattern statistics
    original_means = [np.mean(p) for p in original_patterns]
    generated_means = [np.mean(p) for p in generated_patterns]
    original_stds = [np.std(p) for p in original_patterns]
    generated_stds = [np.std(p) for p in generated_patterns]
    
    axes[1, 1].scatter(original_means, original_stds, alpha=0.7, label='Original', color='blue')
    axes[1, 1].scatter(generated_means, generated_stds, alpha=0.7, label='Generated', color='red')
    axes[1, 1].set_xlabel('Mean Dropout Point')
    axes[1, 1].set_ylabel('Std Dropout Point')
    axes[1, 1].set_title('Pattern Statistics')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('discrete_diffusion_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Individual pattern visualization
    print("\nVisualizing individual patterns...")
    num_samples_to_show = min(5, len(original_masks), len(generated_masks))
    visualize_staggered_patterns(
        original_masks[:num_samples_to_show], 
        generated_masks[:num_samples_to_show], 
        num_samples=num_samples_to_show
    )
    
    # 3. Quantitative analysis
    print("\nQuantitative Analysis:")
    
    # Calculate pattern similarity metrics
    def calculate_pattern_similarity(patterns1, patterns2):
        """Calculate similarity between two sets of patterns."""
        if len(patterns1) != len(patterns2):
            return None
        
        similarities = []
        for p1, p2 in zip(patterns1, patterns2):
            if len(p1) == len(p2):
                # Calculate correlation between dropout points
                correlation = np.corrcoef(p1, p2)[0, 1]
                if not np.isnan(correlation):
                    similarities.append(correlation)
        
        return np.mean(similarities) if similarities else 0.0
    
    # Compare with training patterns
    training_sample = analyze_staggered_patterns(masking_matrices[:len(generated_patterns)])
    training_similarity = calculate_pattern_similarity(training_sample, generated_patterns)
    
    # Compare with random patterns
    random_patterns = []
    for pattern in generated_patterns:
        random_pattern = np.random.randint(1, max(pattern) + 1, size=len(pattern))
        random_patterns.append(random_pattern)
    random_similarity = calculate_pattern_similarity(random_patterns, generated_patterns)
    
    print(f"Pattern similarity with training data: {training_similarity:.3f}")
    print(f"Pattern similarity with random data: {random_similarity:.3f}")
    
    # Calculate staggered adoption metrics
    def calculate_staggered_metrics(patterns):
        """Calculate metrics specific to staggered adoption patterns."""
        metrics = []
        for pattern in patterns:
            # Calculate the "staircase" effect
            sorted_pattern = np.sort(pattern)
            # Measure how much the pattern increases over time
            if len(sorted_pattern) > 1:
                slope = np.polyfit(range(len(sorted_pattern)), sorted_pattern, 1)[0]
                metrics.append(slope)
            else:
                metrics.append(0)
        return np.mean(metrics)
    
    original_staggered_score = calculate_staggered_metrics(original_patterns)
    generated_staggered_score = calculate_staggered_metrics(generated_patterns)
    
    print(f"Staggered adoption score (original): {original_staggered_score:.3f}")
    print(f"Staggered adoption score (generated): {generated_staggered_score:.3f}")
    
    # 4. Detailed pattern analysis
    print("\nDetailed Pattern Analysis:")
    
    # Check for realistic staggered patterns
    realistic_patterns = 0
    for pattern in generated_patterns:
        # A realistic staggered pattern should have increasing dropout points
        sorted_pattern = np.sort(pattern)
        if np.all(np.diff(sorted_pattern) >= 0):  # Monotonically increasing
            realistic_patterns += 1
    
    print(f"Realistic staggered patterns: {realistic_patterns}/{len(generated_patterns)} ({realistic_patterns/len(generated_patterns)*100:.1f}%)")
    
    # Check for early/late adoption patterns
    early_adoption = 0
    late_adoption = 0
    for pattern in generated_patterns:
        mean_dropout = np.mean(pattern)
        max_time = max(pattern)
        if mean_dropout < max_time * 0.3:  # Early adoption
            early_adoption += 1
        elif mean_dropout > max_time * 0.7:  # Late adoption
            late_adoption += 1
    
    print(f"Early adoption patterns: {early_adoption}/{len(generated_patterns)} ({early_adoption/len(generated_patterns)*100:.1f}%)")
    print(f"Late adoption patterns: {late_adoption}/{len(generated_patterns)} ({late_adoption/len(generated_patterns)*100:.1f}%)")
    
    return {
        'model': model,
        'scheduler': scheduler,
        'generated_masks': generated_masks,
        'generated_patterns': generated_patterns,
        'original_masks': original_masks,
        'original_patterns': original_patterns,
        'training_similarity': training_similarity,
        'random_similarity': random_similarity,
        'original_staggered_score': original_staggered_score,
        'generated_staggered_score': generated_staggered_score,
        'realistic_patterns': realistic_patterns,
        'early_adoption': early_adoption,
        'late_adoption': late_adoption
    }

def improved_discrete_test():
    """Run an improved discrete diffusion test with larger dataset and model."""
    import matplotlib.pyplot as plt
    
    print("=== Improved Discrete Diffusion Test ===")
    
    # Set device and seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate better training data with clearer staggered patterns
    def better_staggered_adoption_data(num_samples=200, size_range=(8, 12), random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        data_matrices = []
        masking_matrices = []
        for i in range(num_samples):
            num_units = np.random.randint(*size_range)
            num_timepoints = np.random.randint(*size_range)
            data_matrix = np.random.randn(num_units, num_timepoints)
            mask_matrix = np.ones((num_units, num_timepoints))
            
            # Create more structured staggered patterns
            pattern_type = np.random.choice(['early', 'late', 'mixed', 'gradual'])
            
            if pattern_type == 'early':
                # Most units drop out early
                for unit in range(num_units):
                    if np.random.random() < 0.7:  # 70% chance
                        start_missing = np.random.randint(1, num_timepoints // 2)
                        mask_matrix[unit, start_missing:] = 0
                        
            elif pattern_type == 'late':
                # Most units drop out late
                for unit in range(num_units):
                    if np.random.random() < 0.7:  # 70% chance
                        start_missing = np.random.randint(num_timepoints // 2, num_timepoints)
                        mask_matrix[unit, start_missing:] = 0
                        
            elif pattern_type == 'mixed':
                # Mix of early and late
                for unit in range(num_units):
                    if np.random.random() < 0.6:  # 60% chance
                        if np.random.random() < 0.5:
                            start_missing = np.random.randint(1, num_timepoints // 2)
                        else:
                            start_missing = np.random.randint(num_timepoints // 2, num_timepoints)
                        mask_matrix[unit, start_missing:] = 0
                        
            else:  # gradual
                # Gradual dropout across time
                for unit in range(num_units):
                    if np.random.random() < 0.8:  # 80% chance
                        # Create a more gradual pattern
                        dropout_prob = np.linspace(0.1, 0.9, num_timepoints)
                        for t in range(num_timepoints):
                            if np.random.random() < dropout_prob[t]:
                                mask_matrix[unit, t:] = 0
                                break
            
            mask_matrix = mask_matrix.astype(np.float32)
            data_matrices.append(data_matrix)
            masking_matrices.append(mask_matrix)
        return data_matrices, masking_matrices
    
    print("1. Generating improved training dataset...")
    data_matrices, masking_matrices = better_staggered_adoption_data(
        num_samples=200,
        size_range=(8, 12),
        random_seed=42
    )
    print(f"   Generated {len(data_matrices)} samples")
    print(f"   Matrix sizes: {[m.shape for m in masking_matrices[:5]]}")
    
    # Visualize a few training masks
    print("   Visualizing training masks...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        axes[0, i].imshow(masking_matrices[i], cmap='RdYlBu_r', aspect='auto')
        axes[0, i].set_title(f'Train Mask {i+1}')
        axes[1, i].imshow(masking_matrices[i+5], cmap='RdYlBu_r', aspect='auto')
        axes[1, i].set_title(f'Train Mask {i+6}')
    plt.tight_layout()
    plt.savefig('improved_discrete_train_masks.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    # Analyze patterns
    patterns = analyze_staggered_patterns(masking_matrices[:10])
    print("   Sample patterns:")
    for i, pattern in enumerate(patterns):
        print(f"     Matrix {i+1}: {pattern}")
    
    # Convert data to sequences
    sequences = []
    size_infos = []
    for mask in masking_matrices:
        rows, cols = mask.shape
        sequence = mask.flatten().astype(np.int64)
        if len(sequence) < 144:  # 12x12 max
            sequence = np.pad(sequence, (0, 144 - len(sequence)), constant_values=0)
        else:
            sequence = sequence[:144]
        sequences.append(sequence)
        size_infos.append([rows, cols])
    x_data = torch.LongTensor(np.array(sequences)).to(device)
    size_data = torch.FloatTensor(np.array(size_infos)).to(device)
    print(f"   Training data shape: {x_data.shape}")
    print(f"   Data statistics: 0s={torch.sum(x_data == 0).item()}, 1s={torch.sum(x_data == 1).item()}")
    
    # Compute class weights
    num_zeros = torch.sum(x_data == 0).item()
    num_ones = torch.sum(x_data == 1).item()
    total = num_zeros + num_ones
    weight_0 = total / (2 * num_zeros) if num_zeros > 0 else 1.0
    weight_1 = total / (2 * num_ones) if num_ones > 0 else 1.0
    class_weights = torch.tensor([weight_0, weight_1, 1.0], dtype=torch.float32, device=device)
    
    # Debug: Check data distribution
    print(f"   Training data distribution: {num_zeros} zeros, {num_ones} ones")
    print(f"   Class weights: {class_weights.tolist()}")
    print(f"   Zero/One ratio: {num_zeros/num_ones:.3f}")
    
    # Check if the issue is in the data generation
    print("   Checking training data quality...")
    sample_masks = masking_matrices[:5]
    for i, mask in enumerate(sample_masks):
        zeros = np.sum(mask == 0)
        ones = np.sum(mask == 1)
        print(f"     Sample {i+1}: {zeros} zeros, {ones} ones, ratio: {zeros/ones:.3f}")
    
    # Create and train model
    model = DiscreteDiffusionModel(
        vocab_size=3,
        d_model=128,
        nhead=8,
        num_layers=4,
        max_seq_len=144
    ).to(device)
    scheduler = DiscreteDiffusionScheduler(num_timesteps=50)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Debug: Check model initialization logits
    print("   Checking model initialization logits...")
    with torch.no_grad():
        test_input = torch.full((1, 36), 2, dtype=torch.long, device=device)
        test_t = torch.tensor([0], device=device)
        test_size = torch.tensor([[6, 6]], dtype=torch.float32, device=device)
        test_logits = model(test_input, test_t, test_size)
        logits_0 = test_logits[0, :, 0].cpu().numpy()
        logits_1 = test_logits[0, :, 1].cpu().numpy()
        print(f"     Logits class 0: mean={logits_0.mean():.3f}, min={logits_0.min():.3f}, max={logits_0.max():.3f}")
        print(f"     Logits class 1: mean={logits_1.mean():.3f}, min={logits_1.min():.3f}, max={logits_1.max():.3f}")
        test_probs = F.softmax(test_logits, dim=-1)
        init_prob_0 = test_probs[0, :, 0].mean().item()
        init_prob_1 = test_probs[0, :, 1].mean().item()
        print(f"     Initial predictions: P(0)={init_prob_0:.3f}, P(1)={init_prob_1:.3f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    model.train()
    losses = []
    
    for epoch in range(50):
        total_loss = 0.0
        num_batches = 0
        batch_size = 16
        
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i+batch_size]
            batch_size_info = size_data[i:i+batch_size]
            if len(batch_x) == 0:
                continue
                
            t = torch.randint(0, scheduler.num_timesteps, (len(batch_x),), device=device)
            x_noisy, mask_indicator = scheduler.add_noise(batch_x, t)
            loss_weights = scheduler.get_loss_weight(t)
            logits = model(x_noisy, t, batch_size_info)
            targets = batch_x.clone()
            masked_logits = logits[mask_indicator]
            masked_targets = targets[mask_indicator]
            
            if len(masked_logits) > 0:
                loss = F.cross_entropy(masked_logits, masked_targets, weight=class_weights, reduction='none')
                masked_weights = loss_weights.repeat_interleave(mask_indicator.sum(dim=1))
                loss = (loss * masked_weights).mean()
            else:
                loss = torch.tensor(0.0, device=device)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        lr_scheduler.step()
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/50: Loss = {avg_loss:.6f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}")
    
    print(f"   Final loss: {losses[-1]:.6f}")
    
    # Test generation
    print("\n4. Testing generation...")
    model.eval()
    test_sizes = [(8, 10), (10, 8), (9, 9), (11, 8), (8, 11)]
    generated_samples = []
    
    for rows, cols in test_sizes:
        print(f"   Generating {rows}x{cols} sample...")
        size_info = torch.tensor([[rows, cols]], dtype=torch.float32, device=device)
        sample = sample_simple(model, scheduler, size_info, device, num_steps=25)
        generated_samples.append(sample)
        pattern = analyze_staggered_patterns([sample])[0]
        print(f"     Generated pattern: {pattern}")
        print(f"     Sample matrix (first few rows):")
        print(f"     {sample[:3, :]}")
    
    # Visualize results
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(5):
        if i < len(masking_matrices):
            axes[0, i].imshow(masking_matrices[i], cmap='RdYlBu_r', aspect='auto')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Units')
    for i in range(5):
        if i < len(generated_samples):
            axes[1, i].imshow(generated_samples[i], cmap='RdYlBu_r', aspect='auto')
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Units')
    plt.tight_layout()
    plt.savefig('improved_discrete_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analysis
    print("\n6. Analysis...")
    original_patterns = analyze_staggered_patterns(masking_matrices[:5])
    generated_patterns = analyze_staggered_patterns(generated_samples)
    
    print("   Original patterns:")
    for i, pattern in enumerate(original_patterns):
        print(f"     {pattern}")
    print("   Generated patterns:")
    for i, pattern in enumerate(generated_patterns):
        print(f"     {pattern}")
    
    def check_staggered_properties(patterns):
        results = []
        for pattern in patterns:
            unique_points = len(set(pattern))
            has_variation = unique_points > 1
            early_dropout = any(p < max(pattern) * 0.5 for p in pattern)
            late_dropout = any(p > max(pattern) * 0.5 for p in pattern)
            results.append({
                'has_variation': has_variation,
                'early_dropout': early_dropout,
                'late_dropout': late_dropout,
                'unique_points': unique_points
            })
        return results
    
    original_props = check_staggered_properties(original_patterns)
    generated_props = check_staggered_properties(generated_patterns)
    
    print("   Original pattern properties:")
    for i, props in enumerate(original_props):
        print(f"     Pattern {i+1}: variation={props['has_variation']}, "
              f"early={props['early_dropout']}, late={props['late_dropout']}, "
              f"unique={props['unique_points']}")
    print("   Generated pattern properties:")
    for i, props in enumerate(generated_props):
        print(f"     Pattern {i+1}: variation={props['has_variation']}, "
              f"early={props['early_dropout']}, late={props['late_dropout']}, "
              f"unique={props['unique_points']}")
    
    # Debugging
    print("\n7. Debugging analysis...")
    print("   Testing model predictions...")
    test_x = torch.full((1, 100), 2, dtype=torch.long, device=device)
    test_t = torch.tensor([0], device=device)
    test_size = torch.tensor([[10, 10]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        test_logits = model(test_x, test_t, test_size)
        test_probs = F.softmax(test_logits, dim=-1)
        prob_0 = test_probs[0, :, 0].mean().item()
        prob_1 = test_probs[0, :, 1].mean().item()
        print(f"     Average P(missing) = {prob_0:.3f}")
        print(f"     Average P(observed) = {prob_1:.3f}")
        if prob_0 > 0.8:
            print("     ⚠️  Model strongly prefers missing data")
        elif prob_1 > 0.8:
            print("     ⚠️  Model strongly prefers observed data")
        else:
            print("     ✅ Model shows balanced predictions")
    
    print("\n=== Improved test completed! ===")
    print("Check 'improved_discrete_test.png' and 'improved_discrete_train_masks.png' for results.")
    
    return {
        'model': model,
        'scheduler': scheduler,
        'losses': losses,
        'original_patterns': original_patterns,
        'generated_patterns': generated_patterns,
        'original_props': original_props,
        'generated_props': generated_props
    }

def investigate_sampling():
    """Investigate the sampling process to understand why we get all zeros."""
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    
    print("=== Sampling Process Investigation ===")
    
    # Set device and seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    # First, let's run a quick training to get a model
    print("1. Training a model for investigation...")
    
    # Generate some training data
    def quick_training_data(num_samples=50, size_range=(6, 8), random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        data_matrices = []
        masking_matrices = []
        for i in range(num_samples):
            num_units = np.random.randint(*size_range)
            num_timepoints = np.random.randint(*size_range)
            data_matrix = np.random.randn(num_units, num_timepoints)
            mask_matrix = np.ones((num_units, num_timepoints))
            # Simple staggered pattern
            for unit in range(num_units):
                if np.random.random() < 0.6:
                    start_missing = np.random.randint(1, num_timepoints)
                    mask_matrix[unit, start_missing:] = 0
            mask_matrix = mask_matrix.astype(np.float32)
            data_matrices.append(data_matrix)
            masking_matrices.append(mask_matrix)
        return data_matrices, masking_matrices
    
    data_matrices, masking_matrices = quick_training_data(50, (6, 8), 42)
    
    # Convert to sequences
    sequences = []
    size_infos = []
    for mask in masking_matrices:
        rows, cols = mask.shape
        sequence = mask.flatten().astype(np.int64)
        if len(sequence) < 64:
            sequence = np.pad(sequence, (0, 64 - len(sequence)), constant_values=0)
        else:
            sequence = sequence[:64]
        sequences.append(sequence)
        size_infos.append([rows, cols])
    
    x_data = torch.LongTensor(np.array(sequences)).to(device)
    size_data = torch.FloatTensor(np.array(size_infos)).to(device)
    
    # Compute class weights
    num_zeros = torch.sum(x_data == 0).item()
    num_ones = torch.sum(x_data == 1).item()
    total = num_zeros + num_ones
    weight_0 = total / (2 * num_zeros) if num_zeros > 0 else 1.0
    weight_1 = total / (2 * num_ones) if num_ones > 0 else 1.0
    class_weights = torch.tensor([weight_0, weight_1, 1.0], dtype=torch.float32, device=device)
    
    # Debug: Check data distribution
    print(f"   Training data distribution: {num_zeros} zeros, {num_ones} ones")
    print(f"   Class weights: {class_weights.tolist()}")
    print(f"   Zero/One ratio: {num_zeros/num_ones:.3f}")
    
    # Check if the issue is in the data generation
    print("   Checking training data quality...")
    sample_masks = masking_matrices[:5]
    for i, mask in enumerate(sample_masks):
        zeros = np.sum(mask == 0)
        ones = np.sum(mask == 1)
        print(f"     Sample {i+1}: {zeros} zeros, {ones} ones, ratio: {zeros/ones:.3f}")
    
    # Create and train model
    model = DiscreteDiffusionModel(
        vocab_size=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_seq_len=64
    ).to(device)
    scheduler = DiscreteDiffusionScheduler(num_timesteps=20)
    
    # Debug: Check model initialization logits
    print("   Checking model initialization logits...")
    with torch.no_grad():
        test_input = torch.full((1, 36), 2, dtype=torch.long, device=device)
        test_t = torch.tensor([0], device=device)
        test_size = torch.tensor([[6, 6]], dtype=torch.float32, device=device)
        test_logits = model(test_input, test_t, test_size)
        logits_0 = test_logits[0, :, 0].cpu().numpy()
        logits_1 = test_logits[0, :, 1].cpu().numpy()
        print(f"     Logits class 0: mean={logits_0.mean():.3f}, min={logits_0.min():.3f}, max={logits_0.max():.3f}")
        print(f"     Logits class 1: mean={logits_1.mean():.3f}, min={logits_1.min():.3f}, max={logits_1.max():.3f}")
        test_probs = F.softmax(test_logits, dim=-1)
        init_prob_0 = test_probs[0, :, 0].mean().item()
        init_prob_1 = test_probs[0, :, 1].mean().item()
        print(f"     Initial predictions: P(0)={init_prob_0:.3f}, P(1)={init_prob_1:.3f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    # Quick training
    for epoch in range(10):
        total_loss = 0.0
        num_batches = 0
        batch_size = 8
        total_correct_0 = 0
        total_correct_1 = 0
        total_0 = 0
        total_1 = 0
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i+batch_size]
            batch_size_info = size_data[i:i+batch_size]
            if len(batch_x) == 0:
                continue
            t = torch.randint(0, scheduler.num_timesteps, (len(batch_x),), device=device)
            x_noisy, mask_indicator = scheduler.add_noise(batch_x, t)
            loss_weights = scheduler.get_loss_weight(t)
            logits = model(x_noisy, t, batch_size_info)
            targets = batch_x.clone()
            masked_logits = logits[mask_indicator]
            masked_targets = targets[mask_indicator]
            if len(masked_logits) > 0:
                loss = F.cross_entropy(masked_logits, masked_targets, weight=class_weights, reduction='none')
                masked_weights = loss_weights.repeat_interleave(mask_indicator.sum(dim=1))
                loss = (loss * masked_weights).mean()
                preds = masked_logits.argmax(dim=-1)
                total_correct_0 += ((preds == 0) & (masked_targets == 0)).sum().item()
                total_correct_1 += ((preds == 1) & (masked_targets == 1)).sum().item()
                total_0 += (masked_targets == 0).sum().item()
                total_1 += (masked_targets == 1).sum().item()
            else:
                loss = torch.tensor(0.0, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        acc_0 = total_correct_0 / total_0 if total_0 > 0 else 0.0
        acc_1 = total_correct_1 / total_1 if total_1 > 0 else 0.0
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/10: Loss = {total_loss/num_batches:.6f}, Acc0 = {acc_0:.3f}, Acc1 = {acc_1:.3f}")
    print("   Training completed!")
    print(f"   Final per-class accuracy: Acc0 = {acc_0:.3f}, Acc1 = {acc_1:.3f}")
    # Debug: Check model after training logits
    print("   Checking model after training logits...")
    model.eval()
    with torch.no_grad():
        test_input = torch.full((1, 36), 2, dtype=torch.long, device=device)
        test_t = torch.tensor([0], device=device)
        test_size = torch.tensor([[6, 6]], dtype=torch.float32, device=device)
        test_logits = model(test_input, test_t, test_size)
        logits_0 = test_logits[0, :, 0].cpu().numpy()
        logits_1 = test_logits[0, :, 1].cpu().numpy()
        print(f"     Logits class 0: mean={logits_0.mean():.3f}, min={logits_0.min():.3f}, max={logits_0.max():.3f}")
        print(f"     Logits class 1: mean={logits_1.mean():.3f}, min={logits_1.min():.3f}, max={logits_1.max():.3f}")
        test_probs = F.softmax(test_logits, dim=-1)
        final_prob_0 = test_probs[0, :, 0].mean().item()
        final_prob_1 = test_probs[0, :, 1].mean().item()
        print(f"     Final predictions: P(0)={final_prob_0:.3f}, P(1)={final_prob_1:.3f}")
        print(f"     Change from init: P(0)={final_prob_0-init_prob_0:.3f}, P(1)={final_prob_1-init_prob_1:.3f}")
    model.eval()

    # Now investigate the sampling process
    print("\n2. Investigating sampling process...")
    model.eval()
    
    # Test size
    test_size = torch.tensor([[6, 6]], dtype=torch.float32, device=device)
    seq_len = 36  # 6x6
    
    print("   Analyzing model predictions at different timesteps...")
    
    # Start from all masked tokens
    x = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
    
    # Track predictions at each step
    step_predictions = []
    step_probabilities = []
    
    with torch.no_grad():
        for i in range(19, -1, -1):  # From t=19 to t=0
            t = torch.tensor([i], device=device)
            
            # Get model predictions
            logits = model(x, t, test_size)
            probs = F.softmax(logits, dim=-1)
            
            # Record predictions for masked positions
            masked_positions = (x[0] == 2)
            if masked_positions.any():
                masked_probs = probs[0, masked_positions]
                step_predictions.append(masked_probs.argmax(dim=-1).cpu().numpy())
                step_probabilities.append(masked_probs.cpu().numpy())
                
                # Sample new tokens
                sampled_tokens = torch.multinomial(masked_probs, 1).squeeze(-1)
                x[0, masked_positions] = sampled_tokens
            else:
                step_predictions.append([])
                step_probabilities.append([])
    
    # Analyze the sampling process
    print("   Sampling analysis:")
    print(f"   Number of sampling steps: {len(step_predictions)}")
    
    # Count predictions at each step
    for i, preds in enumerate(step_predictions):
        if len(preds) > 0:
            zeros = np.sum(preds == 0)
            ones = np.sum(preds == 1)
            print(f"   Step {19-i}: {zeros} zeros, {ones} ones (total: {len(preds)})")
    
    # Analyze final result
    final_sequence = x[0].cpu().numpy()[:seq_len]
    final_matrix = final_sequence.reshape(6, 6)
    print(f"   Final matrix:\n{final_matrix}")
    
    # Check what the model learned about different timesteps
    print("\n3. Analyzing model behavior at different timesteps...")
    
    # Test with different timesteps
    test_x = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
    timestep_analysis = []
    
    with torch.no_grad():
        for t_val in [0, 5, 10, 15, 19]:
            t = torch.tensor([t_val], device=device)
            logits = model(test_x, t, test_size)
            probs = F.softmax(logits, dim=-1)
            
            # Get average probabilities
            avg_prob_0 = probs[0, :, 0].mean().item()
            avg_prob_1 = probs[0, :, 1].mean().item()
            
            timestep_analysis.append({
                'timestep': t_val,
                'prob_0': avg_prob_0,
                'prob_1': avg_prob_1
            })
            
            print(f"   Timestep {t_val}: P(0)={avg_prob_0:.3f}, P(1)={avg_prob_1:.3f}")
    
    # Test with partially unmasked input
    print("\n4. Testing with partially unmasked input...")
    
    # Create input with some observed tokens
    partial_x = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
    partial_x[0, :10] = 1  # First 10 tokens are observed
    
    with torch.no_grad():
        t = torch.tensor([0], device=device)
        logits = model(partial_x, t, test_size)
        probs = F.softmax(logits, dim=-1)
        
        # Check predictions for masked positions
        masked_positions = (partial_x[0] == 2)
        if masked_positions.any():
            masked_probs = probs[0, masked_positions]
            avg_prob_0 = masked_probs[:, 0].mean().item()
            avg_prob_1 = masked_probs[:, 1].mean().item()
            print(f"   With partial unmasking: P(0)={avg_prob_0:.3f}, P(1)={avg_prob_1:.3f}")
    
    # Test different sampling strategies
    print("\n5. Testing different sampling strategies...")
    
    def test_sampling_strategy(strategy_name, temperature=1.0):
        x = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(19, -1, -1):
                t = torch.tensor([i], device=device)
                logits = model(x, t, test_size)
                masked_positions = (x[0] == 2)
                if not masked_positions.any():
                    continue
                if strategy_name == "greedy":
                    probs = F.softmax(logits, dim=-1)
                    masked_probs = probs[0, masked_positions]
                    sampled_tokens = torch.multinomial(masked_probs, 1).squeeze(-1)
                    x[0, masked_positions] = sampled_tokens
                elif strategy_name == "temperature":
                    probs = F.softmax(logits / temperature, dim=-1)
                    masked_probs = probs[0, masked_positions]
                    sampled_tokens = torch.multinomial(masked_probs, 1).squeeze(-1)
                    x[0, masked_positions] = sampled_tokens
                elif strategy_name == "top_k":
                    probs = F.softmax(logits, dim=-1)
                    k = 2
                    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                    masked_top_k_probs = top_k_probs[0, masked_positions]
                    masked_top_k_indices = top_k_indices[0, masked_positions]
                    sampled_indices = torch.multinomial(masked_top_k_probs, 1)
                    sampled_tokens = masked_top_k_indices.gather(1, sampled_indices).squeeze(-1)
                    x[0, masked_positions] = sampled_tokens
        final_sequence = x[0].cpu().numpy()[:seq_len]
        final_matrix = final_sequence.reshape(6, 6)
        zeros = np.sum(final_sequence == 0)
        ones = np.sum(final_sequence == 1)
        print(f"   {strategy_name}: {zeros} zeros, {ones} ones")
        return final_matrix
    
    # Test different strategies
    strategies = [
        ("greedy", 1.0),
        ("temperature_0.5", 0.5),
        ("temperature_2.0", 2.0),
        ("top_k", 1.0)
    ]
    
    strategy_results = {}
    for strategy_name, temp in strategies:
        if strategy_name == "top_k":
            result = test_sampling_strategy("top_k")
        else:
            result = test_sampling_strategy(strategy_name, temp)
        strategy_results[strategy_name] = result
    
    # Visualize results
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original training data
    axes[0, 0].imshow(masking_matrices[0], cmap='RdYlBu_r', aspect='auto')
    axes[0, 0].set_title('Training Example')
    
    # Final result
    axes[0, 1].imshow(strategy_results['greedy'], cmap='RdYlBu_r', aspect='auto')
    axes[0, 1].set_title('Greedy Sampling')
    
    # Temperature sampling
    axes[0, 2].imshow(strategy_results['temperature_0.5'], cmap='RdYlBu_r', aspect='auto')
    axes[0, 2].set_title('Temperature 0.5')
    
    # Top-k sampling
    axes[1, 0].imshow(strategy_results['top_k'], cmap='RdYlBu_r', aspect='auto')
    axes[1, 0].set_title('Top-K Sampling')
    
    # Temperature 2.0
    axes[1, 1].imshow(strategy_results['temperature_2.0'], cmap='RdYlBu_r', aspect='auto')
    axes[1, 1].set_title('Temperature 2.0')
    
    # Timestep analysis
    timesteps = [a['timestep'] for a in timestep_analysis]
    prob_0s = [a['prob_0'] for a in timestep_analysis]
    prob_1s = [a['prob_1'] for a in timestep_analysis]
    
    axes[1, 2].plot(timesteps, prob_0s, 'r-', label='P(missing)')
    axes[1, 2].plot(timesteps, prob_1s, 'b-', label='P(observed)')
    axes[1, 2].set_xlabel('Timestep')
    axes[1, 2].set_ylabel('Probability')
    axes[1, 2].set_title('Model Predictions vs Timestep')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('sampling_investigation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n7. Summary of findings:")
    print("   - Model predictions at different timesteps:")
    for analysis in timestep_analysis:
        print(f"     t={analysis['timestep']}: P(0)={analysis['prob_0']:.3f}, P(1)={analysis['prob_1']:.3f}")
    
    print("   - Sampling strategies tested:")
    for strategy_name in strategy_results:
        matrix = strategy_results[strategy_name]
        zeros = np.sum(matrix == 0)
        ones = np.sum(matrix == 1)
        print(f"     {strategy_name}: {zeros} zeros, {ones} ones")
    
    print("\n=== Sampling investigation completed! ===")
    print("Check 'sampling_investigation.png' for visual results.")
    
    return {
        'model': model,
        'scheduler': scheduler,
        'timestep_analysis': timestep_analysis,
        'strategy_results': strategy_results,
        'step_predictions': step_predictions
    }

class SimpleLinearModel(nn.Module):
    def __init__(self, max_seq_len=64, num_classes=3):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        # Use a larger input dimension to handle variable sequence lengths
        self.linear = nn.Linear(max_seq_len * num_classes, num_classes)
    def forward(self, x, t=None, size_info=None):
        # x: [batch, seq_len] (LongTensor)
        # One-hot encode input
        x_onehot = F.one_hot(x, num_classes=self.num_classes).float()
        # Pad or truncate to max_seq_len
        if x_onehot.shape[1] < self.max_seq_len:
            padding = torch.zeros(x_onehot.shape[0], self.max_seq_len - x_onehot.shape[1], self.num_classes, device=x_onehot.device)
            x_onehot = torch.cat([x_onehot, padding], dim=1)
        else:
            x_onehot = x_onehot[:, :self.max_seq_len, :]
        x_flat = x_onehot.view(x_onehot.shape[0], -1)
        out = self.linear(x_flat)
        # Output shape: [batch, num_classes]
        # For compatibility, expand to [batch, seq_len, num_classes]
        out = out.unsqueeze(1).expand(-1, x.shape[1], -1)
        return out

# --- EXPERIMENT: Perfectly balanced training data ---
def perfectly_balanced_training_data(num_samples=50, size=(6, 6), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    data_matrices = []
    masking_matrices = []
    for i in range(num_samples):
        num_units, num_timepoints = size
        data_matrix = np.random.randn(num_units, num_timepoints)
        mask_matrix = np.zeros((num_units, num_timepoints))
        # Fill half with ones, half with zeros
        total = num_units * num_timepoints
        ones_indices = np.random.choice(total, total // 2, replace=False)
        mask_matrix.flat[ones_indices] = 1
        data_matrices.append(data_matrix)
        masking_matrices.append(mask_matrix)
    return data_matrices, masking_matrices

print("\n--- EXPERIMENT: Perfectly balanced training data ---")
data_matrices, masking_matrices = perfectly_balanced_training_data(50, (6, 6), 42)
sequences = []
size_infos = []
for mask in masking_matrices:
    rows, cols = mask.shape
    sequence = mask.flatten().astype(np.int64)
    if len(sequence) < 64:
        sequence = np.pad(sequence, (0, 64 - len(sequence)), constant_values=0)
    else:
        sequence = sequence[:64]
    sequences.append(sequence)
    size_infos.append([rows, cols])
x_data = torch.LongTensor(np.array(sequences)).to(device)
size_data = torch.FloatTensor(np.array(size_infos)).to(device)
num_zeros = torch.sum(x_data == 0).item()
num_ones = torch.sum(x_data == 1).item()
total = num_zeros + num_ones
weight_0 = total / (2 * num_zeros) if num_zeros > 0 else 1.0
weight_1 = total / (2 * num_ones) if num_ones > 0 else 1.0
class_weights = torch.tensor([weight_0, weight_1, 1.0], dtype=torch.float32, device=device)
print(f"   Balanced data: {num_zeros} zeros, {num_ones} ones")
print(f"   Class weights: {class_weights.tolist()}")

# Now both experiments can use x_data, size_data, class_weights, etc.

if __name__ == "__main__":
    # Run the sampling investigation
    print("Running sampling investigation...")
    investigation_results = investigate_sampling()
    
    print("\n" + "="*50)
    print("Sampling investigation completed!")
    print("This investigation will help us understand:")
    print("- Why the model generates all zeros")
    print("- How the model behaves at different timesteps")
    print("- What different sampling strategies produce")
    print("- Whether the issue is in training or sampling")
    print("="*50)
    
    # --- 1. SimpleLinearModel in diffusion setup ---
    print("\n--- EXPERIMENT: SimpleLinearModel in diffusion setup ---")
    model = SimpleLinearModel(max_seq_len=64, num_classes=3).to(device)
    scheduler = DiscreteDiffusionScheduler(num_timesteps=20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(10):
        total_loss = 0.0
        num_batches = 0
        batch_size = 8
        total_correct_0 = 0
        total_correct_1 = 0
        total_0 = 0
        total_1 = 0
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i+batch_size]
            batch_size_info = size_data[i:i+batch_size]
            if len(batch_x) == 0:
                continue
            t = torch.randint(0, scheduler.num_timesteps, (len(batch_x),), device=device)
            x_noisy, mask_indicator = scheduler.add_noise(batch_x, t)
            loss_weights = scheduler.get_loss_weight(t)
            logits = model(x_noisy, t, batch_size_info)
            targets = batch_x.clone()
            masked_logits = logits[mask_indicator]
            masked_targets = targets[mask_indicator]
            if len(masked_logits) > 0:
                loss = F.cross_entropy(masked_logits, masked_targets, weight=class_weights, reduction='none')
                masked_weights = loss_weights.repeat_interleave(mask_indicator.sum(dim=1))
                loss = (loss * masked_weights).mean()
                preds = masked_logits.argmax(dim=-1)
                total_correct_0 += ((preds == 0) & (masked_targets == 0)).sum().item()
                total_correct_1 += ((preds == 1) & (masked_targets == 1)).sum().item()
                total_0 += (masked_targets == 0).sum().item()
                total_1 += (masked_targets == 1).sum().item()
            else:
                loss = torch.tensor(0.0, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        acc_0 = total_correct_0 / total_0 if total_0 > 0 else 0.0
        acc_1 = total_correct_1 / total_1 if total_1 > 0 else 0.0
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/10: Loss = {total_loss/num_batches:.6f}, Acc0 = {acc_0:.3f}, Acc1 = {acc_1:.3f}")
    print("   Training completed!")
    print(f"   Final per-class accuracy: Acc0 = {acc_0:.3f}, Acc1 = {acc_1:.3f}")
    # Logits after training
    with torch.no_grad():
        test_input = torch.full((1, 36), 2, dtype=torch.long, device=device)
        test_logits = model(test_input)
        logits_0 = test_logits[0, :, 0].cpu().numpy()
        logits_1 = test_logits[0, :, 1].cpu().numpy()
        print(f"     Logits class 0: mean={logits_0.mean():.3f}, min={logits_0.min():.3f}, max={logits_0.max():.3f}")
        print(f"     Logits class 1: mean={logits_1.mean():.3f}, min={logits_1.min():.3f}, max={logits_1.max():.3f}")
    
    # --- 2. SimpleLinearModel in standard classification ---
    print("\n--- EXPERIMENT: SimpleLinearModel in standard classification ---")
    # Generate random one-hot input and mask targets
    data = torch.randint(0, 3, (len(x_data), 64), device=device)
    targets = x_data.clone()
    model = SimpleLinearModel(max_seq_len=64, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(10):
        total_loss = 0.0
        total_correct_0 = 0
        total_correct_1 = 0
        total_0 = 0
        total_1 = 0
        for i in range(0, len(data), batch_size):
            batch_x = data[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            logits = model(batch_x)
            logits = logits[:, :batch_x.shape[1], :]
            masked_logits = logits.reshape(-1, 3)
            masked_targets = batch_targets.reshape(-1)
            loss = F.cross_entropy(masked_logits, masked_targets, weight=class_weights)
            preds = masked_logits.argmax(dim=-1)
            total_correct_0 += ((preds == 0) & (masked_targets == 0)).sum().item()
            total_correct_1 += ((preds == 1) & (masked_targets == 1)).sum().item()
            total_0 += (masked_targets == 0).sum().item()
            total_1 += (masked_targets == 1).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc_0 = total_correct_0 / total_0 if total_0 > 0 else 0.0
        acc_1 = total_correct_1 / total_1 if total_1 > 0 else 0.0
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/10: Loss = {total_loss/len(data):.6f}, Acc0 = {acc_0:.3f}, Acc1 = {acc_1:.3f}")
    print("   Training completed!")
    print(f"   Final per-class accuracy: Acc0 = {acc_0:.3f}, Acc1 = {acc_1:.3f}")
    with torch.no_grad():
        test_input = torch.randint(0, 3, (1, 64), device=device)
        test_logits = model(test_input)
        logits_0 = test_logits[0, :, 0].cpu().numpy()
        logits_1 = test_logits[0, :, 1].cpu().numpy()
        print(f"     Logits class 0: mean={logits_0.mean():.3f}, min={logits_0.min():.3f}, max={logits_0.max():.3f}")
        print(f"     Logits class 1: mean={logits_1.mean():.3f}, min={logits_1.min():.3f}, max={logits_1.max():.3f}") 