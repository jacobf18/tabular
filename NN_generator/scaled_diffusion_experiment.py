#!/usr/bin/env python3
"""
Scaled-Up Diffusion Experiment for Staggered Adoption Patterns
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from diffusion_model import (
    generate_staggered_adoption_data,
    train_minimal_diffusion_model,
    sample_from_minimal_diffusion,
    MinimalDiffusionScheduler,
    MinimalDiffusionModel,
    visualize_staggered_patterns,
    analyze_staggered_patterns
)

def run_scaled_diffusion_experiment():
    """Run the scaled-up continuous diffusion experiment."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate MORE training data with staggered adoption patterns
    print("Generating scaled-up staggered adoption training data...")
    data_matrices, masking_matrices = generate_staggered_adoption_data(
        num_samples=2000,  # 10x more samples
        size_range=(8, 20),  # Larger size range
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
    
    # Train the scaled-up diffusion model
    print("\nTraining scaled-up diffusion model...")
    model_path = 'scaled_diffusion_model.pth'
    if os.path.exists(model_path):
        print(f"Model file {model_path} found. Loading trained model...")
        model = MinimalDiffusionModel(max_size=32).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        model = train_minimal_diffusion_model(
            data_matrices=data_matrices,
            masking_matrices=masking_matrices,
            num_epochs=300,  # 3x more epochs
            batch_size=32,  # 2x larger batch size
            max_size=32,
            device=device,
            learning_rate=3e-4,  # Slightly lower learning rate for stability
            save_path=model_path
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    
    # Initialize scheduler with more timesteps
    scheduler = MinimalDiffusionScheduler(num_timesteps=100)  # 5x more timesteps
    
    # Generate test samples with more sampling steps
    print("\nGenerating test samples...")
    test_sizes = [
        (10, 12), (12, 10), (8, 15), (15, 8), (11, 11),
        (9, 13), (13, 9), (14, 10), (10, 14), (12, 12),
        (16, 18), (18, 16), (20, 12), (12, 20), (15, 15)  # More diverse sizes
    ]
    
    generated_masks = []
    for i, (rows, cols) in enumerate(test_sizes):
        print(f"Generating sample {i+1}/{len(test_sizes)}: {rows}x{cols}")
        size_info = torch.tensor([[rows, cols]], dtype=torch.float32, device=device)
        
        # Generate multiple samples for each size
        samples_for_size = []
        for j in range(5):  # Generate 5 samples per size (more diversity)
            sample = sample_from_minimal_diffusion(
                model, scheduler, size_info, num_steps=50, device=device, threshold=0.5  # More sampling steps
            )
            # Ensure mask is binary (0 or 1)
            mask = sample[0, 1, :rows, :cols].cpu().numpy()
            mask = (mask > 0.5).astype(np.float32)
            samples_for_size.append(mask)
        
        generated_masks.extend(samples_for_size)
    
    # Analyze generated patterns
    print("\nAnalyzing generated patterns...")
    generated_patterns = analyze_staggered_patterns(generated_masks)
    
    # Create some original patterns for comparison
    print("\nGenerating comparison original patterns...")
    _, original_masks = generate_staggered_adoption_data(
        num_samples=len(generated_masks),
        size_range=(8, 20),
        random_seed=123  # Different seed for comparison
    )
    original_patterns = analyze_staggered_patterns(original_masks)
    
    # Visualize results
    print("\nCreating visualizations...")
    
    # 1. Pattern comparison - handle different pattern lengths
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original patterns heatmap - pad to uniform size
    max_units = max(len(p) for p in original_patterns)
    original_dropout_matrix = np.zeros((len(original_patterns), max_units))
    for i, pattern in enumerate(original_patterns):
        original_dropout_matrix[i, :len(pattern)] = pattern
    
    im1 = axes[0, 0].imshow(original_dropout_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Original Staggered Adoption Patterns\n(Dropout Points)')
    axes[0, 0].set_xlabel('Units')
    axes[0, 0].set_ylabel('Matrix Index')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Generated patterns heatmap - pad to uniform size
    max_units_gen = max(len(p) for p in generated_patterns)
    generated_dropout_matrix = np.zeros((len(generated_patterns), max_units_gen))
    for i, pattern in enumerate(generated_patterns):
        generated_dropout_matrix[i, :len(pattern)] = pattern
    
    im2 = axes[0, 1].imshow(generated_dropout_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Generated Staggered Adoption Patterns\n(Dropout Points)')
    axes[0, 1].set_xlabel('Units')
    axes[0, 1].set_ylabel('Matrix Index')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Dropout point distributions
    all_original_points = [p for pattern in original_patterns for p in pattern]
    all_generated_points = [p for pattern in generated_patterns for p in pattern]
    
    axes[1, 0].hist(all_original_points, bins=20, alpha=0.7, label='Original', color='blue')
    axes[1, 0].hist(all_generated_points, bins=20, alpha=0.7, label='Generated', color='red')
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
    plt.savefig('scaled_diffusion_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Individual pattern visualization
    print("\nVisualizing individual patterns...")
    num_samples_to_show = min(5, len(original_masks), len(generated_masks))
    def safe_get_mask(mask):
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 4:
                mask = mask[0, 1, :, :].cpu().numpy()
            elif mask.dim() == 3:
                mask = mask[1, :, :].cpu().numpy()
            elif mask.dim() == 2:
                mask = mask.cpu().numpy()
        return (mask > 0.5).astype(np.float32)
    visualize_staggered_patterns(
        [safe_get_mask(m) for m in original_masks[:num_samples_to_show]],
        [safe_get_mask(m) for m in generated_masks[:num_samples_to_show]],
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

if __name__ == "__main__":
    # Run the scaled-up continuous diffusion experiment
    print("Running scaled-up continuous diffusion experiment for staggered adoption patterns...")
    results = run_scaled_diffusion_experiment()
    
    print("\n" + "="*60)
    print("Scaled-Up Continuous Diffusion Experiment Completed!")
    print("This experiment used:")
    print("- 2,000 training samples with staggered adoption patterns")
    print("- 300 epochs of training")
    print("- Larger batch size (32)")
    print("- 100 diffusion timesteps")
    print("- 50 sampling steps")
    print("- More diverse matrix sizes")
    print("="*60) 