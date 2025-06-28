#!/usr/bin/env python3
"""
MASSIVE Scaled-Up Diffusion Experiment for Staggered Adoption Patterns
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

def run_massive_scaled_experiment():
    """Run the massive scaled-up continuous diffusion experiment."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🚀 MASSIVE SCALED-UP DIFFUSION EXPERIMENT")
    print("=" * 50)
    
    # 1. Generate MASSIVE training dataset
    print("\n📊 Generating MASSIVE training dataset...")
    data_matrices, masking_matrices = generate_staggered_adoption_data(
        num_samples=3000,  # 15x more data!
        size_range=(8, 20),  # Larger size range
        random_seed=42
    )
    
    print(f"Generated {len(data_matrices)} training samples")
    
    # Analyze training data distribution
    densities = [mask.mean() for mask in masking_matrices]
    print(f"Training data density range: {min(densities):.3f} - {max(densities):.3f}")
    print(f"Training data mean density: {np.mean(densities):.3f}")
    
    # 2. Train the MASSIVE model
    print("\n🏋️ Training MASSIVE diffusion model...")
    model_path = 'massive_diffusion_model.pth'
    
    if os.path.exists(model_path):
        print(f"Model file {model_path} found. Loading trained model...")
        model = MinimalDiffusionModel(max_size=32).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        model = train_minimal_diffusion_model(
            data_matrices=data_matrices,
            masking_matrices=masking_matrices,
            num_epochs=500,  # 5x more epochs!
            batch_size=32,  # 2x larger batch size
            max_size=32,
            device=device,
            learning_rate=3e-4,  # Lower learning rate for stability
            save_path=model_path
        )
    
    # 3. Generate test samples with different thresholds
    print("\n🎲 Generating test samples...")
    scheduler = MinimalDiffusionScheduler(num_timesteps=100)  # 5x more timesteps!
    
    # Test different matrix sizes
    test_sizes = [
        (10, 12), (12, 10), (8, 15), (15, 8), (11, 11),
        (9, 13), (13, 9), (14, 10), (10, 14), (12, 12),
        (16, 18), (18, 16), (20, 12), (12, 20), (15, 15),
        (10, 10), (14, 14), (16, 16), (18, 18), (20, 20)
    ]
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold {threshold}...")
        results = []
        
        for i, (rows, cols) in enumerate(test_sizes):
            size_info = torch.tensor([[rows, cols]], dtype=torch.float32, device=device)
            
            # Generate sample with more steps
            sample = sample_from_minimal_diffusion(
                model, scheduler, size_info, 
                num_steps=50,  # 5x more sampling steps!
                device=device, 
                threshold=threshold
            )
            
            # Extract mask and crop to actual size
            mask = sample[0, 1, :rows, :cols].cpu().numpy()
            binary_mask = (mask > threshold).astype(int)
            
            results.append({
                'size': (rows, cols),
                'mask': binary_mask,
                'density': binary_mask.mean(),
                'raw_mask': mask  # Keep raw values for analysis
            })
        
        all_results[threshold] = results
    
    # 4. Analyze results
    print("\n📈 Analyzing results...")
    
    # Density analysis per threshold
    print("\nDensity Analysis by Threshold:")
    for threshold, results in all_results.items():
        densities = [r['density'] for r in results]
        print(f"Threshold {threshold}: mean={np.mean(densities):.3f}, std={np.std(densities):.3f}, range=[{min(densities):.3f}, {max(densities):.3f}]")
    
    # 5. Visualize best results
    print("\n🎨 Creating visualizations...")
    
    # Find threshold with best density range
    best_threshold = 0.1  # Start with lowest threshold
    best_results = all_results[best_threshold]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f'Generated Staggered Adoption Patterns (Threshold={best_threshold})', fontsize=16)
    
    for i, result in enumerate(best_results[:20]):
        row, col = i // 5, i % 5
        mask = result['mask']
        size = result['size']
        density = result['density']
        
        im = axes[row, col].imshow(mask, cmap='RdYlBu_r', aspect='auto')
        axes[row, col].set_title(f'{size[0]}×{size[1]}\nDensity: {density:.3f}')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('Units')
    
    plt.tight_layout()
    plt.savefig('massive_generated_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Pattern analysis
    print("\n🔍 Analyzing staggered adoption patterns...")
    
    # Analyze patterns for best threshold
    masks_for_analysis = [result['mask'] for result in best_results]
    patterns = analyze_staggered_patterns(masks_for_analysis)
    
    # Create pattern visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pattern heatmap
    max_units = max(len(p) for p in patterns)
    pattern_matrix = np.zeros((len(patterns), max_units))
    for i, pattern in enumerate(patterns):
        pattern_matrix[i, :len(pattern)] = pattern
    
    im1 = ax1.imshow(pattern_matrix, cmap='viridis', aspect='auto')
    ax1.set_title('Generated Staggered Adoption Patterns\n(Dropout Points)')
    ax1.set_xlabel('Units')
    ax1.set_ylabel('Samples')
    plt.colorbar(im1, ax=ax1)
    
    # Density distribution
    densities = [result['density'] for result in best_results]
    ax2.hist(densities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Distribution of Mask Densities')
    ax2.set_xlabel('Density (fraction of 1s)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(densities), color='red', linestyle='--', label=f'Mean: {np.mean(densities):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('massive_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print("=" * 30)
    print(f"Training samples: {len(data_matrices)}")
    print(f"Training epochs: 500")
    print(f"Diffusion timesteps: 100")
    print(f"Sampling steps: 50")
    print(f"Test samples: {len(test_sizes)}")
    print(f"Test thresholds: {len(thresholds)}")
    
    print(f"\nBest threshold: {best_threshold}")
    best_densities = [r['density'] for r in best_results]
    print(f"Generated density range: {min(best_densities):.3f} - {max(best_densities):.3f}")
    print(f"Generated mean density: {np.mean(best_densities):.3f}")
    
    # Compare with training data
    print(f"\nTraining vs Generated:")
    print(f"Training mean density: {np.mean(densities):.3f}")
    print(f"Generated mean density: {np.mean(best_densities):.3f}")
    print(f"Density difference: {abs(np.mean(densities) - np.mean(best_densities)):.3f}")
    
    print("\n✅ MASSIVE EXPERIMENT COMPLETED!")
    return model, all_results

if __name__ == "__main__":
    run_massive_scaled_experiment() 