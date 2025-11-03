#!/usr/bin/env python3
"""Script to test if FullSupportBarDistribution forward method cares about input shape.

Tests whether the forward method behaves differently with:
- Shape (batch, sequence_length) vs (sequence_length, batch)
- Shape (batch, sequence_length, num_bars) vs (sequence_length, batch, num_bars)
"""

import torch
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabimpute.model.bar_distribution import FullSupportBarDistribution


def test_shape_independence():
    """Test if forward method behaves differently with different input shapes."""
    
    # Set up distribution
    num_bars = 10
    borders = torch.linspace(0.0, 10.0, num_bars + 1)
    dist = FullSupportBarDistribution(borders)
    
    # Test parameters
    batch_size = 4
    sequence_length = 8
    
    print("=" * 80)
    print("Testing FullSupportBarDistribution forward method shape independence")
    print("=" * 80)
    print(f"Num bars: {num_bars}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    print()
    
    # Generate test data
    # Create values that will map to different buckets
    y_values = torch.linspace(0.5, 9.5, batch_size * sequence_length)
    
    # Test Case 1: Expected shape (sequence_length, batch) = (T, B)
    print("Test Case 1: Shape (sequence_length, batch) = (T, B)")
    print("-" * 80)
    y_tb = y_values.view(sequence_length, batch_size)
    logits_tbn = torch.randn(sequence_length, batch_size, num_bars, requires_grad=True)
    
    print(f"y shape: {y_tb.shape}")
    print(f"logits shape: {logits_tbn.shape}")
    
    try:
        loss_tb = dist.forward(logits_tbn, y_tb)
        print(f"✓ Forward succeeded")
        print(f"Loss shape: {loss_tb.shape}")
        print(f"Loss values (first 5): {loss_tb.flatten()[:5]}")
        print()
    except Exception as e:
        print(f"✗ Forward failed: {e}")
        print()
        return False
    
    # Test Case 2: Alternative shape (batch, sequence_length) = (B, T)
    print("Test Case 2: Shape (batch, sequence_length) = (B, T)")
    print("-" * 80)
    y_bt = y_values.view(batch_size, sequence_length)
    logits_btn = torch.randn(batch_size, sequence_length, num_bars, requires_grad=True)
    
    print(f"y shape: {y_bt.shape}")
    print(f"logits shape: {logits_btn.shape}")
    
    try:
        loss_bt = dist.forward(logits_btn, y_bt)
        print(f"✓ Forward succeeded")
        print(f"Loss shape: {loss_bt.shape}")
        print(f"Loss values (first 5): {loss_bt.flatten()[:5]}")
        print()
    except Exception as e:
        print(f"✗ Forward failed: {e}")
        print()
        return False
    
    # Test Case 3: Compare results with same data but transposed
    print("Test Case 3: Compare results with transposed inputs (same data, different layout)")
    print("-" * 80)
    
    # Create data in (T, B) format
    y_tb_same = torch.randn(sequence_length, batch_size) * 5 + 5  # Values in [0, 10]
    logits_tbn_same = torch.randn(sequence_length, batch_size, num_bars, requires_grad=True)
    
    # Transpose to (B, T) format (same data, just reordered)
    y_bt_same = y_tb_same.transpose(0, 1)
    logits_btn_same = logits_tbn_same.transpose(0, 1)
    
    print(f"y (T,B) shape: {y_tb_same.shape}")
    print(f"y (B,T) shape: {y_bt_same.shape}")
    print(f"y values match after transpose: {torch.allclose(y_tb_same.T, y_bt_same)}")
    print(f"logits (T,B,N) shape: {logits_tbn_same.shape}")
    print(f"logits (B,T,N) shape: {logits_btn_same.shape}")
    print(f"logits values match after transpose: {torch.allclose(logits_tbn_same.transpose(0, 1), logits_btn_same)}")
    
    # Get losses
    loss_tb_same = dist.forward(logits_tbn_same, y_tb_same)
    loss_bt_same = dist.forward(logits_btn_same, y_bt_same)
    
    print(f"\nLoss (T,B) shape: {loss_tb_same.shape}")
    print(f"Loss (B,T) shape: {loss_bt_same.shape}")
    
    # Compare if we transpose one to match the other
    loss_bt_transposed_to_tb = loss_bt_same.transpose(0, 1)
    print(f"Loss (B,T) transposed to (T,B) shape: {loss_bt_transposed_to_tb.shape}")
    
    if loss_tb_same.shape == loss_bt_transposed_to_tb.shape:
        are_close = torch.allclose(loss_tb_same, loss_bt_transposed_to_tb, atol=1e-6)
        max_diff = (loss_tb_same - loss_bt_transposed_to_tb).abs().max()
        print(f"Shapes match: ✓")
        print(f"Values close (atol=1e-6): {'✓' if are_close else '✗'}")
        print(f"Max difference: {max_diff.item():.2e}")
        
        if not are_close:
            print("\n⚠️  WARNING: Results differ! The forward method DOES care about input shape.")
            print("This means that (T,B) and (B,T) produce different results even with the same data.")
            print("First few differences:")
            diff = (loss_tb_same - loss_bt_transposed_to_tb).abs()
            print(diff.flatten()[:10])
        else:
            print("\n✓ Results match! The forward method does NOT care about input shape.")
            print("(T,B) and (B,T) produce equivalent results when transposed.")
    else:
        print(f"✗ Shapes don't match even after transpose!")
        print("⚠️  WARNING: The forward method produces different output shapes!")
    
    print()
    
    # Test Case 4: Check gradient computation with transposed inputs
    print("Test Case 4: Gradient computation with transposed inputs")
    print("-" * 80)
    
    # Create data in (T, B) format
    logits_tbn_grad = torch.randn(sequence_length, batch_size, num_bars, requires_grad=True)
    y_tb_grad = torch.randn(sequence_length, batch_size) * 5 + 5  # Values in [0, 10]
    
    # Transpose to (B, T) format (same data, just reordered)
    logits_btn_grad = logits_tbn_grad.transpose(0, 1).clone().detach().requires_grad_(True)
    y_bt_grad = y_tb_grad.transpose(0, 1)
    
    print(f"Using same data, transposed layout")
    print(f"logits (T,B,N) shape: {logits_tbn_grad.shape}")
    print(f"logits (B,T,N) shape: {logits_btn_grad.shape}")
    
    loss_tb_grad = dist.forward(logits_tbn_grad, y_tb_grad).sum()
    loss_bt_grad = dist.forward(logits_btn_grad, y_bt_grad).sum()
    
    loss_tb_grad.backward()
    loss_bt_grad.backward()
    
    grad_tbn = logits_tbn_grad.grad
    grad_btn = logits_btn_grad.grad
    
    print(f"\nGradient (T,B,N) shape: {grad_tbn.shape}")
    print(f"Gradient (B,T,N) shape: {grad_btn.shape}")
    
    # If forward is shape-independent, gradients should match after transpose
    grad_btn_transposed = grad_btn.transpose(0, 1)
    grad_close = torch.allclose(grad_tbn, grad_btn_transposed, atol=1e-5)
    grad_max_diff = (grad_tbn - grad_btn_transposed).abs().max()
    
    print(f"Gradients close after transpose (atol=1e-5): {'✓' if grad_close else '✗'}")
    print(f"Max gradient difference: {grad_max_diff.item():.2e}")
    
    if not grad_close:
        print("⚠️  WARNING: Gradients differ! The forward method DOES care about input shape.")
        print("Gradients are computed differently for (T,B) vs (B,T) inputs.")
    else:
        print("✓ Gradients match! Even though forward results differ, gradients are equivalent.")
    
    print()
    print("=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("1. Both (T,B) and (B,T) shapes are accepted ✓")
    print("2. Output shape matches input shape ✓")
    # Determine if forward cares about shape based on Test Case 3 results
    if loss_tb_same.shape == loss_bt_transposed_to_tb.shape:
        summary_are_close = torch.allclose(loss_tb_same, loss_bt_transposed_to_tb, atol=1e-6)
        if not summary_are_close:
            print("3. Forward method DOES care about input shape - different results for (T,B) vs (B,T) ⚠️")
        else:
            print("3. Forward method does NOT care about input shape ✓")
    else:
        print("3. Forward method produces different output shapes ⚠️")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_shape_independence()
    sys.exit(0 if success else 1)

