#!/usr/bin/env python3
"""
Quick test script for basic functionality without requiring model files.
This tests the core logic and components that don't need external models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict, List

# Import components that don't require external models
from mid_training import LatentEncoder, forward_process

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_latent_encoder_basic():
    """Test the LatentEncoder with basic functionality"""
    print("Testing LatentEncoder...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dim = 256
    thinker_hidden_size = 2560
    batch_size = 4
    seq_len = 128
    
    # Create encoder
    encoder = LatentEncoder(
        thinker_hidden_size=thinker_hidden_size,
        latent_dim=latent_dim
    ).to(device)
    
    # Create dummy input
    dummy_hidden_states = torch.randn(
        batch_size, seq_len, thinker_hidden_size, 
        device=device
    )
    
    # Forward pass
    latents = encoder(dummy_hidden_states)
    
    # Check output shape
    expected_shape = (batch_size, latent_dim)
    assert latents.shape == expected_shape, f"Expected {expected_shape}, got {latents.shape}"
    
    # Check output range (should be [-1, 1] due to Tanh)
    assert torch.all(latents >= -1) and torch.all(latents <= 1), "Latents should be in [-1, 1] range"
    
    print(f"✓ LatentEncoder works! Output shape: {latents.shape}")
    print(f"  Latent range: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
    
    return True

def test_forward_process_basic():
    """Test the forward process (masking)"""
    print("Testing forward process...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    seq_len = 128
    vocab_size = 50000
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Apply forward process
    noisy_batch, masked_indices, p_mask = forward_process(input_ids)
    
    # Check shapes
    assert noisy_batch.shape == input_ids.shape, "Noisy batch shape mismatch"
    assert masked_indices.shape == input_ids.shape, "Masked indices shape mismatch"
    assert p_mask.shape == input_ids.shape, "P mask shape mismatch"
    
    # Check that some tokens are masked (MASK token ID = 126336)
    num_masked = (noisy_batch == 126336).sum().item()
    assert num_masked > 0, "No tokens were masked"
    assert num_masked < input_ids.numel(), "All tokens were masked"
    
    # Check that masked indices match MASK tokens
    assert torch.all((noisy_batch == 126336) == masked_indices), "Masked indices don't match MASK tokens"
    
    print(f"✓ Forward process works!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Masked tokens: {num_masked}/{input_ids.numel()} ({100*num_masked/input_ids.numel():.1f}%)")
    
    return True

def test_loss_computation():
    """Test loss computation logic"""
    print("Testing loss computation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    seq_len = 64
    vocab_size = 50000
    
    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    noisy_batch, masked_indices, p_mask = forward_process(input_ids)
    
    # Create dummy logits
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    
    # Compute loss
    token_loss = F.cross_entropy(
        logits[masked_indices], 
        input_ids[masked_indices], 
        reduction='none'
    ) / p_mask[masked_indices]
    
    loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    
    # Check that loss is a scalar and positive
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"✓ Loss computation works! Loss: {loss.item():.4f}")
    
    return True

def test_gradient_flow_simulation():
    """Simulate gradient flow through the latent encoder"""
    print("Testing gradient flow simulation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dim = 256
    thinker_hidden_size = 2560
    batch_size = 2
    seq_len = 64
    
    # Create encoder
    encoder = LatentEncoder(
        thinker_hidden_size=thinker_hidden_size,
        latent_dim=latent_dim
    ).to(device)
    
    # Create dummy input
    dummy_hidden_states = torch.randn(
        batch_size, seq_len, thinker_hidden_size, 
        device=device, requires_grad=True
    )
    
    # Forward pass
    latents = encoder(dummy_hidden_states)
    
    # Create dummy target and compute loss
    target = torch.randn_like(latents)
    loss = F.mse_loss(latents, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert dummy_hidden_states.grad is not None, "Input gradients should exist"
    
    # Check encoder gradients
    encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert len(encoder_grads) > 0, "Encoder should have gradients"
    
    print(f"✓ Gradient flow simulation works!")
    print(f"  Encoder parameters with gradients: {len(encoder_grads)}/{len(list(encoder.parameters()))}")
    
    return True

def run_quick_tests():
    """Run all quick tests"""
    print("🚀 Running quick sanity checks...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    tests = [
        test_latent_encoder_basic,
        test_forward_process_basic,
        test_loss_computation,
        test_gradient_flow_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print("=" * 50)
    print(f"Quick tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("Your core components are working correctly.")
        print("Run the full sanity_check.py to test with actual models.")
    else:
        print("⚠ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    run_quick_tests() 