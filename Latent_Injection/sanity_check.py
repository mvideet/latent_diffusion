#!/usr/bin/env python3
"""
Sanity check script for the latent injection training pipeline.
This script tests each component individually and then the full pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import random
import numpy as np
from typing import Dict, List

# Import your components
from mid_training import (
    LatentEncoder, 
    ReasoningLatentGenerator, 
    MidTrainingPipeline,
    forward_process
)
from film_llada_model import load_film_model

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_latent_encoder():
    """Test the LatentEncoder component"""
    print("=" * 50)
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
    
    print(f"✓ LatentEncoder test passed! Output shape: {latents.shape}")
    print(f"  Latent range: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
    
    return encoder

def test_forward_process():
    """Test the forward process (masking)"""
    print("=" * 50)
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
    
    print(f"✓ Forward process test passed!")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Masked tokens: {num_masked}/{input_ids.numel()} ({100*num_masked/input_ids.numel():.1f}%)")
    
    return noisy_batch, masked_indices, p_mask

def test_reasoning_latent_generator():
    """Test the ReasoningLatentGenerator component"""
    print("=" * 50)
    print("Testing ReasoningLatentGenerator...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dim = 256
    batch_size = 2
    seq_len = 64
    
    try:
        # Create generator (this will try to load models)
        generator = ReasoningLatentGenerator(
            thinker_path="./thinker",
            latent_dim=latent_dim,
            device=device
        )
        
        # Create dummy input
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        
        # Test latent generation
        latents = generator.generate_latents(
            input_ids, 
            strategy="mixed",
            use_dummy=0.0  # No dummy latents for testing
        )
        
        # Check output shape
        expected_shape = (batch_size, latent_dim)
        assert latents.shape == expected_shape, f"Expected {expected_shape}, got {latents.shape}"
        
        print(f"✓ ReasoningLatentGenerator test passed! Output shape: {latents.shape}")
        
        # Test prompt generation
        contexts = ["This is a test context for reasoning."] * batch_size
        prompts = generator.generate_reasoning_prompts(contexts, strategy="summary")
        
        assert len(prompts) == batch_size, "Number of prompts doesn't match batch size"
        assert all(isinstance(p, str) for p in prompts), "All prompts should be strings"
        assert all(len(p) > 0 for p in prompts), "All prompts should be non-empty"
        
        print(f"✓ Prompt generation test passed! Generated {len(prompts)} prompts")
        
        return generator
        
    except Exception as e:
        print(f"⚠ ReasoningLatentGenerator test failed: {e}")
        print("  This might be due to missing model files. Continuing with other tests...")
        return None

def test_film_model():
    """Test the FiLM-enabled LLaDA model"""
    print("=" * 50)
    print("Testing FiLM-enabled LLaDA model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latent_dim = 256
    batch_size = 2
    seq_len = 64
    
    try:
        # Load model
        model = load_film_model("./llada-8b", latent_dim, device)
        
        # Create dummy inputs
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        latents = torch.randn(batch_size, latent_dim, device=device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, latent=latents)
        
        # Check output shape
        expected_logits_shape = (batch_size, seq_len, model.config.vocab_size)
        assert outputs.logits.shape == expected_logits_shape, f"Expected {expected_logits_shape}, got {outputs.logits.shape}"
        
        print(f"✓ FiLM model test passed! Logits shape: {outputs.logits.shape}")
        
        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return model
        
    except Exception as e:
        print(f"⚠ FiLM model test failed: {e}")
        print("  This might be due to missing model files. Continuing with other tests...")
        return None

def test_full_pipeline():
    """Test the complete training pipeline"""
    print("=" * 50)
    print("Testing full training pipeline...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Initialize pipeline
        pipeline = MidTrainingPipeline(
            llada_path="./llada-8b",
            thinker_path="./thinker",
            latent_dim=256,
            device=device
        )
        
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, 50000, (2, 128)).to(device)
        }
        
        # Run training step
        metrics = pipeline.training_step(batch)
        
        # Check metrics
        assert "loss" in metrics, "Loss should be in metrics"
        assert "masked_tokens" in metrics, "Masked tokens should be in metrics"
        assert "total_tokens" in metrics, "Total tokens should be in metrics"
        
        assert metrics["loss"] > 0, "Loss should be positive"
        assert metrics["masked_tokens"] > 0, "Some tokens should be masked"
        assert metrics["total_tokens"] > 0, "Total tokens should be positive"
        
        print(f"✓ Full pipeline test passed!")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Masked tokens: {metrics['masked_tokens']}")
        print(f"  Total tokens: {metrics['total_tokens']}")
        
        return pipeline
        
    except Exception as e:
        print(f"⚠ Full pipeline test failed: {e}")
        print("  This might be due to missing model files. Check the individual component tests above.")
        return None

def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("=" * 50)
    print("Testing gradient flow...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Initialize pipeline
        pipeline = MidTrainingPipeline(
            llada_path="./llada-8b",
            thinker_path="./thinker",
            latent_dim=256,
            device=device
        )
        
        # Create dummy batch
        batch = {
            "input_ids": torch.randint(0, 50000, (2, 64)).to(device)
        }
        
        # Run training step
        metrics = pipeline.training_step(batch)
        
        # Check that gradients exist for trainable parameters
        film_params = [p for p in pipeline.llada_model.parameters() if 'film_' in p.name]
        latent_params = list(pipeline.latent_generator.latent_encoder.parameters())
        
        # Check FiLM parameters
        film_grads = [p.grad for p in film_params if p.grad is not None]
        assert len(film_grads) > 0, "No gradients for FiLM parameters"
        
        # Check latent encoder parameters
        latent_grads = [p.grad for p in latent_params if p.grad is not None]
        assert len(latent_grads) > 0, "No gradients for latent encoder parameters"
        
        print(f"✓ Gradient flow test passed!")
        print(f"  FiLM parameters with gradients: {len(film_grads)}/{len(film_params)}")
        print(f"  Latent encoder parameters with gradients: {len(latent_grads)}/{len(latent_params)}")
        
    except Exception as e:
        print(f"⚠ Gradient flow test failed: {e}")

def run_all_sanity_checks():
    """Run all sanity checks"""
    print("🧪 Starting sanity checks for latent injection training pipeline...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run individual component tests
    test_latent_encoder()
    test_forward_process()
    test_reasoning_latent_generator()
    test_film_model()
    
    # Run full pipeline test
    test_full_pipeline()
    
    # Test gradient flow
    test_gradient_flow()
    
    print("=" * 50)
    print("🎉 Sanity checks completed!")
    print("\nIf all tests passed (✓), your training pipeline is ready to use.")
    print("If some tests failed (⚠), check that:")
    print("  1. Model files exist in ./llada-8b/ and ./thinker/")
    print("  2. All dependencies are installed")
    print("  3. You have sufficient GPU memory")

if __name__ == "__main__":
    run_all_sanity_checks() 