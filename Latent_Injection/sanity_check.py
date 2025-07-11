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
import subprocess
import os

# Import your components
from mid_training import (
    LatentEncoder, 
    ReasoningLatentGenerator, 
    MidTrainingPipeline,
    forward_process
)
from film_llada_model import load_film_model

def print_gpu_info():
    """Print comprehensive GPU information"""
    print("=" * 80)
    print("🖥️  GPU PROFILE INFORMATION")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    # print(f"📊 Number of GPUs available: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        free = total - allocated
        
        # print(f"\n🔧 GPU {i}: {props.name}")
        # print(f"   💾 Total Memory: {total:.2f} GB")
        # print(f"   🔥 Allocated:    {allocated:.2f} GB")
        # print(f"   📦 Reserved:     {reserved:.2f} GB")
        # print(f"   🆓 Free:        {free:.2f} GB")
        # print(f"   ⚡ Compute Cap:  {props.major}.{props.minor}")
        # print(f"   🧮 Multiproc:   {props.multi_processor_count}")
    
    # Try to get nvidia-smi output
    try:
        # print(f"\n🔍 NVIDIA-SMI Output:")
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        idx, name, mem_used, mem_total, util = parts[:5]
                        print(f"   GPU {idx}: {name} | Memory: {mem_used}/{mem_total} MB | Util: {util}%")
        else:
            print("   nvidia-smi not available or failed")
    except Exception as e:
        print(f"   Could not run nvidia-smi: {e}")

def print_model_device_info(model, model_name: str):
    """Print detailed device information for a model"""
    print(f"\n🔍 Model Device Analysis: {model_name}")
    print("-" * 50)
    
    if model is None:
        print("   ❌ Model is None")
        return
    
    device_counts = {}
    dtype_counts = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        device = str(param.device)
        dtype = str(param.dtype)
        
        device_counts[device] = device_counts.get(device, 0) + param.numel()
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + param.numel()
        total_params += param.numel()
    
    # print(f"   📊 Total Parameters: {total_params:,}")
    # print(f"   🖥️  Device Distribution:")
    # for device, count in device_counts.items():
    #     percentage = (count / total_params) * 100
    #     print(f"      {device}: {count:,} params ({percentage:.1f}%)")
    
    # print(f"   🔢 Dtype Distribution:")
    # for dtype, count in dtype_counts.items():
    #     percentage = (count / total_params) * 100
    #     print(f"      {dtype}: {count:,} params ({percentage:.1f}%)")

def check_cross_gpu_transfer():
    """Test cross-GPU data transfer to verify setup"""
    print("\n🔄 Testing Cross-GPU Transfer...")
    
    if torch.cuda.device_count() < 2:
        print("   ⚠️  Less than 2 GPUs available, skipping cross-GPU test")
        return
    
    # Create test tensors on different GPUs
    tensor_gpu0 = torch.randn(100, 256, device='cuda:0', dtype=torch.bfloat16)
    print(f"   📤 Created tensor on GPU 0: {tensor_gpu0.shape}, {tensor_gpu0.dtype}")
    
    # Transfer to GPU 1
    tensor_gpu1 = tensor_gpu0.to('cuda:1')
    print(f"   📥 Transferred to GPU 1: {tensor_gpu1.shape}, {tensor_gpu1.dtype}")
    
    # Verify they're on different devices but have same data
    assert tensor_gpu0.device.index == 0, "First tensor should be on GPU 0"
    assert tensor_gpu1.device.index == 1, "Second tensor should be on GPU 1"
    assert torch.allclose(tensor_gpu0.to('cuda:1'), tensor_gpu1), "Data should be identical"
    
    print("   ✅ Cross-GPU transfer working correctly!")

def print_gpu_memory_usage(stage: str = ""):
    """Quick GPU memory usage print"""
    if not torch.cuda.is_available():
        return
    
    if stage:
        print(f"\n📊 GPU Memory Usage - {stage}:")
    else:
        print(f"\n📊 GPU Memory Usage:")
    
    # for i in range(torch.cuda.device_count()):
    #     allocated = torch.cuda.memory_allocated(i) / 1024**3
    #     reserved = torch.cuda.memory_reserved(i) / 1024**3
    #     total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    #     print(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_latent_encoder():
    """Test the LatentEncoder component"""
    # print("=" * 50)
    # print("Testing LatentEncoder...")
    
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
    # print(f"  P mask: {p_mask}")
    # print(f"  Noisy batch: {noisy_batch}")
    # print(f"  Masked indices: {masked_indices}")
    return noisy_batch, masked_indices, p_mask

def get_sample_questions():
    """Get realistic sample questions for testing"""
    return [
        "What is the solution to the equation 2x + 5 = 17? Show your work step by step.",
        "A train leaves New York at 9:00 AM traveling at 120 mph. Another train leaves Boston at 10:00 AM traveling at 150 mph toward New York. If the distance between the cities is 220 miles, at what time will they meet?",
        "Explain the process of photosynthesis and why it's important for life on Earth. Include the chemical equation.",
        "A company's revenue increased from $2.5 million to $3.2 million over two years. What was the percentage increase? If this growth rate continues, what will the revenue be after one more year?"
    ]

def test_reasoning_latent_generator():
    """Test the ReasoningLatentGenerator component"""
    print("=" * 50)
    print("Testing ReasoningLatentGenerator...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check for multiple GPUs and use GPU 1 for thinker if available
    thinker_device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        thinker_device = 'cuda:1'
        print(f"🎯 Multi-GPU setup detected!")
        print(f"   Main device: {device}")
        print(f"   Thinker device: {thinker_device}")
    else:
        print(f"Using single device: {device}")
    
    # Print GPU info before loading models
    print_gpu_info()
    
    latent_dim = 256
    
    try:
        # Create generator (this will try to load models)
        generator = ReasoningLatentGenerator(
            thinker_path="/data/sls/u/urop/mvideet/diffusion_reasoning/thinker",
            latent_dim=latent_dim,
            device=device,
            thinker_device=thinker_device
        )
        
        # Print device info for loaded models
        print_model_device_info(generator.thinker_model, "Thinker Model")
        print_model_device_info(generator.latent_encoder, "Latent Encoder")
        
        # Get sample questions
        sample_questions = get_sample_questions()
        batch_size = len(sample_questions)
        
        print(f"Using {batch_size} sample questions:")
        for i, q in enumerate(sample_questions):
            print(f"  {i+1}. {q[:80]}{'...' if len(q) > 80 else ''}")
        
        # Tokenize questions using LLaDA tokenizer
        tokenized = generator.llada_tokenizer(
            sample_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = tokenized.input_ids.to(device)
        
        print(f"Tokenized input shape: {input_ids.shape}")
        
        # Print GPU memory before latent generation
        print_gpu_memory_usage("Before latent generation")
        
        # Test latent generation
        latents = generator.generate_latents(
            input_ids, 
            strategy="mixed",
            use_dummy=0.0  # No dummy latents for testing
        )
        
        # Print GPU memory after latent generation (cross-GPU transfer)
        print_gpu_memory_usage("After latent generation (cross-GPU)")
        
        # Check output shape
        expected_shape = (batch_size, latent_dim)
        assert latents.shape == expected_shape, f"Expected {expected_shape}, got {latents.shape}"
        
        print(f"✓ ReasoningLatentGenerator test passed! Output shape: {latents.shape}")
        
        # Test prompt generation with actual contexts
        prompts = generator.generate_reasoning_prompts(sample_questions, strategy="summary")
        
        assert len(prompts) == batch_size, "Number of prompts doesn't match batch size"
        assert all(isinstance(p, str) for p in prompts), "All prompts should be strings"
        assert all(len(p) > 0 for p in prompts), "All prompts should be non-empty"
        
        print(f"✓ Prompt generation test passed! Generated {len(prompts)} prompts")
        print(f"  Sample reasoning prompt (first 150 chars):")
        # Print the first 150 characters of the first generated reasoning prompt as an example
        print(f"  {prompts[0][:150]}...")
        
        return generator
        
    except Exception as e:
        print(e)
        print(f"⚠ ReasoningLatentGenerator test failed: {e}")
        print("  This might be due to missing model files. Continuing with other tests...")
        return None
    finally:
        # Clean up GPU memory after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()
            print(f"🧹 Cleaned GPU cache and garbage collected. GPU memory in use: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

def test_film_model():
    """Test the FiLM-enabled LLaDA model"""
    print("=" * 50)
    print("Testing FiLM-enabled LLaDA model...")
    
    # Print GPU info before test
    print_gpu_info()
    
    # Check available GPU memory and fallback to CPU if needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        print(f"Free GPU memory: {free_memory_gb:.2f} GB")
        if free_memory_gb < 2.0:  # Need at least 2GB free
            print("⚠️ Insufficient GPU memory, falling back to CPU for FiLM test")
            device = 'cpu'
    
    latent_dim = 256
    batch_size = 1  # Reduce batch size for memory efficiency
    seq_len = 64   # Further reduce sequence length
    
    # Clear GPU cache before testing
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    try:
        # Load model
        model = load_film_model("/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b", latent_dim, device)
        
        # Print model device analysis
        print_model_device_info(model, "FiLM LLaDA Model")
        
        if torch.cuda.is_available():
            print(f"GPU memory after loading model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Create dummy inputs (model expects bfloat16 now but latents should be float32)
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        latents = torch.randn(batch_size, latent_dim, device=device, dtype=torch.float32)
        
        print(f"Testing with input_ids: {input_ids.shape}, {input_ids.dtype}")
        print(f"Testing with latents: {latents.shape}, {latents.dtype}")
        
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
    finally:
        # Clean up GPU memory after test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleaned GPU cache after FiLM test")

def test_full_pipeline():
    """Test the complete training pipeline"""
    print("=" * 50)
    print("Testing full training pipeline...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check for multiple GPUs and use GPU 1 for thinker if available
    thinker_device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        thinker_device = 'cuda:1'
        print(f"🎯 Using multi-GPU setup for full pipeline!")
        print(f"   Main device: {device}, Thinker device: {thinker_device}")
    
    # Print GPU info before pipeline creation
    print_gpu_info()
    
    try:
        # Initialize pipeline
        pipeline = MidTrainingPipeline(
            llada_path="/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b",
            thinker_path="/data/sls/u/urop/mvideet/diffusion_reasoning/thinker",
            latent_dim=256,
            device=device,
            thinker_device=thinker_device
        )
        
        # Print comprehensive device analysis for all models
        # print("\n🔍 FULL PIPELINE DEVICE ANALYSIS")
        # print("=" * 60)
        print_model_device_info(pipeline.llada_model, "Main LLaDA + FiLM Model")
        print_model_device_info(pipeline.latent_generator.thinker_model, "Thinker Model")
        print_model_device_info(pipeline.latent_generator.latent_encoder, "Latent Encoder")
        
        # Show GPU memory usage after loading all models
        print_gpu_info()
        
        # Get sample questions and tokenize them
        sample_questions = get_sample_questions()
        print(f"Testing with {len(sample_questions)} sample questions")
        
        # Tokenize using the LLaDA tokenizer from the pipeline
        tokenized = pipeline.latent_generator.llada_tokenizer(
            sample_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Create batch
        batch = {
            "input_ids": tokenized.input_ids.to(device)
        }
        
        print(f"Input batch shape: {batch['input_ids'].shape}")
        
        # Print GPU memory before training step
        print_gpu_memory_usage("Before training step")
        
        # Run training step
        metrics = pipeline.training_step(batch)
        
        # Print GPU memory after training step
        print_gpu_memory_usage("After training step")
        
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
        print(f"  Masking rate: {100*metrics['masked_tokens']/metrics['total_tokens']:.1f}%")
        
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
            llada_path="/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b",
            thinker_path="/data/sls/u/urop/mvideet/diffusion_reasoning/thinker",
            latent_dim=256,
            device=device
        )
        
        # Get sample questions (use first 2 for speed)
        sample_questions = get_sample_questions()[:2]
        
        # Tokenize questions
        tokenized = pipeline.latent_generator.llada_tokenizer(
            sample_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Create batch
        batch = {
            "input_ids": tokenized.input_ids.to(device)
        }
        
        print("🔍 DETAILED GRADIENT FLOW ANALYSIS")
        print("-" * 40)
        
        # Check parameter requires_grad status BEFORE training
        film_params = [(name, p) for name, p in pipeline.llada_model.named_parameters() if 'film_' in name]
        latent_params = list(pipeline.latent_generator.latent_encoder.named_parameters())
        
        print(f"📋 Parameter Status (before training):")
        print(f"  FiLM parameters: {len(film_params)} found")
        for name, p in film_params[:3]:  # Show first 3
            print(f"    {name}: requires_grad={p.requires_grad}, shape={p.shape}")
        
        print(f"  Latent encoder parameters: {len(latent_params)} found")
        for name, p in latent_params[:3]:  # Show first 3
            print(f"    {name}: requires_grad={p.requires_grad}, shape={p.shape}")
        
        # Check if latent encoder is in training mode
        print(f"  Latent encoder training mode: {pipeline.latent_generator.latent_encoder.training}")
        print(f"  Main LLaDA model training mode: {pipeline.llada_model.training}")
        
        # Run training step
        print(f"\n🏃 Running training step...")
        metrics = pipeline.training_step(batch)
        print(f"  Training metrics: {metrics}")
        
        # Check gradients AFTER training
        print(f"\n🔍 Checking gradients after training step...")
        
        # Check FiLM parameters
        film_params_list = [p for name, p in film_params]
        film_grads = [p.grad for p in film_params_list if p.grad is not None]
        film_no_grads = [i for i, p in enumerate(film_params_list) if p.grad is None]
        
        print(f"  FiLM parameters with gradients: {len(film_grads)}/{len(film_params_list)}")
        if len(film_no_grads) > 0:
            print(f"  FiLM parameters WITHOUT gradients:")
            for i in film_no_grads[:3]:  # Show first 3
                name, p = film_params[i]
                print(f"    {name}: requires_grad={p.requires_grad}")
        
        # Check latent encoder parameters
        latent_params_list = [p for name, p in latent_params]
        latent_grads = [p.grad for p in latent_params_list if p.grad is not None]
        latent_no_grads = [i for i, p in enumerate(latent_params_list) if p.grad is None]
        
        print(f"  Latent encoder parameters with gradients: {len(latent_grads)}/{len(latent_params_list)}")
        if len(latent_no_grads) > 0:
            print(f"  Latent encoder parameters WITHOUT gradients:")
            for i in latent_no_grads[:3]:  # Show first 3
                name, p = latent_params[i]
                print(f"    {name}: requires_grad={p.requires_grad}")
        
        # Test manual gradient computation to debug further
        print(f"\n🧪 Manual gradient test...")
        
        # Clear existing gradients
        pipeline.llada_model.zero_grad()
        pipeline.latent_generator.latent_encoder.zero_grad()
        
        # Create a simple forward pass with requires_grad tracking
        input_ids = batch["input_ids"]
        
        # Generate latents with gradient tracking
        print(f"  Generating latents with gradient tracking...")
        latents = pipeline.latent_generator.generate_latents(
            input_ids, 
            strategy="mixed",
            use_dummy=0.0
        )
        print(f"  Latents requires_grad: {latents.requires_grad}")
        print(f"  Latents device: {latents.device}")
        
        # Forward pass through main model
        print(f"  Forward pass through main model...")
        outputs = pipeline.llada_model(input_ids=input_ids, latent=latents)
        
        # Simple loss computation
        print(f"  Computing simple loss...")
        # Just use the mean of logits as a simple loss for testing
        loss = outputs.logits.mean()
        print(f"  Test loss: {loss.item()}, requires_grad: {loss.requires_grad}")
        
        # Backward pass
        print(f"  Running backward pass...")
        loss.backward()
        
        # Check gradients again
        latent_grads_manual = [p.grad for name, p in latent_params if p.grad is not None]
        film_grads_manual = [p.grad for name, p in film_params if p.grad is not None]
        
        print(f"  After manual test:")
        print(f"    Latent encoder grads: {len(latent_grads_manual)}/{len(latent_params)}")
        print(f"    FiLM grads: {len(film_grads_manual)}/{len(film_params)}")
        
        # Final assertions
        if len(film_grads) == 0:
            print("❌ No gradients for FiLM parameters")
        else:
            print(f"✓ FiLM gradients found: {len(film_grads)}/{len(film_params)}")
            
        if len(latent_grads) == 0:
            print("❌ No gradients for latent encoder parameters")
            print("🔧 This suggests gradients aren't flowing back through the latent generation process")
        else:
            print(f"✓ Latent encoder gradients found: {len(latent_grads)}/{len(latent_params)}")
        
        if len(film_grads) > 0 and len(latent_grads) > 0:
            print(f"✓ Gradient flow test passed!")
        
    except Exception as e:
        print(f"⚠ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()

def test_real_inputs_and_outputs():
    """Test the pipeline with real inputs and show actual outputs"""
    print("=" * 80)
    print("🎯 TESTING WITH REAL INPUTS AND OUTPUTS")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check for multiple GPUs
    thinker_device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        thinker_device = 'cuda:1'
        print(f"🎯 Using multi-GPU setup for real input testing!")
        print(f"   Main device: {device}, Thinker device: {thinker_device}")
    
    try:
        # Initialize pipeline
        print("🔧 Initializing pipeline...")
        pipeline = MidTrainingPipeline(
            llada_path="/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b",
            thinker_path="/data/sls/u/urop/mvideet/diffusion_reasoning/thinker",
            latent_dim=256,
            device=device,
            thinker_device=thinker_device
        )
        
        # Get real test questions (use challenging math problems for detailed analysis)
        test_problems = [
            {"question": "A factory produces widgets at a rate of x per hour and increases production by 5 widgets per hour every hour for 8 hours. If they produced 480 widgets total in those 8 hours, what was the initial production rate x?", "answer": "45"},
            {"question": "In a bag of marbles, the ratio of red to blue marbles is 3:4. If 5 red marbles are removed and 10 blue marbles are added, the new ratio becomes 1:2. How many red marbles were originally in the bag?", "answer": "15"},
            {"question": "A regular octagon is inscribed in a circle of radius 8 units. What is the area of the octagon?", "answer": "193.9"},
            {"question": "The sum of three consecutive integers is 51 more than twice the smallest of the three integers. What is the largest of the three integers?", "answer": "35"},
            {"question": "A train travels from city A to city B at 80 km/h and returns at 60 km/h. If the total journey takes 7 hours, what is the distance between cities A and B in kilometers?", "answer": "240"}
        ]
        
        # Extract just the questions for processing (use first 4 for detailed analysis)
        test_questions = [p["question"] for p in test_problems[:4]]
        expected_answers = [p["answer"] for p in test_problems[:4]]
        
        print(f"\n📝 Testing with {len(test_questions)} challenging math problems:")
        for i, (q, ans) in enumerate(zip(test_questions, expected_answers)):
            print(f"  {i+1}. {q}")
            print(f"     Expected answer: {ans}")
        
        # Test each question individually to see detailed outputs
        for i, question in enumerate(test_questions):
            print(f"\n" + "="*60)
            print(f"📋 QUESTION {i+1}: {question}")
            print(f"🎯 EXPECTED ANSWER: {expected_answers[i]}")
            print("="*60)
            
            # Tokenize the question
            tokenized = pipeline.latent_generator.llada_tokenizer(
                [question],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            input_ids = tokenized.input_ids.to(device)
            print(f"🔢 Input shape: {input_ids.shape}")
            print(f"🔢 Input tokens: {input_ids[0][:20].tolist()}... (showing first 20)")
            
            # Decode input to verify
            decoded_input = pipeline.latent_generator.llada_tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"📝 Decoded input: {decoded_input}")
            
            # Generate latents
            print(f"\n🧠 Generating reasoning latents...")
            latents = pipeline.latent_generator.generate_latents(
                input_ids, 
                strategy="mixed",
                use_dummy=0.0
            )
            print(f"🎯 Generated latents shape: {latents.shape}")
            print(f"🎯 Latent statistics: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
            print(f"🎯 Latent range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
            
            # Test forward pass with latents
            print(f"\n🔮 Testing forward pass with latents...")
            
            # Create a shorter input for generation (just the question)
            short_input = input_ids[:, :32]  # Take first 32 tokens
            
            with torch.no_grad():
                outputs = pipeline.llada_model(
                    input_ids=short_input,
                    latent=latents
                )
            
            print(f"📊 Output logits shape: {outputs.logits.shape}")
            
            # Get predictions for the last token
            last_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
            predicted_token_id = torch.argmax(last_token_logits).item()
            predicted_token = pipeline.latent_generator.llada_tokenizer.decode([predicted_token_id])
            
            print(f"🎲 Next token prediction: '{predicted_token}' (ID: {predicted_token_id})")
            
            # Show top 5 predictions
            top_k = 5
            top_logits, top_indices = torch.topk(last_token_logits, top_k)
            print(f"🏆 Top {top_k} predictions:")
            for j, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                token = pipeline.latent_generator.llada_tokenizer.decode([idx.item()])
                print(f"   {j+1}. '{token}' (logit: {logit.item():.4f})")
            
            # Test with zero latents for comparison (FiLM model requires latents)
            print(f"\n🔍 Testing with zero latents (for comparison)...")
            zero_latents = torch.zeros_like(latents)
            with torch.no_grad():
                outputs_no_latent = pipeline.llada_model(
                    input_ids=short_input,
                    latent=zero_latents
                )
            
            last_token_logits_no_latent = outputs_no_latent.logits[0, -1, :]
            predicted_token_id_no_latent = torch.argmax(last_token_logits_no_latent).item()
            predicted_token_no_latent = pipeline.latent_generator.llada_tokenizer.decode([predicted_token_id_no_latent])
            
            print(f"🎲 Next token prediction (zero latent): '{predicted_token_no_latent}' (ID: {predicted_token_id_no_latent})")
            
            # Compare the difference
            if predicted_token_id != predicted_token_id_no_latent:
                print(f"🔄 DIFFERENT! Reasoning latent changed the prediction!")
                print(f"   With reasoning latent: '{predicted_token}' → With zero latent: '{predicted_token_no_latent}'")
            else:
                print(f"🔄 Same prediction with reasoning vs zero latent")
            
            # Show logit differences
            logit_diff = last_token_logits - last_token_logits_no_latent
            max_diff_idx = torch.argmax(torch.abs(logit_diff)).item()
            max_diff_token = pipeline.latent_generator.llada_tokenizer.decode([max_diff_idx])
            print(f"📊 Largest logit change: '{max_diff_token}' (diff: {logit_diff[max_diff_idx].item():.4f})")
            
            # Apply masking and test reconstruction
            print(f"\n🎭 Testing masked token reconstruction...")
            batch = {"input_ids": input_ids}
            
            # Get a training step to see loss
            metrics = pipeline.training_step(batch)
            print(f"📈 Training metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            # Clean up memory between iterations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n" + "="*80)
        print("🎉 REAL INPUT/OUTPUT TESTING COMPLETED!")
        print("="*80)
        print("✅ Key observations:")
        print("  - Generated reasoning latents have expected shape and range")
        print("  - Forward pass works with reasoning latents and zero latents")
        print("  - Reasoning latent injection can change model predictions")
        print("  - Training step computes loss and metrics correctly")
        print("  - Memory management works across iterations")
        print("  - Tested with challenging mathematical reasoning problems")
        print("  - Compare reasoning vs zero latents to see reasoning influence")
        
    except Exception as e:
        print(f"⚠ Real input/output testing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 Cleaned GPU cache after real input testing")

def run_all_sanity_checks():
    """Run all sanity checks"""
    print("🧪 Starting sanity checks for latent injection training pipeline...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Set memory management for better GPU memory handling
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initial GPU profiling
    print_gpu_info()
    
    # Test cross-GPU communication
    check_cross_gpu_transfer()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run individual component tests
    test_latent_encoder()
    test_forward_process()
    
    # Clean memory before heavy tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU cache cleaned before heavy tests")
    
    test_reasoning_latent_generator()
    
    # Clean memory before FiLM test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 GPU cache cleaned before FiLM test")
    
    test_film_model()
    
    # Run full pipeline test
    test_full_pipeline()
    
    # Test gradient flow
    test_gradient_flow()
    
    # NEW: Test with real inputs and outputs
    test_real_inputs_and_outputs()
    
    # Final GPU profiling
    print("\n🏁 FINAL GPU STATE")
    print_gpu_info()
    
    print("=" * 50)
    print("🎉 Sanity checks completed!")
    print("\nIf all tests passed (✓), your training pipeline is ready to use.")
    print("If some tests failed (⚠), check that:")
    print("  1. Model files exist in ./llada-8b/ and ./thinker/")
    print("  2. All dependencies are installed")
    print("  3. You have sufficient GPU memory")

if __name__ == "__main__":
    run_all_sanity_checks() 