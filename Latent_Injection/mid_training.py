# mid_training_with_latents.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
import math
import random

# Import your custom FiLM model from my previous response
from film_llada_model import LLaDAModelLMWithFiLM, load_film_model
from llada_8b.configuration_llada import LLaDAConfig

class LatentEncoder(nn.Module):
    """
    Encode thinker outputs into compact latent vectors
    """
    def __init__(self, thinker_hidden_size: int = 2560, latent_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(thinker_hidden_size, latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
    def forward(self, thinker_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thinker_hidden_states: [batch, seq_len, hidden_size] from thinker's last layer
        Returns:
            latent: [batch, latent_dim] compressed reasoning latent
        """
        # Pool over sequence dimension (you could also use attention pooling)
        pooled = thinker_hidden_states.mean(dim=1)  # [batch, hidden_size]
        return self.encoder(pooled)  # [batch, latent_dim]

class ReasoningLatentGenerator:
    """
    Generate reasoning latents using the thinker model
    """
    def __init__(self, thinker_path: str = "./thinker", latent_dim: int = 256, device: str = 'cuda', thinker_device: Optional[str] = None):
        self.device = device
        self.latent_dim = latent_dim
        
        # Use separate device for thinker if specified, otherwise same as main device
        self.thinker_device = thinker_device if thinker_device is not None else device
        print(f"🧠 Thinker model (frozen): {self.thinker_device}")
        print(f"🎯 LLaDA + Latent Encoder (trainable): {self.device}")
        if self.thinker_device != self.device:
            print(f"🔄 Cross-GPU setup: Thinker outputs will be transferred to main device")
        
        # Load thinker model and tokenizer on separate device
        print(f"Loading thinker model on {self.thinker_device}...")
        self.thinker_model = AutoModelForCausalLM.from_pretrained(
            thinker_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.thinker_device).eval()
        
        self.thinker_tokenizer = AutoTokenizer.from_pretrained(thinker_path)
        if self.thinker_tokenizer.pad_token is None:
            self.thinker_tokenizer.pad_token = self.thinker_tokenizer.eos_token
        
        # DEBUG: Verify thinker model is working
        print(f"🔍 THINKER MODEL VERIFICATION")
        print(f"   Model type: {type(self.thinker_model)}")
        print(f"   Config: {self.thinker_model.config}")
        print(f"   Vocab size: {self.thinker_model.config.vocab_size}")
        try:
            print(f"   Hidden size: {self.thinker_model.config.hidden_size}")
        except:
            print(f"   Hidden size: {getattr(self.thinker_model.config, 'd_model', 'unknown')}")
        
        # Test with a simple prompt
        test_prompt = "The capital of France is"
        test_inputs = self.thinker_tokenizer(test_prompt, return_tensors="pt").to(self.thinker_device)
        print(f"   Test prompt: '{test_prompt}'")
        print(f"   Test input ids: {test_inputs.input_ids}")
        
        with torch.no_grad():
            # Try proper generation instead of argmax
            test_outputs = self.thinker_model.generate(
                **test_inputs,
                max_new_tokens=20,
                do_sample=False,  # Deterministic for testing
                pad_token_id=self.thinker_tokenizer.eos_token_id
            )
            test_decoded = self.thinker_tokenizer.decode(test_outputs[0], skip_special_tokens=True)
            print(f"   Test output: '{test_decoded}'")
            
            # Also test the raw forward pass
            raw_outputs = self.thinker_model(**test_inputs)
            raw_decoded = self.thinker_tokenizer.decode(raw_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
            print(f"   Raw argmax: '{raw_decoded}'")
        print(f"🔍 END THINKER VERIFICATION")
            
        # Initialize latent encoder on MAIN device (for gradient updates with LLaDA model)
        self.latent_encoder = LatentEncoder(
            thinker_hidden_size=2560,  # Phi-2 hidden size
            latent_dim=latent_dim
        ).to(self.device).to(torch.bfloat16)  # Keep on main device for training
        
        # DEBUG: Print model dtypes and devices after initialization
        thinker_param = next(self.thinker_model.parameters())
        encoder_param = next(self.latent_encoder.parameters())
        print(f"🔍 DEBUG - Thinker model: {thinker_param.dtype} on {thinker_param.device}")
        print(f"🔍 DEBUG - Latent encoder: {encoder_param.dtype} on {encoder_param.device}")
        
        # Load LLaDA tokenizer for context preparation
        self.llada_tokenizer = AutoTokenizer.from_pretrained("/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b", trust_remote_code=True)
        
    def generate_reasoning_prompts(self, contexts: List[str], strategy: str = "mixed") -> List[str]:
        """
        Generate reasoning prompts for the thinker model
        """
        prompts = []
        
        for context in contexts:
            if strategy == "summary":
                prompt = f"Summarize the key ideas and next logical steps for: {context[:500]}...\nSummary:"
            elif strategy == "cot":
                prompt = f"Think step by step about what comes next in: {context[:500]}...\nReasoning:"
            elif strategy == "hint":
                prompt = f"What hints would help understand this text: {context[:500]}...\nHints:"
            elif strategy == "mixed":
                # Randomly choose strategy for each context individually
                strategies = ["summary", "cot", "hint"]
                chosen = random.choice(strategies)
                if chosen == "summary":
                    prompt = f"Summarize the key ideas and next logical steps for: {context[:500]}...\nSummary:"
                elif chosen == "cot":
                    prompt = f"Think step by step about what comes next in: {context[:500]}...\nReasoning:"
                else:  # hint
                    prompt = f"What hints would help understand this text: {context[:500]}...\nHints:"
            else:
                prompt = f"Analyze: {context[:500]}...\nAnalysis:"
                
            prompts.append(prompt)
        
        return prompts
    
    def generate_latents(
        self, 
        input_ids: torch.Tensor, 
        strategy: str = "mixed",
        use_dummy: float = 0.1  # Probability of using dummy latent
    ) -> torch.Tensor:
        """
        Generate reasoning latents from input context
        
        Args:
            input_ids: [batch, seq_len] - LLaDA input tokens
            strategy: Type of reasoning prompt to use
            use_dummy: Probability of returning zero latent (for no-hint scenarios)
        
        Returns:
            latents: [batch, latent_dim] reasoning latents
        """
        batch_size = input_ids.shape[0]
        print(f"🔍 DEBUG - generate_latents called with input_ids dtype: {input_ids.dtype}, shape: {input_ids.shape}")
        
        # Sometimes use dummy latents (zero vector) to handle no-hint scenarios
        if random.random() < use_dummy:
            # Create dummy latents directly in float32 to avoid gradient-breaking dtype conversion
            dummy_latents = torch.zeros(batch_size, self.latent_dim, device=self.device, dtype=torch.float32, requires_grad=True)
            print(f"🔍 DEBUG - Using dummy latents with dtype: {dummy_latents.dtype}")
            return dummy_latents
        
        # Convert LLaDA tokens to text
        contexts = []
        for i in range(batch_size):
            # Take first 70% of context as input for reasoning
            context_length = int(input_ids.shape[1] * 0.7)
            context_tokens = input_ids[i, :context_length]
            
            # Remove padding and special tokens
            context_tokens = context_tokens[context_tokens != self.llada_tokenizer.pad_token_id]
            context_text = self.llada_tokenizer.decode(context_tokens, skip_special_tokens=True)
            contexts.append(context_text)
        
        # Generate reasoning prompts
        reasoning_prompts = self.generate_reasoning_prompts(contexts, strategy)
        print(f"🔍 DEBUG - Generated {len(reasoning_prompts)} reasoning prompts for {len(contexts)} contexts")
        
        # Tokenize for thinker (move to thinker device)
        thinker_inputs = self.thinker_tokenizer(
            reasoning_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.thinker_device)
        print(f"🔍 DEBUG - Thinker input_ids shape: {thinker_inputs.input_ids.shape}")
        
        # Generate reasoning with thinker (on thinker device)
        with torch.no_grad():
            thinker_outputs = self.thinker_model(**thinker_inputs, output_hidden_states=True)
            # Get hidden states from last layer
            last_hidden_states = thinker_outputs.hidden_states[-1]
            
            # DEBUG: Use proper generation instead of broken argmax
            print(f"🔍 DEBUG - Testing thinker generation...")
            debug_outputs = self.thinker_model.generate(
                **thinker_inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.thinker_tokenizer.eos_token_id
            )
            thinker_decoded = self.thinker_tokenizer.batch_decode(debug_outputs, skip_special_tokens=True)
            print(f"🔍 DEBUG - Thinker generated output:")
            for i, (prompt, output) in enumerate(zip(reasoning_prompts[:2], thinker_decoded[:2])):
                print(f"   Input {i+1}: {prompt[:100]}...")
                print(f"   Output {i+1}: {output}")
                print("-" * 40)
        # print(f"🔍 DEBUG - Thinker output dtype: {last_hidden_states.dtype}, device: {last_hidden_states.device}")
        # print(f"🔍 DEBUG - Thinker output shape: {last_hidden_states.shape}")
        
        # Transfer thinker output to main device for latent encoder
        if last_hidden_states.device != self.device:
            print(f"🔄 Transferring thinker output: {last_hidden_states.device} → {self.device}")
            last_hidden_states = last_hidden_states.to(self.device)
        
        # print(f"🔍 DEBUG - About to pass to latent encoder on {self.device}...")
        
        # Encode to latent space (on main device)
        try:
            latents = self.latent_encoder(last_hidden_states)  # [batch, latent_dim]
            print(f"🔍 DEBUG - Latent encoder output dtype: {latents.dtype}, device: {latents.device}")
            print(f"🔍 DEBUG - Latent encoder output shape: {latents.shape}")
        except Exception as e:
            # print(f"❌ ERROR in latent encoder: {e}")
            # print(f"🔍 DEBUG - Input device: {last_hidden_states.device}, encoder device: {next(self.latent_encoder.parameters()).device}")
            # print(f"🔍 DEBUG - First few encoder layer dtypes:")
            for i, layer in enumerate(self.latent_encoder.encoder[:3]):
                if hasattr(layer, 'weight'):
                    print(f"   Layer {i} ({type(layer).__name__}): {layer.weight.dtype}, device: {layer.weight.device}")
            raise e
        
        # Convert to float32 for compatibility with main model 
        # (latents are already on main device, FiLM adapters will auto-convert to match model dtype)
        # Use dtype conversion that preserves gradients
        if latents.dtype != torch.float32:
            latents = latents.float()  # .float() preserves gradients unlike .to(torch.float32)
        print(f"🔍 DEBUG - Final output dtype: {latents.dtype}, device: {latents.device}")
        return latents

def forward_process(input_ids, eps=1e-3):
    """Your existing forward process function"""
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

class MidTrainingPipeline:
    """
    Complete mid-training pipeline with latent injection
    """
    def __init__(
        self, 
        llada_path: str = "./llada-8b",
        thinker_path: str = "./thinker", 
        latent_dim: int = 256,
        device: str = 'cuda',
        thinker_device: Optional[str] = None
    ):
        self.device = device
        self.latent_dim = latent_dim
        
        # Load FiLM-enabled LLaDA model on main device
        print("Loading LLaDA model with FiLM adapters...")
        self.llada_model = load_film_model(llada_path, latent_dim, device)
        
        # Initialize reasoning latent generator (potentially on separate device)
        print("Setting up reasoning latent generator...")
        self.latent_generator = ReasoningLatentGenerator(thinker_path, latent_dim, device, thinker_device)
        
        # Setup optimizer (only train FiLM adapters and latent encoder)
        self.setup_optimizer()
        
    def setup_optimizer(self, lr: float = 1e-4):
        """Setup optimizer for only trainable parameters"""
        # Freeze base LLaDA model
        for name, param in self.llada_model.named_parameters():
            if 'film_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Make latent encoder trainable
        for param in self.latent_generator.latent_encoder.parameters():
            param.requires_grad = True
        
        # Collect trainable parameters
        trainable_params = [
            p for p in self.llada_model.parameters() if p.requires_grad
        ] + [
            p for p in self.latent_generator.latent_encoder.parameters()
        ]
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.llada_model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({100*trainable_params_count/total_params:.2f}%)")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with latent conditioning
        """
        self.llada_model.train()
        self.latent_generator.latent_encoder.train()
        
        # Get input data
        input_ids = batch["input_ids"].to(self.device)  # [batch, seq_len]
        
        # Handle variable length sequences (your existing logic)
        if torch.rand(1) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            input_ids = input_ids[:, :random_length]
        
        # Generate reasoning latents from context
        # IMPORTANT: Remove no_grad to allow gradients to flow back to latent encoder
        reasoning_latents = self.latent_generator.generate_latents(
            input_ids, 
            strategy="mixed",
            use_dummy=0.1  # 10% chance of no-hint scenario
        )
        
        # Apply diffusion forward process (your existing masking)
        noisy_batch, masked_indices, p_mask = forward_process(input_ids)
        
        # Forward pass with latent conditioning
        outputs = self.llada_model(
            input_ids=noisy_batch,
            latent=reasoning_latents  # This is the key addition!
        )
        
        logits = outputs.logits
        
        # Calculate loss (your existing loss computation)
        token_loss = F.cross_entropy(
            logits[masked_indices], 
            input_ids[masked_indices], 
            reduction='none'
        ) / p_mask[masked_indices]
        
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.llada_model.parameters() if p.requires_grad] + 
            [p for p in self.latent_generator.latent_encoder.parameters()], 
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "masked_tokens": masked_indices.sum().item(),
            "total_tokens": input_ids.numel()
        }
    
    def save_checkpoint(self, save_path: str, step: int):
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'llada_model_state_dict': self.llada_model.state_dict(),
            'latent_encoder_state_dict': self.latent_generator.latent_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.latent_dim,
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.llada_model.load_state_dict(checkpoint['llada_model_state_dict'])
        self.latent_generator.latent_encoder.load_state_dict(checkpoint['latent_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']

# Training script
def run_mid_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize pipeline
    pipeline = MidTrainingPipeline(
        llada_path="/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b",
        thinker_path="/data/sls/u/urop/mvideet/diffusion_reasoning/thinker",
        latent_dim=256,
        device=device
    )
    
    # Example training loop
    print("Starting mid-training with latent injection...")
    
    for step in range(10000):  # Your 180B token training
        # Your data loading logic here
        # For example:
        batch = {
            "input_ids": torch.randint(0, 50000, (4, 2048)).to(device)  # Example batch
        }
        
        # Training step
        metrics = pipeline.training_step(batch)
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {metrics['loss']:.4f}, "
                  f"Masked tokens = {metrics['masked_tokens']}, "
                  f"Total tokens = {metrics['total_tokens']}")
        
        if step % 1000 == 0 and step > 0:
            pipeline.save_checkpoint(f"checkpoint_step_{step}.pt", step)
    
    print("Mid-training completed!")

if __name__ == "__main__":
    run_mid_training()