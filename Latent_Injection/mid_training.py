# mid_training_with_latents.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
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
    def __init__(self, thinker_path: str = "./thinker", latent_dim: int = 256, device: str = 'cuda'):
        self.device = device
        self.latent_dim = latent_dim
        
        # Load thinker model and tokenizer
        print("Loading thinker model...")
        self.thinker_model = AutoModel.from_pretrained(
            thinker_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device).eval()
        
        self.thinker_tokenizer = AutoTokenizer.from_pretrained(thinker_path)
        if self.thinker_tokenizer.pad_token is None:
            self.thinker_tokenizer.pad_token = self.thinker_tokenizer.eos_token
            
        # Initialize latent encoder
        self.latent_encoder = LatentEncoder(
            thinker_hidden_size=2560,  # Phi-2 hidden size
            latent_dim=latent_dim
        ).to(device)
        
        # Load LLaDA tokenizer for context preparation
        self.llada_tokenizer = AutoTokenizer.from_pretrained("./llada-8b", trust_remote_code=True)
        
        # Define reasoning styles and frames
        self.thinker_styles = {
            "summary": (
                "Briefly summarise the big picture and the very next operation "
                "the writer should perform."
            ),
            "cot": (
                "Think step-by-step, writing each sub-goal as an imperative "
                "clause separated by semicolons."
            ),
            "hint": (
                "Write 1-3 short hints that would unblock a stuck student."
            ),
        }

        # Extra "frames" keep the surface form diverse while the substance is similar.
        self.rhetorical_frames = [
            "As an expert problem-solver:",
            "Internal planning note:",
            "Self-checklist:",
        ]

    def generate_reasoning_prompts(
            self,
            contexts: List[str],
            *,
            strategy: str = "mixed",
            cycle_idx: Optional[int] = None,
            total_cycles: Optional[int] = None,
            max_context_chars: int = 400,
    ) -> List[str]:
        """
        Build prompts for the thinker LLM that will later be compressed into latents.

        Args
        ----
        contexts:          List of full context strings (prompt + current partial answer).
        strategy:          'summary' | 'cot' | 'hint' | 'mixed'
        cycle_idx:         Which think→write cycle we’re in (0-based).  Optional.
        total_cycles:      Total number of cycles planned.              Optional.
        max_context_chars: Truncate context so the thinker sees only the
                        recent portion (prevents leaking solutions).

        Returns
        -------
        List[str] – same length as `contexts`.
        """

        prompts = []
        for ctx in contexts:
            # 1) choose strategy
            chosen = (
                strategy
                if strategy != "mixed"
                else random.choice(list(self.thinker_styles.keys()))
            )
            guidance = self.thinker_styles[chosen]

            # 2) choose rhetorical frame
            frame = random.choice(self.rhetorical_frames)

            # 3) optional cycle annotation
            if cycle_idx is not None and total_cycles is not None:
                cycle_tag = f"(cycle {cycle_idx + 1}/{total_cycles})"
            else:
                cycle_tag = ""

            # 4) truncate context – include both prompt head and answer tail
            head = ctx[: max_context_chars // 2]
            tail = ctx[-max_context_chars // 2 :]
            trimmed_ctx = f"{head} ... {tail}" if len(ctx) > max_context_chars else ctx

            # 5) assemble final prompt
            prompt = (
                f"<PROMPT_BEGIN>{frame} {cycle_tag}\n\n"
                f"<CONTEXT>\n{trimmed_ctx}\n</CONTEXT>\n\n"
                f"<GUIDE>\n{guidance}\n\n"
                f"• Output ≤100 tokens\n"
                f"• Use comma-separated phrases, no full stops\n"
                f"<REASONING>"
            )
            prompts.append(prompt)

        return prompts
    
    def generate_latents(
        self, 
        input_ids: torch.Tensor, 
        strategy: str = "mixed",
        use_dummy: float = 0.1  # Dropout probability for using a dummy (zero) latent
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
        
        # Sometimes use dummy latents (zero vector) to handle no-hint scenarios
        if random.random() < use_dummy:
            return torch.zeros(batch_size, self.latent_dim, device=self.device)
        
        # Convert LLaDA tokens to text
        contexts = []
        for i in range(batch_size):
            # Use the full context as input for reasoning.
            # Originally, only the first 70% of the context was used, possibly to focus the reasoning on the initial part of the input
            # and avoid including answer tokens or trailing irrelevant information. However, using the full context may provide richer information
            # for the reasoning latent, especially if the context is already trimmed or well-formed.
            context_tokens = input_ids[i]
            
            # Remove padding and special tokens
            context_tokens = context_tokens[context_tokens != self.llada_tokenizer.pad_token_id]
            context_text = self.llada_tokenizer.decode(context_tokens, skip_special_tokens=True)
            contexts.append(context_text)
        # Generate reasoning prompts
        reasoning_prompts = self.generate_reasoning_prompts(contexts, strategy=strategy)
        
        # This section generates reasoning latents from the input contexts using a "thinker" model.
        # 1. First, we tokenize the generated reasoning prompts using the thinker's tokenizer.
        #    - We pad and truncate the prompts to a maximum length of 512 tokens.
        #    - The resulting tokenized batch is moved to the target device (e.g., GPU).
        thinker_inputs = self.thinker_tokenizer(
            reasoning_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400
        ).to(self.device)
        
        # 2. Next, we use the thinker model to process these tokenized prompts.
        #    - We disable gradient computation (no training here).
        #    - The model returns hidden states for each layer; we take the last layer's hidden states.
        #    - Shape: [batch, seq_len, hidden_size]
        with torch.no_grad():
            thinker_outputs = self.thinker_model(**thinker_inputs, output_hidden_states=True)
            last_hidden_states = thinker_outputs.hidden_states[-1]
        
        # 3. We then encode these last hidden states into a fixed-size latent vector for each input.
        #    - The latent encoder projects the sequence of hidden states into a single [batch, latent_dim] vector.
        latents = self.latent_encoder(last_hidden_states)
        
        # 4. Return the resulting latents, which will be used as conditioning for the main model.
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
        device: str = 'cuda'
    ):
        self.device = device
        self.latent_dim = latent_dim
        
        # Load FiLM-enabled LLaDA model
        print("Loading LLaDA model with FiLM adapters...")
        self.llada_model = load_film_model(llada_path, latent_dim, device)
        
        # Initialize reasoning latent generator
        print("Setting up reasoning latent generator...")
        self.latent_generator = ReasoningLatentGenerator(thinker_path, latent_dim, device)
        
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
        # This sets the LLaDA model and the latent encoder to training mode,
        # enabling features like dropout and layer norm updates during training.
        self.llada_model.train()
        self.latent_generator.latent_encoder.train()
        
        # Get input data
        input_ids = batch["input_ids"].to(self.device)  # [batch, seq_len]
        
        # Handle variable length sequences (your existing logic)
        if torch.rand(1) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            input_ids = input_ids[:, :random_length]
        
        # Generate reasoning latents from context
        with torch.no_grad():
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
        llada_path="./llada-8b",
        thinker_path="./thinker",
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