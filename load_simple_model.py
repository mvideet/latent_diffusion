import torch
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

        # Load the LLaDA model and tokenizer from local directory
print("Loading LLaDA model...")
model = AutoModel.from_pretrained(
            "./llada-8b",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
print(model)
