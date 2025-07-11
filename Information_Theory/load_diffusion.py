# Import required libraries for tensor operations and neural network functions
import torch
import numpy as np
import torch.nn.functional as F

# Import transformers for model and tokenizer functionality
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel

# Import safetensors for memory-efficient tensor saving
from safetensors.torch import save_file, load_file


def load_averaged_logits(filename="averaged_logit_trajectory.safetensors"):
    """
    Load averaged logits from safetensors format back to list of tensors
    """
    logits_dict = load_file(filename)
    # Sort by step number to maintain order
    sorted_keys = sorted(logits_dict.keys(), key=lambda x: int(x.split('_')[1]))
    return [logits_dict[key] for key in sorted_keys]


def load_all_problem_logits(filename="all_problems_logits.safetensors"):
    """
    Load all problem logits from safetensors format back to nested list structure
    """
    logits_dict = load_file(filename)
    
    # Group by problem and step
    problems_dict = {}
    for key, tensor in logits_dict.items():
        parts = key.split('_')
        problem_idx = int(parts[1])
        step_idx = int(parts[3])
        
        if problem_idx not in problems_dict:
            problems_dict[problem_idx] = {}
        problems_dict[problem_idx][step_idx] = tensor
    
    # Convert back to nested list structure
    all_logits_list = []
    for problem_idx in sorted(problems_dict.keys()):
        problem_logits = []
        for step_idx in sorted(problems_dict[problem_idx].keys()):
            problem_logits.append(problems_dict[problem_idx][step_idx])
        all_logits_list.append(problem_logits)
    
    return all_logits_list


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    # If temperature is 0, return original logits without noise
    if temperature == 0:
        return logits
    # Convert logits to float64 for better precision
    logits = logits.to(torch.float64)
    # Generate random noise with same shape as logits
    noise = torch.rand_like(logits, dtype=torch.float64)
    # Calculate Gumbel noise using the formula: (-log(noise))^temperature
    gumbel_noise = (- torch.log(noise)) ** temperature
    # Return modified logits with Gumbel noise
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    # Calculate total number of masked tokens per batch
    mask_num = mask_index.sum(dim=1, keepdim=True)

    # Calculate base number of tokens to transfer per step
    base = mask_num // steps
    # Calculate remainder tokens that need to be distributed
    remainder = mask_num % steps

    # Initialize tensor with base values for all steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    # Distribute remainder tokens to first few steps
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # Initialize output tensor with mask tokens, extending prompt length by gen_length
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    # Copy the original prompt to the beginning of the output tensor
    x[:, :prompt.shape[1]] = prompt.clone()
    all_logits = []
    # Create index mask for prompt tokens (non-mask tokens)
    prompt_index = (x != mask_id)

    # Ensure gen_length is divisible by block_length for block-wise processing
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # Ensure steps is divisible by num_blocks for even distribution
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Process each block sequentially
    for num_block in range(num_blocks):
        # Get mask index for current block only
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        # Calculate number of tokens to transfer at each step for this block
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        # Perform diffusion steps for current block
        for i in range(steps):
            # Get current mask positions
            mask_index = (x == mask_id)
            
            # Apply classifier-free guidance if cfg_scale > 0
            if cfg_scale > 0.:
                # Create unconditioned input by masking prompt tokens
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                # Concatenate conditioned and unconditioned inputs
                x_ = torch.cat([x, un_x], dim=0)
                # Get logits from model
                logits = model(x_).logits
                # Split logits into conditioned and unconditioned parts
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                # Apply classifier-free guidance formula
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # Get logits directly without guidance
                logits = model(x).logits
            all_logits.append(logits.detach().cpu())

            # Add Gumbel noise to logits for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # Get predicted tokens by taking argmax
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            # Calculate confidence scores based on remasking strategy
            if remasking == 'low_confidence':
                # Use softmax probabilities as confidence scores
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                # Use random values as confidence scores
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Set confidence to negative infinity for tokens beyond current block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # Only update masked positions with predicted tokens
            x0 = torch.where(mask_index, x0, x)
            # Set confidence to negative infinity for non-masked positions
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Select tokens to transfer based on confidence scores
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # Get top-k tokens with highest confidence
                k_value = int(num_transfer_tokens[j, i].item())
                _, select_index = torch.topk(confidence[j], k=k_value)
                transfer_index[j, select_index] = True
            # Update selected tokens in the output tensor
            x[transfer_index] = x0[transfer_index]

    return x, all_logits


def run_multiple_problems(model, tokenizer, device, num_problems=10):
    """
    Run inference on multiple sample math problems and aggregate logits.
    """
    print("Using 10 sample math problems...")
    # Use sample problems directly (no dataset loading)
    test_problems = [
        {"question": "A factory produces widgets at a rate of x per hour and increases production by 5 widgets per hour every hour for 8 hours. If they produced 480 widgets total in those 8 hours, what was the initial production rate x?", "answer": "45"},
        {"question": "In a bag of marbles, the ratio of red to blue marbles is 3:4. If 5 red marbles are removed and 10 blue marbles are added, the new ratio becomes 1:2. How many red marbles were originally in the bag?", "answer": "15"},
        {"question": "A regular octagon is inscribed in a circle of radius 8 units. What is the area of the octagon?", "answer": "193.9"},
        {"question": "The sum of three consecutive integers is 51 more than twice the smallest of the three integers. What is the largest of the three integers?", "answer": "35"},
        {"question": "A train travels from city A to city B at 80 km/h and returns at 60 km/h. If the total journey takes 7 hours, what is the distance between cities A and B in kilometers?", "answer": "240"},
        {"question": "In a geometric sequence, the sum of the first three terms is 26 and the product is 216. What is the second term of the sequence?", "answer": "6"},
        {"question": "A box contains red, blue and green balls. The probability of drawing a red ball is 1/3, and the probability of drawing a blue ball is 1/4. If there are 24 balls in total, how many green balls are there?", "answer": "10"},
        {"question": "The angles of a triangle are in arithmetic progression. The smallest angle is 20° less than the middle angle. What is the largest angle in degrees?", "answer": "80"},
        {"question": "A rectangle has perimeter 30 units. If its area is 56 square units, what is the length of its diagonal?", "answer": "13"},
        {"question": "If log_2(x) + log_2(x+3) = 5, what is the value of x?", "answer": "5"}
    ]
    
    # Limit to requested number of problems
    num_problems = min(num_problems, len(test_problems))
    problems_to_test = test_problems[:num_problems]
    
    print(f"\n{'='*60}")
    print(f"Running inference on {num_problems} problems...")
    print(f"{'='*60}")
    
    all_aggregated_logits = []
    all_results = []
    
    for i, problem in enumerate(problems_to_test):
        if isinstance(problem, dict) and "question" in problem:
            question = problem["question"]
            expected_answer = problem.get("answer", "Unknown")
        else:
            question = str(problem)
            expected_answer = "Unknown"
            
        prompt = f"{question} Answer:"
        
        print(f"\nProblem {i+1}/{num_problems}:")
        print(f"Q: {question}")
        print(f"Expected: {expected_answer}")
        
        # Tokenize the prompt
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        # Generate response
            # Generate with remasking - this means tokens with low confidence will be
            # remasked and re-predicted during diffusion. This helps the model refine
            # uncertain predictions by giving it multiple chances to predict difficult tokens.
            # The 'low_confidence' strategy remasks tokens where the model's confidence 
            # falls below an adaptive threshold based on the average confidence.
        out, problem_logits = generate(
                model, input_ids, 
                steps=128, gen_length=256, block_length=16, 
                temperature=0.1, cfg_scale=0., remasking='low_confidence'
            )
            
            # Decode the generated response
        generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"Generated: {generated_text}")
            
            # Store the logits for this problem
        all_aggregated_logits.append(problem_logits)
            
            # Store results
        all_results.append({
                'problem_id': i+1,
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_text,
                'prompt_length': input_ids.shape[1]
            })
            
        
    return all_aggregated_logits, all_results

def average_logits_across_problems(all_logits_list):
    """
    Average logits across multiple problems at each diffusion step.
    Handles different sequence lengths by padding to the maximum length.
    """
    if not all_logits_list:
        return []
    
    # Find the minimum number of steps across all problems
    min_steps = min(len(problem_logits) for problem_logits in all_logits_list)
    print(f"Averaging logits across {len(all_logits_list)} problems with {min_steps} steps each...")
    
    # Find the maximum sequence length across all problems and steps
    max_seq_len = 0
    for problem_logits in all_logits_list:
        for step_logits in problem_logits[:min_steps]:
            max_seq_len = max(max_seq_len, step_logits.shape[1])
    
    print(f"Maximum sequence length found: {max_seq_len}")
    
    averaged_logits = []
    for step in range(min_steps):
        step_logits = []
        for problem_logits in all_logits_list:
            if step < len(problem_logits):
                logits = problem_logits[step]  # Shape: (batch, seq, vocab)
                
                # Pad sequence dimension to max_seq_len if needed
                if logits.shape[1] < max_seq_len:
                    # Create padding tensor with zeros
                    pad_size = max_seq_len - logits.shape[1]
                    padding = torch.zeros(logits.shape[0], pad_size, logits.shape[2], 
                                        dtype=logits.dtype, device=logits.device)
                    logits = torch.cat([logits, padding], dim=1)
                elif logits.shape[1] > max_seq_len:
                    # Truncate if longer than max (shouldn't happen but just in case)
                    logits = logits[:, :max_seq_len, :]
                
                step_logits.append(logits)
        
        if step_logits:
            # Now all tensors have the same shape, we can stack and average
            stacked_logits = torch.stack(step_logits, dim=0)  # (num_problems, batch, seq, vocab)
            averaged_step_logits = stacked_logits.mean(dim=0)  # (batch, seq, vocab)
            averaged_logits.append(averaged_step_logits)
    
    print(f"Averaged logits shape per step: {averaged_logits[0].shape if averaged_logits else 'N/A'}")
    return averaged_logits

def main():
    # Set device to CUDA for GPU acceleration (fallback to CPU if CUDA unavailable)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # Load the LLaDA model and tokenizer from local directory
        print("Loading LLaDA model...")
        model = AutoModel.from_pretrained(
            "/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/sls/u/urop/mvideet/diffusion_reasoning/llada_8b",
            trust_remote_code=True,
        )
        print("Model and tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference on multiple problems
    num_problems = 10  # Change this to test more/fewer problems
    all_logits_list, results = run_multiple_problems(model, tokenizer, device, num_problems)
    
    if not all_logits_list:
        print("No successful inferences. Exiting.")
        return
    
    # Average logits across all problems
    averaged_logits = average_logits_across_problems(all_logits_list)
    
    # Save the averaged logits using safetensors (memory efficient)
    print(f"\nSaving averaged logits from {len(all_logits_list)} problems...")
    # Convert list of tensors to dictionary for safetensors
    averaged_logits_dict = {f"step_{i}": tensor for i, tensor in enumerate(averaged_logits)}
    save_file(averaged_logits_dict, "averaged_logit_trajectory.safetensors")
    print(f"Saved averaged logits to 'averaged_logit_trajectory.safetensors'")
    
    # Also save individual problem logits using safetensors
    print(f"Saving individual problem logits...")
    # Convert nested list to flat dictionary
    all_logits_dict = {}
    for problem_idx, problem_logits in enumerate(all_logits_list):
        for step_idx, step_tensor in enumerate(problem_logits):
            all_logits_dict[f"problem_{problem_idx}_step_{step_idx}"] = step_tensor
    save_file(all_logits_dict, "all_problems_logits.safetensors")
    print(f"Saved all individual problem logits to 'all_problems_logits.safetensors'")
    
    # Save results summary (keep as torch.save since it's not tensors)
    torch.save(results, "inference_results.pt", pickle_protocol=4)
    print(f"Saved inference results to 'inference_results.pt'")
    
    # Print summary
    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems processed: {len(results)}")
    print(f"Average prompt length: {np.mean([r['prompt_length'] for r in results]):.1f} tokens")
    print(f"Diffusion steps per problem: {len(averaged_logits)}")
    print(f"Logits shape per step: {averaged_logits[0].shape if averaged_logits else 'N/A'}")
    
    print("\nFirst few problem summaries:")
    for i, result in enumerate(results[:3]):
        print(f"\nProblem {result['problem_id']}:")
        print(f"Q: {result['question'][:60]}...")
        print(f"Expected: {result['expected_answer']}")
        print(f"Generated: {result['generated_answer'][:60]}...")
        
    print(f"\nUse 'averaged_logit_trajectory.safetensors' for analysis with plot_logits.py")
    print(f"📋 To load saved data:")
    print(f"   averaged_logits = load_averaged_logits()")
    print(f"   all_logits = load_all_problem_logits()")
    print(f"   results = torch.load('inference_results.pt')")


if __name__ == '__main__':
    main()
