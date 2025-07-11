import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import re
from collections import defaultdict
"""
This file, `mutual_information_analysis.py`, provides utility functions and tools for analyzing the mutual information and answer confidence in the context of language model outputs. It includes methods for:

- Extracting answer tokens from generated text using a tokenizer and expected answer string.
- Computing confidence scores for model-generated answers based on output logits and answer token IDs.
- (Presumably) additional functions for mutual information estimation and analysis, such as using nearest neighbors or distance metrics, as suggested by the imports.

The file is intended to support research or evaluation workflows where understanding the relationship between model predictions, answer correctness, and information-theoretic measures is important.
"""

def extract_answer_tokens(tokenizer, generated_text, expected_answer):
    """
    Extract tokens that correspond to the expected answer from generated text.
    Returns token indices that match the expected answer.
    """
    # Clean and normalize both texts
    expected_clean = re.sub(r'[^\w\s.]', '', str(expected_answer).lower().strip())
    generated_clean = re.sub(r'[^\w\s.]', '', generated_text.lower().strip())
    
    # Try to find the expected answer in the generated text
    answer_tokens = []
    if expected_clean in generated_clean:
        # Tokenize the expected answer
        expected_tokens = tokenizer(expected_clean, add_special_tokens=False)['input_ids']
        answer_tokens = expected_tokens
    else:
        # If exact match fails, try to find numeric answer
        expected_nums = re.findall(r'\d+\.?\d*', expected_clean)
        generated_nums = re.findall(r'\d+\.?\d*', generated_clean)
        
        if expected_nums and any(num in generated_nums for num in expected_nums):
            # Find the matching number and tokenize it
            for num in expected_nums:
                if num in generated_nums:
                    num_tokens = tokenizer(num, add_special_tokens=False)['input_ids']
                    answer_tokens.extend(num_tokens)
                    break
    
    return answer_tokens

def compute_answer_confidence(logits, answer_token_ids, method='softmax_sum'):
    """
    Compute confidence score for the correct answer tokens.
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        answer_token_ids: List of token IDs that comprise the correct answer
        method: Method for computing confidence ('softmax_sum', 'max_prob', 'mean_prob')
    
    Returns:
        Confidence score (higher = more confident in correct answer)
    """
    if not answer_token_ids:
        return 0.0
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
    
    # Get probabilities for answer tokens across all positions
    answer_probs = []
    for token_id in answer_token_ids:
        token_probs = probs[:, :, token_id]  # (batch, seq_len)
        answer_probs.append(token_probs)
    
    # Aggregate probabilities
    if method == 'softmax_sum':
        # Sum probabilities for all answer tokens across all positions
        total_prob = sum(torch.sum(prob) for prob in answer_probs)
        return total_prob.item()
    elif method == 'max_prob':
        # Maximum probability of any answer token at any position
        max_prob = max(torch.max(prob) for prob in answer_probs)
        return max_prob.item()
    elif method == 'mean_prob':
        # Mean probability of answer tokens
        all_probs = torch.cat([prob.flatten() for prob in answer_probs])
        return torch.mean(all_probs).item()
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_cross_entropy_confidence(logits, answer_token_ids, target_positions=None):
    """
    Compute negative cross-entropy as a confidence measure.
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        answer_token_ids: List of token IDs for the correct answer
        target_positions: Positions where we expect the answer tokens (optional)
    
    Returns:
        Negative cross-entropy (higher = more confident)
    """
    if not answer_token_ids:
        return 0.0
    
    # If no target positions specified, use the last few positions
    if target_positions is None:
        seq_len = logits.shape[1]
        target_positions = list(range(max(0, seq_len - len(answer_token_ids)), seq_len))
    
    # Ensure we don't exceed sequence length
    target_positions = [pos for pos in target_positions if pos < logits.shape[1]]
    
    if not target_positions:
        return 0.0
    
    # Compute cross-entropy for each target position
    cross_entropies = []
    for i, pos in enumerate(target_positions):
        if i < len(answer_token_ids):
            target_token = answer_token_ids[i]
            logits_at_pos = logits[:, pos, :]  # (batch, vocab_size)
            ce = F.cross_entropy(logits_at_pos, torch.tensor([target_token], device=logits.device))
            cross_entropies.append(-ce.item())  # Negative for "confidence"
    
    return np.mean(cross_entropies) if cross_entropies else 0.0

def estimate_mutual_information_knn(X, Y, k=3):
    """
    Estimate mutual information using k-NN method.
    
    Args:
        X: Features (e.g., logit centroids), shape (n_samples, n_features)
        Y: Target labels, shape (n_samples,)
        k: Number of nearest neighbors
    
    Returns:
        MI estimate in bits
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
        # Use sklearn's implementation as it's more robust
        mi = mutual_info_regression(X, Y, discrete_features=False, random_state=42)
        return np.mean(mi)
    except ImportError:
        # Fallback to simpler implementation
        n_samples = X.shape[0]
        if n_samples < k + 1:
            return 0.0
        
        # Simple MI estimation using nearest neighbors
        # This is a basic approximation - for real research, use more sophisticated methods
        unique_y = np.unique(Y)
        if len(unique_y) <= 1:
            return 0.0
        
        # Approximate MI using variance in Y given X neighborhoods
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Compute local variance in Y
        total_var = np.var(Y)
        local_vars = []
        for i in range(n_samples):
            neighbor_y = Y[indices[i][1:]]  # Exclude self
            local_vars.append(np.var(neighbor_y))
        
        avg_local_var = np.mean(local_vars)
        
        # MI approximation: I(X;Y) ≈ 0.5 * log(total_var / local_var)
        if avg_local_var > 0:
            mi_estimate = 0.5 * np.log2(total_var / avg_local_var)
            return max(0, mi_estimate)
        else:
            return 0.0

def compute_logit_centroids(logits_list):
    """
    Compute 2D centroids for each diffusion step using t-SNE or PCA.
    
    Args:
        logits_list: List of logit tensors for each step
    
    Returns:
        List of 2D centroids for each step
    """
    try:
        from sklearn.decomposition import PCA
        
        centroids = []
        for logits in logits_list:
            # Flatten logits: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            logits_flat = logits.reshape(-1, logits.size(-1))
            
            # Convert to probabilities
            probs = F.softmax(logits_flat, dim=-1)
            
            # Compute mean probability distribution (centroid in probability space)
            centroid_prob = torch.mean(probs, dim=0)  # (vocab_size,)
            
            # Reduce to 2D using PCA
            pca = PCA(n_components=2)
            centroid_2d = pca.fit_transform(centroid_prob.cpu().numpy().reshape(1, -1))
            centroids.append(centroid_2d[0])
        
        return np.array(centroids)
    except ImportError:
        print("Warning: sklearn not available. Using simple mean as centroids.")
        # Fallback: use simple statistics
        centroids = []
        for logits in logits_list:
            # Simple centroid: mean and std of logits
            mean_logit = torch.mean(logits).item()
            std_logit = torch.std(logits).item()
            centroids.append([mean_logit, std_logit])
        return np.array(centroids)

def analyze_information_gain(logits_list, results_list, tokenizer, method='softmax_sum'):
    """
    Analyze information gain about the correct answer across diffusion steps.
    
    Args:
        logits_list: List of logit tensors for each step
        results_list: List of dictionaries with problem information
        tokenizer: Tokenizer for extracting answer tokens
        method: Method for computing confidence
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing information gain using method: {method}")
    
    # Extract answer tokens for each problem
    answer_tokens_list = []
    for result in results_list:
        answer_tokens = extract_answer_tokens(
            tokenizer, 
            result['generated_answer'], 
            result['expected_answer']
        )
        answer_tokens_list.append(answer_tokens)
    
    # Compute confidence trajectory for each problem
    confidence_trajectories = []
    for prob_idx, problem_logits in enumerate(logits_list):
        if prob_idx >= len(answer_tokens_list):
            break
            
        answer_tokens = answer_tokens_list[prob_idx]
        
        confidence_trajectory = []
        for step_logits in problem_logits:
            confidence = compute_answer_confidence(step_logits, answer_tokens, method)
            confidence_trajectory.append(confidence)
        
        confidence_trajectories.append(confidence_trajectory)
    
    # Average confidence across problems
    if confidence_trajectories:
        min_steps = min(len(traj) for traj in confidence_trajectories)
        avg_confidence = []
        for step in range(min_steps):
            step_confidences = [traj[step] for traj in confidence_trajectories if step < len(traj)]
            avg_confidence.append(np.mean(step_confidences))
    else:
        avg_confidence = []
    
    # Compute logit centroids for MI estimation
    if logits_list and len(logits_list) > 0:
        # Use averaged logits for centroid computation
        centroids = compute_logit_centroids(logits_list[0])  # Use first problem's logits
        
        # Create dummy target variable (answer correctness)
        target_correctness = []
        for result in results_list:
            # Simple correctness check (contains expected answer)
            expected = str(result['expected_answer']).lower().strip()
            generated = result['generated_answer'].lower().strip()
            correct = expected in generated or any(word in generated for word in expected.split())
            target_correctness.append(int(correct))
        
        # Estimate MI between centroids and answer correctness
        mi_trajectory = []
        if len(centroids) > 1 and len(target_correctness) > 0:
            # For each step, estimate MI between that step's representation and correctness
            for step in range(len(centroids)):
                # Use current centroid as feature
                X = centroids[step:step+1]  # Single sample
                Y = np.array(target_correctness[:1])  # Single target
                
                # Simple MI approximation based on confidence
                if step < len(avg_confidence):
                    # Use confidence as proxy for MI
                    mi_estimate = avg_confidence[step] * np.log2(len(set(target_correctness)) + 1)
                else:
                    mi_estimate = 0.0
                
                mi_trajectory.append(mi_estimate)
        else:
            mi_trajectory = avg_confidence.copy()
    else:
        centroids = []
        mi_trajectory = []
    
    return {
        'confidence_trajectories': confidence_trajectories,
        'avg_confidence': avg_confidence,
        'mi_trajectory': mi_trajectory,
        'centroids': centroids,
        'answer_tokens_list': answer_tokens_list,
        'method': method
    }

def plot_information_gain_analysis(analysis_results, save_path='information_gain_analysis.png'):
    """
    Plot the information gain analysis results.
    
    Args:
        analysis_results: Dictionary from analyze_information_gain
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average confidence trajectory
    ax1 = axes[0, 0]
    steps = range(len(analysis_results['avg_confidence']))
    ax1.plot(steps, analysis_results['avg_confidence'], 'b-', linewidth=2, label='Average Confidence')
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Answer Confidence')
    ax1.set_title('Average Confidence in Correct Answer')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Individual confidence trajectories
    ax2 = axes[0, 1]
    for i, traj in enumerate(analysis_results['confidence_trajectories'][:5]):  # Show first 5
        ax2.plot(range(len(traj)), traj, alpha=0.6, label=f'Problem {i+1}')
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Answer Confidence')
    ax2.set_title('Individual Problem Confidence Trajectories')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Mutual information trajectory
    ax3 = axes[1, 0]
    if analysis_results['mi_trajectory']:
        ax3.plot(range(len(analysis_results['mi_trajectory'])), 
                analysis_results['mi_trajectory'], 'r-', linewidth=2, label='MI Estimate')
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Mutual Information (bits)')
        ax3.set_title('Estimated I(State; Correct Answer)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'MI trajectory not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Mutual Information Trajectory')
    
    # Plot 4: Information gain (derivative of confidence)
    ax4 = axes[1, 1]
    if len(analysis_results['avg_confidence']) > 1:
        info_gain = np.diff(analysis_results['avg_confidence'])
        ax4.plot(range(len(info_gain)), info_gain, 'g-', linewidth=2, label='Information Gain')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Diffusion Step')
        ax4.set_ylabel('Change in Confidence')
        ax4.set_title('Information Gain per Step')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Not enough data for info gain', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Information Gain per Step')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Information gain analysis saved to {save_path}")

def analyze_information_spikes(analysis_results, spike_threshold=0.1):
    """
    Identify and analyze information spikes in the confidence trajectory.
    
    Args:
        analysis_results: Dictionary from analyze_information_gain
        spike_threshold: Minimum change to consider a spike
    
    Returns:
        Dictionary with spike analysis
    """
    avg_confidence = analysis_results['avg_confidence']
    
    if len(avg_confidence) < 2:
        return {'spikes': [], 'total_gain': 0, 'spike_steps': []}
    
    # Calculate information gain per step
    info_gain = np.diff(avg_confidence)
    
    # Find spikes (large positive changes)
    spikes = []
    spike_steps = []
    
    for i, gain in enumerate(info_gain):
        if gain > spike_threshold:
            spikes.append({
                'step': i,
                'gain': gain,
                'confidence_before': avg_confidence[i],
                'confidence_after': avg_confidence[i + 1]
            })
            spike_steps.append(i)
    
    # Calculate total information gain
    total_gain = avg_confidence[-1] - avg_confidence[0] if avg_confidence else 0
    
    # Find the largest spike
    largest_spike = max(spikes, key=lambda x: x['gain']) if spikes else None
    
    return {
        'spikes': spikes,
        'spike_steps': spike_steps,
        'total_gain': total_gain,
        'largest_spike': largest_spike,
        'num_spikes': len(spikes),
        'avg_spike_gain': np.mean([s['gain'] for s in spikes]) if spikes else 0
    }

def main():
    """
    Main function to run the mutual information analysis.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the saved data
    try:
        print("Loading saved data...")
        logits_list = torch.load('all_problems_logits.pt', map_location=device, weights_only=False)
        results_list = torch.load('inference_results.pt', map_location=device, weights_only=False)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/data/sls/u/urop/mvideet/diffusion_reasoning/models/llada_8b", trust_remote_code=True)
        
        print(f"Loaded {len(logits_list)} problems with {len(results_list)} results")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run load_diffusion.py first to generate the required data files.")
        return
    
    # Run analysis with different methods
    methods = ['softmax_sum', 'max_prob', 'mean_prob']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Analyzing with method: {method}")
        print(f"{'='*60}")
        
        # Perform analysis
        analysis_results = analyze_information_gain(
            logits_list, results_list, tokenizer, method=method
        )
        
        # Plot results
        plot_information_gain_analysis(
            analysis_results, 
            save_path=f'information_gain_analysis_{method}.png'
        )
        
        # Analyze spikes
        spike_analysis = analyze_information_spikes(analysis_results)
        
        print(f"\nSPIKE ANALYSIS ({method}):")
        print(f"Total information gain: {spike_analysis['total_gain']:.4f}")
        print(f"Number of spikes: {spike_analysis['num_spikes']}")
        print(f"Average spike gain: {spike_analysis['avg_spike_gain']:.4f}")
        
        if spike_analysis['largest_spike']:
            largest = spike_analysis['largest_spike']
            print(f"Largest spike at step {largest['step']}: {largest['gain']:.4f}")
        
        print(f"Spike steps: {spike_analysis['spike_steps']}")
        
        # Save analysis results
        torch.save(analysis_results, f'information_gain_results_{method}.pt')
        torch.save(spike_analysis, f'spike_analysis_{method}.pt')
        
        print(f"Results saved to information_gain_results_{method}.pt")

if __name__ == '__main__':
    main() 