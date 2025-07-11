import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import re
import json
from collections import defaultdict

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

def compute_entropy(logits):
    """
    Compute entropy of the probability distribution from logits.
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
    
    Returns:
        Entropy value in bits
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy: H = -sum(p * log2(p))
    log_probs = F.log_softmax(logits, dim=-1)
    entropy_val = -torch.sum(probs * log_probs, dim=-1)  # (batch, seq_len)
    
    # Convert to bits and average
    entropy_bits = entropy_val / np.log(2)
    return torch.mean(entropy_bits).item()

def compute_conditional_entropy(logits, target_tokens, target_positions):
    """
    Compute conditional entropy H(Y|X) where Y is the target tokens and X is the context.
    
    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        target_tokens: List of target token IDs
        target_positions: Positions where target tokens should appear
    
    Returns:
        Conditional entropy in bits
    """
    if not target_tokens or not target_positions:
        return 0.0
    
    # Get probabilities for target tokens at target positions
    probs = F.softmax(logits, dim=-1)
    
    conditional_entropies = []
    for i, pos in enumerate(target_positions):
        if i < len(target_tokens) and pos < logits.shape[1]:
            target_token = target_tokens[i]
            # Get probability of target token at this position
            target_prob = probs[:, pos, target_token]
            
            # Conditional entropy contribution: -p * log2(p)
            if target_prob > 0:
                cond_entropy = -target_prob * torch.log2(target_prob + 1e-10)
                conditional_entropies.append(cond_entropy.item())
    
    return np.mean(conditional_entropies) if conditional_entropies else 0.0

def estimate_mi_with_binning(X, Y, n_bins=10):
    """
    Estimate mutual information using binning method.
    
    Args:
        X: Continuous features, shape (n_samples, n_features)
        Y: Discrete target labels, shape (n_samples,)
        n_bins: Number of bins for discretization
    
    Returns:
        MI estimate in bits
    """
    if len(X) != len(Y) or len(X) == 0:
        return 0.0
    
    # Discretize continuous features
    X_binned = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        bins = np.linspace(feature.min(), feature.max(), n_bins + 1)
        binned_feature = np.digitize(feature, bins) - 1
        binned_feature = np.clip(binned_feature, 0, n_bins - 1)
        X_binned.append(binned_feature)
    
    X_binned = np.column_stack(X_binned)
    
    # Convert to joint states
    X_states = [tuple(row) for row in X_binned]
    Y_states = Y.tolist()
    
    # Count joint and marginal frequencies
    joint_counts = defaultdict(int)
    x_counts = defaultdict(int)
    y_counts = defaultdict(int)
    
    for x_state, y_state in zip(X_states, Y_states):
        joint_counts[(x_state, y_state)] += 1
        x_counts[x_state] += 1
        y_counts[y_state] += 1
    
    n_samples = len(X)
    
    # Compute MI: I(X;Y) = sum_{x,y} P(x,y) * log2(P(x,y) / (P(x) * P(y)))
    mi = 0.0
    for (x_state, y_state), joint_count in joint_counts.items():
        if joint_count > 0:
            p_joint = joint_count / n_samples
            p_x = x_counts[x_state] / n_samples
            p_y = y_counts[y_state] / n_samples
            
            if p_x > 0 and p_y > 0:
                mi += p_joint * np.log2(p_joint / (p_x * p_y))
    
    return max(0, mi)

def estimate_mi_kraskov(X, Y, k=3):
    """
    Estimate mutual information using Kraskov-Stögbauer-Grassberger (KSG) estimator.
    This is a more sophisticated k-NN based estimator.
    
    Args:
        X: Features, shape (n_samples, n_features)
        Y: Target, shape (n_samples,)
        k: Number of nearest neighbors
    
    Returns:
        MI estimate in bits
    """
    n_samples = len(X)
    if n_samples < k + 1:
        return 0.0
    
    # Handle case where Y is constant
    if len(np.unique(Y)) <= 1:
        return 0.0
    
    # Convert Y to 2D if needed
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    # Combine X and Y
    XY = np.column_stack([X, Y])
    
    # Find k-nearest neighbors in joint space
    nbrs_joint = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(XY)
    distances_joint, indices_joint = nbrs_joint.kneighbors(XY)
    
    # For each point, find neighbors in marginal spaces
    mi_sum = 0.0
    for i in range(n_samples):
        # Distance to k-th nearest neighbor in joint space
        eps = distances_joint[i, k]
        
        # Count neighbors within eps distance in marginal spaces
        x_neighbors = np.sum(pdist(np.vstack([X[i], X]), metric='euclidean') < eps) - 1
        y_neighbors = np.sum(pdist(np.vstack([Y[i], Y]), metric='euclidean') < eps) - 1
        
        # Add to MI sum (with smoothing to avoid log(0))
        mi_sum += np.log(max(1, x_neighbors)) + np.log(max(1, y_neighbors))
    
    # KSG estimator formula
    psi_k = sum(1.0 / j for j in range(1, k+1))  # Digamma function approximation
    mi_estimate = psi_k - mi_sum / n_samples + np.log(n_samples)
    
    # Convert to bits
    return max(0, mi_estimate / np.log(2))

def advanced_information_analysis(logits_list, results_list, tokenizer, method='softmax_sum'):
    """
    Advanced information theoretic analysis of diffusion steps.
    
    Args:
        logits_list: List of logit tensors for each problem
        results_list: List of problem results
        tokenizer: Tokenizer for text processing
        method: Method for confidence computation
    
    Returns:
        Dictionary with comprehensive analysis results
    """
    print(f"Running advanced information analysis with method: {method}")
    
    # Extract answer information
    answer_info = []
    for result in results_list:
        expected = str(result['expected_answer']).lower().strip()
        generated = result['generated_answer'].lower().strip()
        
        # Extract answer tokens
        answer_tokens = []
        if expected in generated:
            answer_tokens = tokenizer(expected, add_special_tokens=False)['input_ids']
        else:
            # Try to find numeric matches
            expected_nums = re.findall(r'\d+\.?\d*', expected)
            generated_nums = re.findall(r'\d+\.?\d*', generated)
            if expected_nums and any(num in generated_nums for num in expected_nums):
                for num in expected_nums:
                    if num in generated_nums:
                        answer_tokens = tokenizer(num, add_special_tokens=False)['input_ids']
                        break
        
        # Determine correctness
        correct = expected in generated or any(word in generated for word in expected.split())
        
        answer_info.append({
            'tokens': answer_tokens,
            'correct': correct,
            'expected': expected,
            'generated': generated
        })
    
    # Analysis containers
    entropy_trajectory = []
    confidence_trajectory = []
    mi_trajectory_binning = []
    mi_trajectory_ksg = []
    conditional_entropy_trajectory = []
    
    # Process each diffusion step
    num_steps = len(logits_list[0]) if logits_list else 0
    
    for step in range(num_steps):
        print(f"Processing step {step+1}/{num_steps}")
        
        step_entropies = []
        step_confidences = []
        step_cond_entropies = []
        
        # Collect features for MI estimation
        features_for_mi = []
        labels_for_mi = []
        
        for prob_idx, problem_logits in enumerate(logits_list):
            if step < len(problem_logits) and prob_idx < len(answer_info):
                logits = problem_logits[step]
                info = answer_info[prob_idx]
                
                # Compute entropy
                entropy_val = compute_entropy(logits)
                step_entropies.append(entropy_val)
                
                # Compute confidence
                if info['tokens']:
                    confidence = compute_answer_confidence(logits, info['tokens'], method)
                    step_confidences.append(confidence)
                    
                    # Compute conditional entropy
                    target_positions = list(range(max(0, logits.shape[1] - len(info['tokens'])), logits.shape[1]))
                    cond_entropy = compute_conditional_entropy(logits, info['tokens'], target_positions)
                    step_cond_entropies.append(cond_entropy)
                else:
                    step_confidences.append(0.0)
                    step_cond_entropies.append(0.0)
                
                # Prepare features for MI estimation
                # Use simple statistics as features
                logits_flat = logits.reshape(-1, logits.size(-1))
                mean_logit = torch.mean(logits_flat).item()
                std_logit = torch.std(logits_flat).item()
                max_logit = torch.max(logits_flat).item()
                
                features_for_mi.append([mean_logit, std_logit, max_logit, entropy_val])
                labels_for_mi.append(int(info['correct']))
        
        # Aggregate step results
        entropy_trajectory.append(np.mean(step_entropies) if step_entropies else 0.0)
        confidence_trajectory.append(np.mean(step_confidences) if step_confidences else 0.0)
        conditional_entropy_trajectory.append(np.mean(step_cond_entropies) if step_cond_entropies else 0.0)
        
        # Estimate MI using different methods
        if len(features_for_mi) > 5:  # Need minimum samples for MI estimation
            X = np.array(features_for_mi)
            Y = np.array(labels_for_mi)
            
            # Binning-based MI
            mi_binning = estimate_mi_with_binning(X, Y)
            mi_trajectory_binning.append(mi_binning)
            
            # KSG-based MI
            mi_ksg = estimate_mi_kraskov(X, Y)
            mi_trajectory_ksg.append(mi_ksg)
        else:
            mi_trajectory_binning.append(0.0)
            mi_trajectory_ksg.append(0.0)
    
    # Compute information gain (derivative of confidence)
    info_gain = np.diff(confidence_trajectory) if len(confidence_trajectory) > 1 else []
    
    # Detect information spikes
    spikes = []
    if len(info_gain) > 0:
        threshold = np.std(info_gain) * 2  # Adaptive threshold
        for i, gain in enumerate(info_gain):
            if gain > threshold:
                spikes.append({
                    'step': i,
                    'gain': gain,
                    'confidence_before': confidence_trajectory[i],
                    'confidence_after': confidence_trajectory[i + 1]
                })
    
    return {
        'entropy_trajectory': entropy_trajectory,
        'confidence_trajectory': confidence_trajectory,
        'conditional_entropy_trajectory': conditional_entropy_trajectory,
        'mi_trajectory_binning': mi_trajectory_binning,
        'mi_trajectory_ksg': mi_trajectory_ksg,
        'info_gain': info_gain,
        'spikes': spikes,
        'answer_info': answer_info,
        'method': method
    }

def plot_comprehensive_analysis(analysis_results, save_path='comprehensive_mi_analysis.png'):
    """
    Create comprehensive plots for information theoretic analysis.
    
    Args:
        analysis_results: Dictionary from advanced_information_analysis
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot 1: Entropy trajectory
    ax1 = axes[0, 0]
    steps = range(len(analysis_results['entropy_trajectory']))
    ax1.plot(steps, analysis_results['entropy_trajectory'], 'b-', linewidth=2, label='Entropy')
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Entropy (bits)')
    ax1.set_title('Entropy Trajectory H(P_t)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Confidence trajectory
    ax2 = axes[0, 1]
    ax2.plot(steps, analysis_results['confidence_trajectory'], 'g-', linewidth=2, label='Confidence')
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Answer Confidence')
    ax2.set_title('Confidence in Correct Answer')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Mutual Information (Binning)
    ax3 = axes[1, 0]
    ax3.plot(steps, analysis_results['mi_trajectory_binning'], 'r-', linewidth=2, label='MI (Binning)')
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Mutual Information (bits)')
    ax3.set_title('I(State; Answer) - Binning Method')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Mutual Information (KSG)
    ax4 = axes[1, 1]
    ax4.plot(steps, analysis_results['mi_trajectory_ksg'], 'purple', linewidth=2, label='MI (KSG)')
    ax4.set_xlabel('Diffusion Step')
    ax4.set_ylabel('Mutual Information (bits)')
    ax4.set_title('I(State; Answer) - KSG Method')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Information Gain
    ax5 = axes[2, 0]
    if len(analysis_results['info_gain']) > 0:
        ax5.plot(range(len(analysis_results['info_gain'])), analysis_results['info_gain'], 
                'orange', linewidth=2, label='Information Gain')
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark spikes
        for spike in analysis_results['spikes']:
            ax5.axvline(x=spike['step'], color='red', linestyle=':', alpha=0.7)
            ax5.text(spike['step'], spike['gain'], f'{spike["gain"]:.3f}', 
                    rotation=90, ha='center', va='bottom', fontsize=8)
    
    ax5.set_xlabel('Diffusion Step')
    ax5.set_ylabel('Change in Confidence')
    ax5.set_title('Information Gain per Step (with spikes)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Conditional Entropy
    ax6 = axes[2, 1]
    ax6.plot(steps, analysis_results['conditional_entropy_trajectory'], 'brown', linewidth=2, label='H(Y|X)')
    ax6.set_xlabel('Diffusion Step')
    ax6.set_ylabel('Conditional Entropy (bits)')
    ax6.set_title('Conditional Entropy H(Answer|State)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive analysis saved to {save_path}")

def analyze_s_curve_behavior(confidence_trajectory, threshold_low=0.1, threshold_high=0.8):
    """
    Analyze S-curve behavior in the confidence trajectory.
    
    Args:
        confidence_trajectory: List of confidence values over time
        threshold_low: Lower threshold for S-curve detection
        threshold_high: Upper threshold for S-curve detection
    
    Returns:
        Dictionary with S-curve analysis
    """
    if len(confidence_trajectory) < 3:
        return {'has_s_curve': False, 'transition_start': None, 'transition_end': None}
    
    # Find transition points
    transition_start = None
    transition_end = None
    
    for i, conf in enumerate(confidence_trajectory):
        if transition_start is None and conf > threshold_low:
            transition_start = i
        if transition_start is not None and conf > threshold_high:
            transition_end = i
            break
    
    # Check for S-curve characteristics
    has_s_curve = False
    if transition_start is not None and transition_end is not None:
        # Check if there's a rapid increase in the middle
        if transition_end > transition_start:
            transition_length = transition_end - transition_start
            if transition_length > 0:
                slope = (confidence_trajectory[transition_end] - confidence_trajectory[transition_start]) / transition_length
                has_s_curve = slope > 0.1  # Significant slope
    
    return {
        'has_s_curve': has_s_curve,
        'transition_start': transition_start,
        'transition_end': transition_end,
        'transition_length': transition_end - transition_start if transition_end and transition_start else 0,
        'max_confidence': max(confidence_trajectory) if confidence_trajectory else 0,
        'final_confidence': confidence_trajectory[-1] if confidence_trajectory else 0
    }

def main():
    """
    Main function to run the enhanced mutual information analysis.
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
    
    # Run enhanced analysis
    methods = ['softmax_sum', 'max_prob']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running enhanced analysis with method: {method}")
        print(f"{'='*60}")
        
        # Perform comprehensive analysis
        analysis_results = advanced_information_analysis(
            logits_list, results_list, tokenizer, method=method
        )
        
        # Plot results
        plot_comprehensive_analysis(
            analysis_results, 
            save_path=f'comprehensive_mi_analysis_{method}.png'
        )
        
        # Analyze S-curve behavior
        s_curve_analysis = analyze_s_curve_behavior(analysis_results['confidence_trajectory'])
        
        print(f"\nS-CURVE ANALYSIS ({method}):")
        print(f"Has S-curve behavior: {s_curve_analysis['has_s_curve']}")
        print(f"Transition start: {s_curve_analysis['transition_start']}")
        print(f"Transition end: {s_curve_analysis['transition_end']}")
        print(f"Transition length: {s_curve_analysis['transition_length']}")
        print(f"Max confidence: {s_curve_analysis['max_confidence']:.4f}")
        print(f"Final confidence: {s_curve_analysis['final_confidence']:.4f}")
        
        # Print spike information
        print(f"\nINFORMATION SPIKES ({method}):")
        print(f"Number of spikes: {len(analysis_results['spikes'])}")
        for i, spike in enumerate(analysis_results['spikes'][:5]):  # Show first 5
            print(f"  Spike {i+1}: Step {spike['step']}, Gain {spike['gain']:.4f}")
        
        # Save results
        torch.save(analysis_results, f'enhanced_mi_results_{method}.pt')
        torch.save(s_curve_analysis, f's_curve_analysis_{method}.pt')
        
        print(f"Results saved to enhanced_mi_results_{method}.pt")

if __name__ == '__main__':
    main() 