import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def linear_cka(X, Y, device='cuda'):
    """
    Compute the linear CKA similarity between two feature matrices X and Y.
    X, Y: 2D tensors of shape (n_samples, n_features).
    """
    # Ensure tensors are on GPU
    X = X.to(device)
    Y = Y.to(device)
    
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    HSIC = torch.norm(X.T @ Y)**2
    var1 = torch.norm(X.T @ X)**2
    var2 = torch.norm(Y.T @ Y)**2
    return (HSIC / (var1 * var2)**0.5).item()

def kl_divergence(logits_a, logits_b, device='cuda'):
    """
    Compute KL divergence between two logit distributions.
    logits_a, logits_b: tensors of shape (batch, seq_len, vocab_size)
    """
    logits_a = logits_a.to(device)
    logits_b = logits_b.to(device)
    
    # Flatten to (batch * seq_len, vocab_size) for proper KL computation
    logits_a_flat = logits_a.reshape(-1, logits_a.size(-1))
    logits_b_flat = logits_b.reshape(-1, logits_b.size(-1))
    
    # Compute log softmax
    log_pa = F.log_softmax(logits_a_flat, dim=-1)
    log_pb = F.log_softmax(logits_b_flat, dim=-1)
    
    # KL divergence: KL(p_a || p_b) = sum(p_a * (log_p_a - log_p_b))
    return F.kl_div(log_pb, log_pa, reduction="batchmean", log_target=True).item()

def run_tsne_analysis(logits_list, device='cuda', n_samples=1000, perplexity=30):
    """
    Run t-SNE analysis on logits across diffusion steps.
    
    Args:
        logits_list: List of logit tensors for each diffusion step
        device: Device to run computations on
        n_samples: Number of tokens to sample for t-SNE (for speed)
        perplexity: t-SNE perplexity parameter
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not installed. Install with: pip install scikit-learn")
        return None, None, None
    
    print(f"Running t-SNE analysis with {n_samples} samples...")
    
    # Prepare data for t-SNE
    all_embeddings = []
    step_labels = []
    
    # Sample tokens from each step
    for step, logits in enumerate(logits_list):
        if step % 5 == 0:  # Sample every 5th step to avoid overcrowding
            print(f"Processing step {step} for t-SNE...")
            
            # Flatten logits: (batch, seq_len, vocab) -> (batch*seq_len, vocab)
            logits_flat = logits.reshape(-1, logits.size(-1)).to(device)
            
            # Sample random subset of tokens for speed
            if logits_flat.size(0) > n_samples:
                indices = torch.randperm(logits_flat.size(0))[:n_samples]
                logits_sample = logits_flat[indices]
            else:
                logits_sample = logits_flat
            
            # Convert to probabilities for better t-SNE representation
            probs = F.softmax(logits_sample, dim=-1)
            
            # Convert to float32 and move to CPU for sklearn (numpy doesn't support bfloat16)
            embeddings = probs.float().cpu().numpy()
            all_embeddings.append(embeddings)
            step_labels.extend([step] * embeddings.shape[0])
    
    if not all_embeddings:
        print("No data collected for t-SNE")
        return None, None, None
    
    # Concatenate all embeddings
    X = np.vstack(all_embeddings)
    y = np.array(step_labels)
    
    print(f"Running t-SNE on {X.shape[0]} samples with {X.shape[1]} dimensions...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                n_iter=1000, learning_rate=200, verbose=1)
    X_tsne = tsne.fit_transform(X)
    
    return X_tsne, y, len(logits_list)

def main():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Path to your saved logits tensor (.pt file)
    logits_path = 'averaged_logit_trajectory.pt'

    # Load list of logits tensors
    print("Loading logits...")
    logits_list = torch.load(logits_path, map_location=device, weights_only=False)  # Load directly to GPU
    print(f"Loaded {len(logits_list)} logits tensors")
    print(f"Logits shape per step: {logits_list[0].shape}")

    # Flatten each logits tensor to (batch * seq_len, vocab) for CKA computation
    print("Flattening tensors for CKA...")
    features = [l.reshape(-1, l.size(-1)).to(device) for l in logits_list]
    print(f"Feature shape: {features[0].shape}")

    # Compute CKA between consecutive steps on GPU
    print("Computing CKA similarities...")
    cka_scores = []
    for i in range(len(features) - 1):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing CKA step {i}/{len(features)-1}")
        score = linear_cka(features[i], features[i+1], device=device)
        cka_scores.append(score)

    print(f"Computed {len(cka_scores)} CKA scores")

    # Compute KL divergences between consecutive steps
    print("Computing KL divergences...")
    kl_scores = []
    for i in range(len(logits_list) - 1):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing KL step {i}/{len(logits_list)-1}")
        kl_score = kl_divergence(logits_list[i], logits_list[i+1], device=device)
        kl_scores.append(kl_score)

    print(f"Computed {len(kl_scores)} KL divergence scores")

    # Run t-SNE analysis
    print("\n" + "="*50)
    print("Running t-SNE analysis...")
    X_tsne, step_labels, total_steps = run_tsne_analysis(logits_list, device=device, n_samples=500)
    
    if X_tsne is not None:
        print("t-SNE analysis completed successfully!")
    else:
        print("t-SNE analysis failed or skipped.")

    # Create subplots for all metrics
    if X_tsne is not None:
        fig = plt.figure(figsize=(15, 12))
        
        # CKA similarity
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(range(len(cka_scores)), cka_scores, linewidth=2, color='blue')
        ax1.set_xlabel('Diffusion step (t)')
        ax1.set_ylabel('Linear CKA similarity (t vs t+1)')
        ax1.set_title('CKA Similarity Trajectory')
        ax1.grid(True, alpha=0.3)
        
        # KL divergence
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(range(len(kl_scores)), kl_scores, linewidth=2, color='red')
        ax2.set_xlabel('Diffusion step (t)')
        ax2.set_ylabel('KL divergence (t vs t+1)')
        ax2.set_title('KL Divergence Trajectory')
        ax2.grid(True, alpha=0.3)
        
        # t-SNE scatter plot
        ax3 = plt.subplot(2, 2, 3)
        unique_steps = np.unique(step_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_steps)))
        
        for i, step in enumerate(unique_steps):
            mask = step_labels == step
            ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[colors[i]], alpha=0.6, s=20, 
                       label=f'Step {step}')
        
        ax3.set_xlabel('t-SNE Dimension 1')
        ax3.set_ylabel('t-SNE Dimension 2')
        ax3.set_title('t-SNE: Logits Evolution Across Steps')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # t-SNE trajectory (connect centroids)
        ax4 = plt.subplot(2, 2, 4)
        centroids = []
        for step in unique_steps:
            mask = step_labels == step
            centroid = X_tsne[mask].mean(axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        ax4.plot(centroids[:, 0], centroids[:, 1], 'o-', linewidth=2, markersize=8)
        
        # Add step labels
        for i, (x, y) in enumerate(centroids):
            ax4.annotate(f'{unique_steps[i]}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('t-SNE Dimension 1')
        ax4.set_ylabel('t-SNE Dimension 2')
        ax4.set_title('t-SNE: Diffusion Trajectory (Centroids)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('averaged_diffusion_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved complete analysis as 'averaged_diffusion_analysis.png'")
        
    else:
        # Fallback to original 2-subplot layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot CKA similarity
    ax1.plot(range(len(cka_scores)), cka_scores, linewidth=2, color='blue', label='CKA Similarity')
    ax1.set_xlabel('Diffusion step (t)')
    ax1.set_ylabel('Linear CKA similarity (t vs t+1)')
    ax1.set_title('CKA Similarity Trajectory Across Diffusion Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot KL divergence
    ax2.plot(range(len(kl_scores)), kl_scores, linewidth=2, color='red', label='KL Divergence')
    ax2.set_xlabel('Diffusion step (t)')
    ax2.set_ylabel('KL divergence (t vs t+1)')
    ax2.set_title('KL Divergence Trajectory Across Diffusion Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the combined figure
    plt.savefig('averaged_diffusion_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved combined plot as 'averaged_diffusion_analysis.png'")
    
    # Also create individual plots
    # CKA only
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cka_scores)), cka_scores, linewidth=2, color='blue')
    plt.xlabel('Diffusion step (t)')
    plt.ylabel('Linear CKA similarity (t vs t+1)')
    plt.title('CKA Similarity Trajectory Across Diffusion Steps')
    plt.grid(True, alpha=0.3)
    plt.savefig('averaged_cka_similarity.png', dpi=300, bbox_inches='tight')
    print("Saved CKA plot as 'averaged_cka_similarity.png'")
    
    # KL divergence only
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(kl_scores)), kl_scores, linewidth=2, color='red')
    plt.xlabel('Diffusion step (t)')
    plt.ylabel('KL divergence (t vs t+1)')
    plt.title('KL Divergence Trajectory Across Diffusion Steps')
    plt.grid(True, alpha=0.3)
    plt.savefig('averaged_kl_divergence.png', dpi=300, bbox_inches='tight')
    print("Saved KL plot as 'averaged_kl_divergence.png'")
    
    plt.show()

if __name__ == '__main__':
    main()
