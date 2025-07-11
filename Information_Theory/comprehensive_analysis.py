import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from mutual_information_analysis import analyze_information_gain, analyze_information_spikes
from enhanced_mi_analysis import advanced_information_analysis, analyze_s_curve_behavior
from plot_logits import linear_cka, kl_divergence, run_tsne_analysis

def combined_analysis(logits_path='averaged_logit_trajectory.pt', 
                     individual_logits_path='all_problems_logits.pt',
                     results_path='inference_results.pt'):
    """
    Run comprehensive analysis combining MI, CKA, KL, and t-SNE analysis.
    
    Args:
        logits_path: Path to averaged logits
        individual_logits_path: Path to individual problem logits
        results_path: Path to inference results
    
    Returns:
        Dictionary with all analysis results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    try:
        averaged_logits = torch.load(logits_path, map_location=device, weights_only=False)
        individual_logits = torch.load(individual_logits_path, map_location=device, weights_only=False)
        results_list = torch.load(results_path, map_location=device, weights_only=False)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/data/sls/u/urop/mvideet/diffusion_reasoning/models/llada_8b", trust_remote_code=True)
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run load_diffusion.py first to generate the required data files.")
        return None
    
    print(f"Loaded {len(averaged_logits)} averaged steps and {len(individual_logits)} individual problems")
    
    # 1. CKA Analysis (from plot_logits.py)
    print("\n1. Running CKA analysis...")
    features = [l.reshape(-1, l.size(-1)).to(device) for l in averaged_logits]
    cka_scores = []
    for i in range(len(features) - 1):
        if i % 10 == 0:
            print(f"CKA step {i}/{len(features)-1}")
        score = linear_cka(features[i], features[i+1], device=device)
        cka_scores.append(score)
    
    # 2. KL Divergence Analysis
    print("\n2. Running KL divergence analysis...")
    kl_scores = []
    for i in range(len(averaged_logits) - 1):
        if i % 10 == 0:
            print(f"KL step {i}/{len(averaged_logits)-1}")
        kl_score = kl_divergence(averaged_logits[i], averaged_logits[i+1], device=device)
        kl_scores.append(kl_score)
    
    # 3. Mutual Information Analysis
    print("\n3. Running mutual information analysis...")
    mi_analysis = analyze_information_gain(individual_logits, results_list, tokenizer, method='softmax_sum')
    spike_analysis = analyze_information_spikes(mi_analysis)
    
    # 4. Enhanced Information Analysis
    print("\n4. Running enhanced information analysis...")
    enhanced_analysis = advanced_information_analysis(individual_logits, results_list, tokenizer, method='softmax_sum')
    s_curve_analysis = analyze_s_curve_behavior(enhanced_analysis['confidence_trajectory'])
    
    # 5. t-SNE Analysis
    print("\n5. Running t-SNE analysis...")
    X_tsne, step_labels, total_steps = run_tsne_analysis(averaged_logits, device=device, n_samples=500)
    
    return {
        'cka_scores': cka_scores,
        'kl_scores': kl_scores,
        'mi_analysis': mi_analysis,
        'spike_analysis': spike_analysis,
        'enhanced_analysis': enhanced_analysis,
        's_curve_analysis': s_curve_analysis,
        'tsne_data': (X_tsne, step_labels, total_steps),
        'num_steps': len(averaged_logits)
    }

def plot_comprehensive_dashboard(analysis_results, save_path='comprehensive_dashboard.png'):
    """
    Create a comprehensive dashboard showing all analysis results.
    
    Args:
        analysis_results: Dictionary from combined_analysis
        save_path: Path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 24))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. CKA Similarity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(len(analysis_results['cka_scores'])), analysis_results['cka_scores'], 
             'b-', linewidth=2, label='CKA Similarity')
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('CKA Similarity')
    ax1.set_title('CKA Similarity Between Consecutive Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. KL Divergence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(analysis_results['kl_scores'])), analysis_results['kl_scores'], 
             'r-', linewidth=2, label='KL Divergence')
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Divergence Between Consecutive Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Answer Confidence
    ax3 = fig.add_subplot(gs[0, 2])
    confidence = analysis_results['mi_analysis']['avg_confidence']
    ax3.plot(range(len(confidence)), confidence, 'g-', linewidth=2, label='Confidence')
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Answer Confidence')
    ax3.set_title('Confidence in Correct Answer')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Entropy Trajectory
    ax4 = fig.add_subplot(gs[1, 0])
    entropy_traj = analysis_results['enhanced_analysis']['entropy_trajectory']
    ax4.plot(range(len(entropy_traj)), entropy_traj, 'purple', linewidth=2, label='Entropy')
    ax4.set_xlabel('Diffusion Step')
    ax4.set_ylabel('Entropy (bits)')
    ax4.set_title('Entropy Trajectory H(P_t)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Information Gain
    ax5 = fig.add_subplot(gs[1, 1])
    info_gain = analysis_results['enhanced_analysis']['info_gain']
    if len(info_gain) > 0:
        ax5.plot(range(len(info_gain)), info_gain, 'orange', linewidth=2, label='Info Gain')
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark spikes
        spikes = analysis_results['spike_analysis']['spikes']
        for spike in spikes:
            ax5.axvline(x=spike['step'], color='red', linestyle=':', alpha=0.7)
    
    ax5.set_xlabel('Diffusion Step')
    ax5.set_ylabel('Information Gain')
    ax5.set_title('Information Gain per Step')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Mutual Information Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    mi_binning = analysis_results['enhanced_analysis']['mi_trajectory_binning']
    mi_ksg = analysis_results['enhanced_analysis']['mi_trajectory_ksg']
    
    steps = range(len(mi_binning))
    ax6.plot(steps, mi_binning, 'red', linewidth=2, label='MI (Binning)')
    ax6.plot(steps, mi_ksg, 'blue', linewidth=2, label='MI (KSG)')
    ax6.set_xlabel('Diffusion Step')
    ax6.set_ylabel('Mutual Information (bits)')
    ax6.set_title('I(State; Answer) - Different Estimators')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. t-SNE Visualization
    ax7 = fig.add_subplot(gs[2, :2])
    X_tsne, step_labels, total_steps = analysis_results['tsne_data']
    
    if X_tsne is not None:
        unique_steps = np.unique(step_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_steps)))
        
        for i, step in enumerate(unique_steps):
            mask = step_labels == step
            ax7.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[colors[i]], alpha=0.6, s=30, 
                       label=f'Step {step}')
        
        ax7.set_xlabel('t-SNE Dimension 1')
        ax7.set_ylabel('t-SNE Dimension 2')
        ax7.set_title('t-SNE: Logits Evolution Across Diffusion Steps')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax7.text(0.5, 0.5, 't-SNE data not available', ha='center', va='center')
        ax7.set_title('t-SNE Visualization')
    
    # 8. Combined Trajectory Analysis
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Normalize all trajectories to [0, 1] for comparison
    def normalize(data):
        data = np.array(data)
        if len(data) > 0 and np.max(data) > np.min(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        return data
    
    steps = range(min(len(confidence), len(entropy_traj), len(cka_scores)))
    
    ax8.plot(steps, normalize(confidence[:len(steps)]), 'g-', linewidth=2, label='Confidence')
    ax8.plot(steps, normalize(entropy_traj[:len(steps)]), 'purple', linewidth=2, label='Entropy')
    ax8.plot(steps, normalize(cka_scores[:len(steps)]), 'b-', linewidth=2, label='CKA')
    ax8.plot(steps, normalize(kl_scores[:len(steps)]), 'r-', linewidth=2, label='KL Div')
    
    ax8.set_xlabel('Diffusion Step')
    ax8.set_ylabel('Normalized Value')
    ax8.set_title('Normalized Trajectories Comparison')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # 9. Summary Statistics
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""
    COMPREHENSIVE ANALYSIS SUMMARY
    
    Dataset: {len(analysis_results['mi_analysis']['confidence_trajectories'])} problems, {analysis_results['num_steps']} diffusion steps
    
    S-CURVE ANALYSIS:
    • Has S-curve behavior: {analysis_results['s_curve_analysis']['has_s_curve']}
    • Transition start: Step {analysis_results['s_curve_analysis']['transition_start']}
    • Transition end: Step {analysis_results['s_curve_analysis']['transition_end']}
    • Max confidence: {analysis_results['s_curve_analysis']['max_confidence']:.3f}
    
    INFORMATION SPIKES:
    • Number of spikes: {len(analysis_results['spike_analysis']['spikes'])}
    • Total information gain: {analysis_results['spike_analysis']['total_gain']:.3f}
    • Average spike gain: {analysis_results['spike_analysis']['avg_spike_gain']:.3f}
    
    TRAJECTORY CHARACTERISTICS:
    • Final confidence: {confidence[-1]:.3f} (vs initial: {confidence[0]:.3f})
    • Final entropy: {entropy_traj[-1]:.3f} (vs initial: {entropy_traj[0]:.3f})
    • Average CKA similarity: {np.mean(analysis_results['cka_scores']):.3f}
    • Average KL divergence: {np.mean(analysis_results['kl_scores']):.3f}
    
    KEY INSIGHTS:
    • Information gain spikes occur at steps: {[s['step'] for s in analysis_results['spike_analysis']['spikes'][:5]]}
    • Confidence exhibits {'S-curve' if analysis_results['s_curve_analysis']['has_s_curve'] else 'non-S-curve'} behavior
    • Entropy {'decreases' if entropy_traj[-1] < entropy_traj[0] else 'increases'} during diffusion
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Comprehensive Information Dynamics Analysis\nDiffusion Model - Mutual Information, CKA, KL Divergence, and t-SNE', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comprehensive dashboard saved to {save_path}")

def export_results_summary(analysis_results, save_path='analysis_summary.txt'):
    """
    Export a detailed text summary of all analysis results.
    
    Args:
        analysis_results: Dictionary from combined_analysis
        save_path: Path to save the summary
    """
    summary = []
    summary.append("="*80)
    summary.append("COMPREHENSIVE INFORMATION DYNAMICS ANALYSIS")
    summary.append("="*80)
    summary.append("")
    
    # Dataset info
    summary.append(f"Dataset: {len(analysis_results['mi_analysis']['confidence_trajectories'])} problems")
    summary.append(f"Diffusion steps: {analysis_results['num_steps']}")
    summary.append("")
    
    # S-curve analysis
    s_curve = analysis_results['s_curve_analysis']
    summary.append("S-CURVE ANALYSIS:")
    summary.append(f"  Has S-curve behavior: {s_curve['has_s_curve']}")
    summary.append(f"  Transition start: Step {s_curve['transition_start']}")
    summary.append(f"  Transition end: Step {s_curve['transition_end']}")
    summary.append(f"  Transition length: {s_curve['transition_length']} steps")
    summary.append(f"  Max confidence: {s_curve['max_confidence']:.4f}")
    summary.append(f"  Final confidence: {s_curve['final_confidence']:.4f}")
    summary.append("")
    
    # Spike analysis
    spike_analysis = analysis_results['spike_analysis']
    summary.append("INFORMATION SPIKES:")
    summary.append(f"  Number of spikes: {len(spike_analysis['spikes'])}")
    summary.append(f"  Total information gain: {spike_analysis['total_gain']:.4f}")
    summary.append(f"  Average spike gain: {spike_analysis['avg_spike_gain']:.4f}")
    summary.append("  Spike details:")
    for i, spike in enumerate(spike_analysis['spikes'][:10]):  # Show first 10
        summary.append(f"    Spike {i+1}: Step {spike['step']}, Gain {spike['gain']:.4f}")
    summary.append("")
    
    # Trajectory statistics
    confidence = analysis_results['mi_analysis']['avg_confidence']
    entropy_traj = analysis_results['enhanced_analysis']['entropy_trajectory']
    
    summary.append("TRAJECTORY STATISTICS:")
    summary.append(f"  Confidence: {confidence[0]:.4f} → {confidence[-1]:.4f} (change: {confidence[-1] - confidence[0]:.4f})")
    summary.append(f"  Entropy: {entropy_traj[0]:.4f} → {entropy_traj[-1]:.4f} (change: {entropy_traj[-1] - entropy_traj[0]:.4f})")
    summary.append(f"  Average CKA similarity: {np.mean(analysis_results['cka_scores']):.4f}")
    summary.append(f"  Average KL divergence: {np.mean(analysis_results['kl_scores']):.4f}")
    summary.append("")
    
    # Key insights
    summary.append("KEY INSIGHTS:")
    summary.append(f"  • Information gain spikes occur at steps: {[s['step'] for s in spike_analysis['spikes'][:5]]}")
    summary.append(f"  • Confidence shows {'S-curve' if s_curve['has_s_curve'] else 'non-S-curve'} behavior")
    summary.append(f"  • Entropy {'decreases' if entropy_traj[-1] < entropy_traj[0] else 'increases'} during diffusion")
    summary.append(f"  • CKA similarity {'increases' if np.mean(analysis_results['cka_scores'][:10]) < np.mean(analysis_results['cka_scores'][-10:]) else 'decreases'} over time")
    summary.append("")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"Analysis summary saved to {save_path}")

def main():
    """
    Main function to run the comprehensive analysis.
    """
    print("Starting comprehensive information dynamics analysis...")
    
    # Run combined analysis
    results = combined_analysis()
    
    if results is None:
        print("Analysis failed. Please check data files.")
        return
    
    # Create comprehensive dashboard
    plot_comprehensive_dashboard(results)
    
    # Export summary
    export_results_summary(results)
    
    # Save results
    torch.save(results, 'comprehensive_analysis_results.pt')
    print("Complete analysis results saved to 'comprehensive_analysis_results.pt'")
    
    print("\nAnalysis complete! Check the generated files:")
    print("  - comprehensive_dashboard.png: Visual dashboard")
    print("  - analysis_summary.txt: Detailed text summary")
    print("  - comprehensive_analysis_results.pt: Complete results data")

if __name__ == '__main__':
    main() 