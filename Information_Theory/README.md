# Mutual Information Analysis for Diffusion Models

This directory contains a comprehensive suite of tools for analyzing information dynamics in diffusion-based language models. The analysis focuses on measuring how much information about the final answer is gained at each step of the diffusion process.

## 🎯 Theoretical Framework

The core hypothesis is that during diffusion, the model gradually accumulates information about the correct answer, exhibiting:

1. **S-curve behavior**: Low information initially, rapid gain during "answer crystallization", then saturation
2. **Information spikes**: Discrete jumps in mutual information at specific steps (block unveilings, remasking events)
3. **Correlation with other metrics**: Information gain should align with CKA similarity changes and KL divergence spikes

## 📋 Files Overview

### Core Analysis Scripts

1. **`load_diffusion.py`** - Runs diffusion inference and saves logits trajectory
2. **`mutual_information_analysis.py`** - Basic MI analysis with confidence tracking
3. **`enhanced_mi_analysis.py`** - Advanced MI analysis with multiple estimators
4. **`comprehensive_analysis.py`** - Combines all analyses into a unified dashboard
5. **`plot_logits.py`** - Original CKA/KL analysis (from existing codebase)

### Analysis Types

- **Confidence Tracking**: Measures model's confidence in correct answer tokens
- **Mutual Information**: Estimates I(State; Answer) using multiple methods
- **Entropy Analysis**: Tracks H(P_t) over diffusion steps
- **Information Spikes**: Identifies discrete jumps in information gain
- **S-curve Detection**: Analyzes sigmoid-like behavior in confidence
- **CKA Similarity**: Measures representation similarity between steps
- **KL Divergence**: Quantifies distribution changes between steps
- **t-SNE Visualization**: Projects logits to 2D for visualization

## 🚀 Quick Start

### 1. Generate Diffusion Data

First, run the diffusion model on sample problems:

```bash
python load_diffusion.py
```

This will:
- Load the LLaDA diffusion model
- Run inference on 10 sample math problems
- Save three files:
  - `averaged_logit_trajectory.pt`: Averaged logits across problems
  - `all_problems_logits.pt`: Individual problem logits
  - `inference_results.pt`: Problem metadata and results

### 2. Run Comprehensive Analysis

Run the complete analysis suite:

```bash
python comprehensive_analysis.py
```

This produces:
- `comprehensive_dashboard.png`: Visual dashboard with all metrics
- `analysis_summary.txt`: Detailed text summary
- `comprehensive_analysis_results.pt`: Complete results data

### 3. Alternative: Run Individual Analyses

You can also run individual analysis scripts:

```bash
# Basic MI analysis
python mutual_information_analysis.py

# Enhanced MI analysis with multiple estimators
python enhanced_mi_analysis.py

# Original CKA/KL analysis
python plot_logits.py
```

## 📊 Interpreting Results

### Key Metrics

1. **Answer Confidence**: How confident the model is in the correct answer
   - Should show S-curve: low → rapid rise → saturation
   - Spikes indicate "eureka moments" when answer crystallizes

2. **Mutual Information I(State; Answer)**:
   - Measures how much the model's state reveals about the correct answer
   - Multiple estimators (binning, KSG) provide robustness
   - Should correlate with confidence trajectory

3. **Entropy H(P_t)**:
   - Uncertainty in the model's probability distribution
   - Should decrease as model becomes more confident
   - Inversely related to information gain

4. **Information Gain (∂Confidence/∂t)**:
   - Rate of information acquisition per step
   - Spikes indicate rapid information acquisition
   - Should align with CKA/KL changes

### Expected Patterns

✅ **Healthy Diffusion Process**:
- Clear S-curve in confidence trajectory
- 2-5 major information spikes
- Entropy decreases over time
- CKA similarity increases (representations stabilize)
- KL divergence spikes align with information spikes

❌ **Problematic Patterns**:
- Flat confidence trajectory (no learning)
- No clear information spikes (gradual drift)
- Entropy increases (model becomes less confident)
- Random spike patterns (unstable process)

### Dashboard Interpretation

The comprehensive dashboard shows:

1. **Row 1**: Core metrics (CKA, KL, Confidence)
2. **Row 2**: Information theory (Entropy, Info Gain, MI comparison)
3. **Row 3**: Visualization (t-SNE, normalized trajectories)
4. **Row 4**: Summary statistics and key insights

## 📈 Advanced Analysis

### S-Curve Analysis

The S-curve analysis detects sigmoid-like behavior:
- `transition_start`: When rapid information gain begins
- `transition_end`: When information gain saturates
- `transition_length`: Duration of rapid transition

### Information Spikes

Spikes are detected using adaptive thresholds:
- `threshold = 2 * std(info_gain)`
- Each spike records: step, gain magnitude, confidence before/after
- Largest spikes indicate most significant "breakthrough" moments

### Mutual Information Estimators

Two MI estimators provide robustness:
1. **Binning**: Discretizes continuous features, exact for discrete case
2. **KSG**: k-nearest neighbors estimator, better for continuous features

## 🔧 Customization

### Adjusting Parameters

In `load_diffusion.py`:
- `num_problems`: Number of test problems (default: 10)
- `steps`: Diffusion steps (default: 128)
- `gen_length`: Generation length (default: 128)
- `temperature`: Sampling temperature (default: 0.0)

In analysis scripts:
- `spike_threshold`: Minimum change for spike detection
- `method`: Confidence computation method ('softmax_sum', 'max_prob', 'mean_prob')
- `n_bins`: Number of bins for MI estimation

### Adding New Problems

Modify the `test_problems` list in `load_diffusion.py` to add custom problems:

```python
test_problems = [
    {"question": "Your question here", "answer": "Expected answer"},
    # Add more problems...
]
```

## 🔍 Troubleshooting

### Common Issues

1. **"Model not found"**: Ensure LLaDA model is in `./llada-8b/` directory
2. **"Data files not found"**: Run `load_diffusion.py` first
3. **Empty plots**: Check if logits were saved correctly
4. **Memory issues**: Reduce `num_problems` or `gen_length`

### GPU Memory Management

For large-scale analysis:
- Use `torch.cuda.empty_cache()` between analyses
- Process problems in smaller batches
- Use `torch.float16` instead of `torch.bfloat16` if needed

## 📚 References

### Theoretical Background

1. **Mutual Information**: I(X;Y) = H(X) - H(X|Y)
2. **CKA Similarity**: Linear CKA measures representation similarity
3. **KL Divergence**: KL(P||Q) = Σ P(x) log(P(x)/Q(x))
4. **Information Theory**: Cover & Thomas, "Elements of Information Theory"

### Related Work

- **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models"
- **Information Dynamics**: Tishby & Zaslavsky, "The Information Bottleneck Method"
- **MI Estimation**: Kraskov et al., "Estimating Mutual Information"

## 🤝 Contributing

To extend this analysis:

1. Add new MI estimators in `enhanced_mi_analysis.py`
2. Implement additional confidence metrics
3. Add support for different model architectures
4. Extend visualization capabilities

## 📄 License

This code is provided for research purposes. Please cite appropriately if used in publications.

---

**Questions?** Check the troubleshooting section or examine the comprehensive dashboard for insights into your specific diffusion process. 