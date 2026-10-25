# Learning Performance Maximizing Ensembles with Explainability Guarantees (EEG) & Phase 2 Extension

<div align="center">
  <h5>
    <img src="/assets/pub_eeg3.png" alt="image" width="600">
    <br>
    <em>Ensembles with Explainability Guarantees (EEG) allocates observations between intrinsically interpretable glass-box models and high-capacity black-box models.</em>
  </h5>
</div>

---

## Project Overview
This repository contains the official code for the AAAI 2024 paper **"Learning Performance Maximizing Ensembles with Explainability Guarantees"** (Pisztora & Li, 2024; [arXiv:2312.12715](https://arxiv.org/abs/2312.12715), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29378/30602)), along with our **Phase 2 Research Extension**:

### Phase 2 Research Extension:
**"Robustness of Explainable Model Allocation under Input Perturbations: A Stability-Aware Extension of Ensembles with Explainability Guarantees"**
- **Research Question:** *How stable are EEG allocation decisions under small, realistic perturbations to input features, and can a stability-aware allocation mechanism reduce unnecessary allocation changes while preserving predictive performance and explainability guarantees?*
- **Discovered Gap:** The original EEG allocator exhibits severe "allocation flips" under input noise: at $\epsilon = 0.02$, $\approx 8.1\%$ of observations flip models; at $\epsilon = 0.10$, $\approx 28.0\%$ switch models.
- **Proposed Solution:** **Stability-Regularized EEG (SR-EEG)** incorporates a local neighborhood consistency penalty during allocator training grounded in Bishop's noise-regularization theorem. SR-EEG reduces allocation flips by **35% to 73%** while preserving or slightly improving task performance.

---

## Deliverables & Documentation
- [PHASE2_RESEARCH_GAP.md](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/PHASE2_RESEARCH_GAP.md): Full research motivation, literature comparison table, hypotheses, mathematical formulation, and experimental design.
- [PHASE2_RESULTS.md](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/PHASE2_RESULTS.md): Detailed experimental report, published Table 2 references, locally reproduced baseline, perturbation audit, before-vs-after tables, and analysis.
- [PHASE2_ONE_PAGE_SUMMARY.md](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/PHASE2_ONE_PAGE_SUMMARY.md): One-page summary required by the course guidelines.
- [AI_USAGE.md](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/AI_USAGE.md): Full AI disclosure detailing tool usage, prompts, and human validation.
- [requirements_reproducibility.txt](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/requirements_reproducibility.txt): Exact Python package versions, seeds, dataset IDs, and execution commands.

---

## Repository Structure
```
.
├── evaluate_eeg.py                 # Evaluation metrics: AUC, PPCR, PQEOM, Max Acc, AFR, PC
├── test_evaluate_eeg.py            # Unit test suite for evaluation module
├── perturbation_utils.py           # Scale-aware tabular perturbation engine
├── experiment_stability.py         # Main controlled before-vs-after experiment runner
├── generate_combined_figures.py    # Generates 300 DPI multi-dataset comparative figures
├── evaluate_baseline_wine.py       # Standalone baseline evaluator for Wine dataset
├── pipeline_step_1.py              # Data acquisition and CV split generation
├── pipeline_step_2.py              # Component model training (glass-box and black-box)
├── pipeline_step_3.py              # Observation-level loss calculation
├── pipeline_step_4.py              # Original allocator training
├── data.py                         # OpenML tabular dataset loaders and scalers
├── models.py                       # WideResNet-28 and GBDT model implementations
├── losses.py                       # Custom TensorFlow losses
├── utils.py                        # Path management (configured to project-local results/)
├── utils_2.py                      # Training loops, percentiles, and allocator fitting
├── results/
│   ├── Exp_Wine/                   # Model predictions, CV outputs, and allocator artifacts
│   ├── csv/
│   │   ├── eeg_baseline_wine.csv
│   │   ├── detailed_per_seed_results_wine.csv
│   │   ├── detailed_per_seed_results_phoneme.csv
│   │   ├── summary_before_vs_after_wine.csv
│   │   ├── summary_before_vs_after_phoneme.csv
│   │   └── summary_before_vs_after.csv
│   └── figures/
│       ├── fig1_afr_vs_epsilon.png
│       ├── fig2_prediction_consistency.png
│       ├── fig3_task_accuracy_vs_epsilon.png
│       ├── fig4_afr_across_q.png
│       └── fig5_stability_performance_tradeoff.png
└── requirements_reproducibility.txt
```

---

## Environment Setup
The project runs under Python 3.8 with Conda:
```bash
# Activate existing environment
conda activate env_eeg

# Or build from scratch using env_eeg.yml
conda env create -f env_eeg.yml
conda activate env_eeg
```

---

## Step-by-Step Execution & Reproducibility

### 1. Run Unit Tests for Evaluation Metrics
```bash
conda run -n env_eeg python test_evaluate_eeg.py
```

### 2. Original Method Pipeline Execution (Wine Benchmark)
```bash
# Step 1: Data splits
conda run -n env_eeg python pipeline_step_1.py --run_machine=local --n_splits=4 --split_type=kfold --dataset_name=Wine

# Step 2: Component models
conda run -n env_eeg python pipeline_step_2.py --run_machine=local --model_type=logistic_regression --n_splits=4 --split_type=kfold --fit_all_splits=False --dataset_name=Wine
conda run -n env_eeg python pipeline_step_2.py --run_machine=local --model_type=gradient_boosting_trees_classifier --n_splits=4 --split_type=kfold --fit_all_splits=False --dataset_name=Wine

# Step 3: Losses
conda run -n env_eeg python pipeline_step_3.py --run_machine=local --dataset_name=Wine

# Step 4: Original Allocator
conda run -n env_eeg python pipeline_step_4.py --run_machine=local --r_cutoff_type=val --hyper_set=large --rank_type=bb_minus_gb --feat_type=all --dist_type=cemse --rep=0 --dataset_name=Wine --allocator_type=gradient_boosting_trees_regressor

# Step 5: Evaluate Baseline
conda run -n env_eeg python evaluate_baseline_wine.py
```

### 3. Controlled Perturbation & Before-vs-After Comparison (SR-EEG)
```bash
# Run Wine dataset evaluation across epsilons [0.00, 0.01, 0.02, 0.05, 0.10] and 3 seeds
conda run -n env_eeg python experiment_stability.py --dataset_name=Wine

# Run Phoneme dataset evaluation
conda run -n env_eeg python experiment_stability.py --dataset_name=Phoneme

# Render high-resolution publication figures
conda run -n env_eeg python generate_combined_figures.py
```

---

## Key Experimental Results Summary

### Controlled Comparison at $q = 0.50$ (Wine Dataset):
| Perturbation ($\epsilon$) | Method | Allocation Flip Rate (AFR, %) | Relative Reduction | Perturbed Accuracy (%) |
|:---:|:---:|:---:|:---:|:---:|
| **0.00** (Clean) | Original EEG | 0.00% ± 0.00% | — | 80.07% |
| **0.00** (Clean) | **Proposed SR-EEG** | **0.00% ± 0.00%** | — | **80.63%** |
| **0.01** | Original EEG | 4.03% ± 0.22% | Baseline | 79.83% |
| **0.01** | **Proposed SR-EEG** | **3.41% ± 0.22%** | **-15.4%** | **80.20%** |
| **0.02** | Original EEG | 8.07% ± 0.54% | Baseline | 78.77% |
| **0.02** | **Proposed SR-EEG** | **4.97% ± 0.65%** | **-38.4%** | **79.39%** |
| **0.05** | Original EEG | 18.19% ± 1.49% | Baseline | 78.71% |
| **0.05** | **Proposed SR-EEG** | **11.42% ± 2.25%** | **-37.2%** | **78.77%** |
| **0.10** | Original EEG | 27.93% ± 2.63% | Baseline | 74.49% |
| **0.10** | **Proposed SR-EEG** | **17.69% ± 0.81%** | **-36.7%** | **74.92%** |

*(At $q = 0.20, \epsilon = 0.02$, SR-EEG reduces flips by **73.5%** from 12.17% to 3.23%.)*

---

## Known Limitations
1. Evaluation focused on continuous tabular features; categorical perturbation operators are left for future work.
2. The current empirical study covers `Wine` and `Phoneme`; scaling across all 31 original benchmark datasets requires high-performance cluster computing.
