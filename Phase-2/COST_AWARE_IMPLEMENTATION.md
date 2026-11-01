# Implementation Architecture & Technical Specification: Cost-Aware EEG

**Document:** System Architecture, Code Design & Algorithmic Modules  
**Component:** Phase 2 Research Extension — *Cost-Aware Allocation*  
**Auditor:** Senior ML Research Engineer  
**Status:** Implemented, Fully Tested (100% Pass Rate) & Documented  

---

## 1. Modular Directory & Codebase Organization

All newly created modules for the Cost-Aware extension reside cleanly inside `Phase-2/`, strictly isolated from root (Phase 1) and existing SR-EEG code:

```
Phase-2/
├── cost_aware_allocator.py           # Core decoupled allocator & conditional execution engine
├── threshold_selection.py            # Validation-only threshold selection & multi-objective optimizer
├── benchmark_inference_timing.py     # High-precision wall-clock latency benchmark utility
├── experiment_cost_aware.py          # Controlled multi-seed experiment runner
├── generate_cost_aware_results.py    # Generates all CSVs, JSON config, and 300 DPI figures
├── test_cost_aware_allocator.py      # Unit tests for allocator and conditional execution
├── test_threshold_selection.py       # Unit tests for thresholding, zero-leakage, and edge cases
├── docs/
│   └── cost_aware_audit.md           # Prompt 1 technical audit document
└── results/
    └── cost_aware_extension/         # Machine-readable CSVs, JSON, and publication figures
        ├── experiment_config.json
        ├── baseline_results.csv
        ├── cost_aware_results.csv
        ├── comparison_results.csv
        ├── inference_timing.csv
        ├── allocation_details.csv
        ├── fig1_coverage_vs_predictive_performance.png
        ├── fig2_coverage_vs_inference_latency.png
        └── fig3_cost_sensitivity_pareto.png
```

---

## 2. Core Modules & Interface Specifications

### 2.1 Decoupled Allocator Module (`cost_aware_allocator.py`)

#### `CostAwareAllocator` Class
- **Constructor:** `CostAwareAllocator(estimator=None, rank_type='bb_minus_gb', random_state=0)`
  - Reuses the existing model family: `GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=400, subsample=0.8)`.
- **Training Method:** `fit(X_train, y_train, loss_gb_train=None, loss_bb_train=None, ...)`
  - Generates ground-truth percentile target $r \in [0, 1]$ using [`compute_ground_truth_ranking`](file:///Users/shreyashsanjaygadling/Downloads/XAI-Project/Phase-2/cost_aware_allocator.py#L65-L113).
  - Fits the regressor strictly on $X_{\text{train}}$: zero dependency on black-box predictions at test time.
- **Inference Method:** `predict_ranking(X)`
  - Evaluates continuous glass-box desirability percentile: $\hat{r}_{\text{CA}}(X) \in [0, 1]$.
  - Input: Raw tabular feature matrix $X \in \mathbb{R}^{N \times d}$.
- **Routing Decision:** `route(X=None, r_hat=None, q=0.5)`
  - Applies threshold $\tau_q$: $a = (\hat{r}_{\text{CA}} > \tau_q).\text{astype}(\text{int})$.
  - Convention: $a = 1 \implies$ Glass-Box; $a = 0 \implies$ Black-Box.

#### Strict Conditional Execution Engine
- **Function:** `execute_cost_aware_inference(X, a_decisions, gb_model, bb_model, problem_type='classification', ...)`
- **Mechanism:**
  1. Identifies subsets: $\text{idx}_{\text{gb}} = \{i : a_i = 1\}$ and $\text{idx}_{\text{bb}} = \{i : a_i = 0\}$.
  2. If $\text{len}(\text{idx}_{\text{gb}}) > 0$: invokes strictly `gb_model.predict_proba(X[idx_gb])`. **$b(x)$ is never called for these samples.**
  3. If $\text{len}(\text{idx}_{\text{bb}}) > 0$: invokes strictly `bb_model.predict_proba(X[idx_bb])`. **$g(x)$ is never called for these samples.**
  4. Assembles composite probability matrix $P \in \mathbb{R}^{N \times K}$ and assigns class predictions via $\arg\max_c P_{i, c}$.
  5. Computes cost diagnostics: query counts ($n_{\text{gb}}, n_{\text{bb}}$), actual coverage ($\hat{q}$), wall-clock timings, realized compute cost, baseline cost ($N \cdot (C_G + C_B)$), and compute savings percentage.

---

### 2.2 Threshold Selection Module (`threshold_selection.py`)

#### Mode 1: Matched-Coverage Quantile Selection
- **Function:** `find_matched_coverage_threshold(r_hat_val, target_q)`
- Calculates the empirical $(1.0 - \text{target\_q})$-th quantile of validation scores $\hat{r}_{\text{val}}$.
- Guarantees that baseline and proposed allocators are evaluated under identical empirical coverage levels.

#### Mode 2: Multi-Objective Cost-Performance Optimization
- **Function:** `optimize_cost_aware_threshold(r_hat_val, y_val, ..., q_min, lambda_cost)`
- Evaluates candidate thresholds across validation rankings.
- Objective Function:
  $$J(\tau) = L_{\text{norm}}(\tau) + \lambda_{\text{cost}} \cdot C_{\text{norm}}(\tau)$$
  subject to:
  $$\text{Coverage}_{\text{gb}}(\tau) \ge q_{\min}$$
- Normalization guarantees:
  - $L_{\text{norm}} = \frac{\mathcal{L}_{\text{ce}}}{\log(K)} \in [0, 1]$
  - $C_{\text{norm}} = \frac{n_{\text{bb}}}{N} = 1 - \hat{q}(\tau) \in [0, 1]$

#### Guard Clauses & Edge-Case Protection
1. **Empty / Infeasible Sets:** Falls back to maximum possible coverage threshold ($\tau \to -\infty$).
2. **All-Glass-Box ($q=1.0$):** Threshold set below $\min(r)$; $n_{\text{gb}} = N, n_{\text{bb}} = 0$.
3. **All-Black-Box ($q=0.0$):** Threshold set above $\max(r)$; $n_{\text{gb}} = 0, n_{\text{bb}} = N$.
4. **Tied Scores:** Deterministic tie-breaking prioritizing lower $J$, then higher coverage, then lower threshold.
5. **Non-finite Values (NaN / Inf):** Sanitized using `np.nan_to_num`.

---

## 3. Latency Measurement Protocol (`benchmark_inference_timing.py`)

- **High-Precision Clock:** Utilizes `time.perf_counter()` to capture monotonic CPU time.
- **Warm-Up Runs:** 10 unrecorded iterations to warm CPU caches and eliminate instruction jitters.
- **Repetitions:** 30 to 50 repeated measurements per step.
- **Reported Statistics:** Mean, standard deviation, min, max, and milliseconds per observation ($\frac{\text{Mean}}{N}$).
- **Scope:** Excludes dataset loading, training, and initialization overhead. Focuses strictly on the inference path.

---

## 4. Verification & Unit Testing Suite

All unit tests are automated and execute cleanly in Python 3.12:

```bash
# 1. Cost-Aware Allocator Suite (6 tests)
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 Phase-2/test_cost_aware_allocator.py
# Output: All Cost-Aware Allocator unit tests passed successfully!

# 2. Threshold Selection Suite (6 tests including LeakageDetectorArray)
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 Phase-2/test_threshold_selection.py
# Output: All Threshold Selection unit tests passed successfully!

# 3. Existing Baseline Evaluator Suite
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 Phase-2/test_evaluate_eeg.py
# Output: All unit tests passed successfully!
```
