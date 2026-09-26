# Phase 1 Replication Summary
## EEG: Ensembles with Explainability Guarantees (AAAI 2024)

**Paper:** *Learning Performance Maximizing Ensembles with Explainability Guarantees*  
**Authors:** Vincent Pisztora, Jia Li  
**Venue:** AAAI 2024  
**Student:** [Student Name] — Data Science & AI, Year 3  
**Course:** Explainable AI (XAI)

---

## 1. Objective

This Phase 1 replication independently implements the EEG (Ensembles with Explainability Guarantees) methodology from Pisztora & Li (AAAI 2024). The goal is to reproduce the paper's experimental results on 5 selected tabular datasets using a clean, independently-written codebase.

---

## 2. EEG Methodology Summary

EEG trains a **glass-box model** `g` (intrinsically explainable) and a **black-box model** `b` independently on the full training data. It then learns an **allocator** `a_q` that decides, for each test observation, whether `g` or `b` should make the prediction.

The key control parameter is **q ∈ [0, 1]**: the proportion of test observations routed to `g`. At `q = 0`, all observations go to `b`; at `q = 1`, all go to `g`.

The **desirability score** (Equation from paper, p. 14618) determines which observations benefit most from glass-box assignment:

```
r̃(z) = 2·s_g(z) − s_b(z) − σ(l_U(g(x), y) − l_U(b(x), y))
```

where `σ` is the sigmoid function and `s_f(z)` is the sufficiency indicator.

---

## 3. Datasets Used

| Dataset | Type | Samples | Features | OpenML ID |
|---------|------|---------|----------|-----------|
| Wine | Classification | 2,554 | 11 | 44091 |
| Bank | Classification | 10,578 | 7 | 44126 |
| PolR | Regression | 15,000 | 26 | 44133 |
| SuperconductR | Regression | 21,263 | 79 | 44148 |
| BrazilianHousesR | Regression | 10,692 | 8 | 44141 |

All datasets from the Grinsztajn benchmark (OpenML). Split: 70% train / 9% val / 21% test.

---

## 4. Component Models

### Glass-Box (Intrinsically Explainable)
- **Classification**: Logistic Regression (L1/L2) + Decision Tree Classifier
  - Tuned via 4-fold CV on training set (parameter: `C` or `max_depth`)
  - Best on validation selected as glass-box
- **Regression**: Lasso Linear Regression + Decision Tree Regressor
  - Tuned via 4-fold CV on training set (parameter: `alpha` or `max_depth`)

### Black-Box
- **Classification**: Gradient Boosting Trees Classifier (GradientBoostingClassifier)
- **Regression**: Gradient Boosting Trees Regressor (GradientBoostingRegressor)
- Hyperparameter tuning: 4-fold CV grid search over `learning_rate, n_estimators, max_depth, subsample`

### Neural Network (TabWRN-28)
- Tabular adaptation of Wide ResNet-28 (Zagoruyko & Komodakis 2016)
- Replaces convolutional layers with fully connected Dense layers
- Architecture: stem → 3 groups × 4 residual blocks (24 residual layers) → head
- Group widths: [base_nodes·k, 2·base_nodes·k, 4·base_nodes·k]
- Pre-activation residual blocks with BatchNorm, ReLU, Dropout
- Tuned on validation set (not CV), as specified in the paper

---

## 5. Sufficiency Definitions

**Classification:** `s_f(z) = I{f(x) = y}` — correct prediction indicator

**Regression:** `s_f(z) = I{l_U(f(x), y) < ε}` where ε = min(mean val loss of g, mean val loss of b)

**Sufficiency Categories:**
- **Zg**: g sufficient, b insufficient
- **Zb**: g insufficient, b sufficient  
- **Z2**: both sufficient
- **Z0**: neither sufficient

---

## 6. EEG Allocation Mechanism

1. Compute per-sample desirability score `r̃(z)` using training data
2. Normalize to rank `r(z) = rank(r̃(z)) / n`
3. Construct augmented allocator features: `[x, g(x), b(x), d_CE, d_MSE]`
4. Train GBT allocator to predict `r(z)` from augmented features
5. At test time, feature-dependent allocator `a'_q` sorts observations by predicted rank
6. Feature-independent allocator `a''_q` sorts by disagreement distance
7. At each q, compare `a'_q` vs `a''_q` on validation set, select better one → PCFA

---

## 7. Metrics Implemented

| Metric | Description |
|--------|-------------|
| AUC | Area under the performance-vs-q curve |
| PPCR | % performance captured over random baseline |
| PQEOM | % q values where EEG ≥ max(g, b) |
| PQOM | % q values where EEG > max(g, b) |
| PCFA | % q values where feature-dependent allocator is chosen |
| 95TQM | Highest q where EEG ≥ 95% of max(g, b) performance |
| Max Acc | Maximum EEG performance across all q |
| Argmax q | Highest q at which maximum performance is maintained |
| s Acc | Category prediction accuracy for the 4 sufficiency zones |

---

## 8. Implementation Notes and Discrepancies

### Compared to Paper's Reported Architecture

1. **Neural Network**: The paper describes TabWRN-28 adapted for tabular data. Our implementation faithfully replicates the wide ResNet structure (3 groups × 4 residual blocks + stem + head = 28 layer depth), replacing convolutions with Dense layers. TensorFlow 2.3 (via env_eeg) is used for compatibility.

2. **Number of Datasets**: Paper evaluates on 31 datasets; we replicate on 5 as specified by course requirements.

3. **Hyperparameter Grid**: The paper does not publish the full hyperparameter grid in the main text. We extracted the grid from the reference implementation and implemented it independently (see `src/models/model_factory.py`).

4. **Epsilon for Regression Sufficiency**: The paper states ε is set to "the lower of the average validation losses." We compute this strictly from validation data, fitted independently per seed, to avoid leakage.

5. **Allocator Selection (PCFA)**: At each q, we compare feature-dependent `a'_q` vs feature-independent `a''_q` on validation sufficient performance, selecting the better one. This matches the paper's description of the combined allocator.

---

## 9. Results vs Paper (Seed 0 Preliminary)

| Dataset | Metric | Paper | Ours (Seed 0) | Difference |
|---------|--------|-------|----------------|------------|
| Wine | AUC | 79 | 81.4 | +2.4 |
| Wine | PPCR | 21 | 36.2 | +15.2 |
| Wine | s_Acc | 78 | 79.0 | +1.0 |
| Bank | AUC | 76 | 79.1 | +3.1 |
| Bank | 95TQM | 100 | 100.0 | 0.0 |
| Bank | s_Acc | 71 | 73.7 | +2.7 |
| PolR | AUC | 98 | 87.8 | -10.2 |
| PolR | PQOM | 93 | 98.0 | +5.0 |
| PolR | s_Acc | 84 | 81.6 | -2.4 |
| BrazilianHousesR | AUC | 96 | 88.9 | -7.1 |
| BrazilianHousesR | 95TQM | 93 | 89.0 | -4.0 |

*Final 5-seed averages will be reported in Table 3 after all replicates complete.*

---

## 10. Possible Sources of Difference

1. **Hyperparameter Tuning Grid Size**: The reference implementation uses a larger grid (`xl` or `large` variant). Our grid is smaller for computational feasibility but matches the medium grid structure.

2. **Random Seed for Data Shuffling**: The original code shuffles with `seed=1` before train/test split. Our implementation uses seed-specific shuffling per replicate, which is more principled for multi-seed replication.

3. **GBT vs TabWRN Black-Box**: For our seed 0 runs, the GBT black-box was selected. The paper likely also reports results with the best black-box per dataset.

4. **Epsilon Calculation Window**: The paper's `r_cutoff_type='val'` is faithfully replicated; epsilon is always computed from the validation set only.

5. **q Grid Resolution**: We use q step = 0.01 (101 points) vs the paper's reported q step of 0.05 in some figures. This affects AUC computation slightly.

---

## 11. Reproducibility Checklist

- [x] All random seeds documented (0, 1, 2, 3, 4)
- [x] All preprocessing fitted strictly on training data
- [x] No test data used for model selection or hyperparameter tuning
- [x] Results saved in machine-readable CSV + JSON
- [x] Full q grid (0.00 to 1.00, step 0.01) implemented
- [x] Random allocator averaged over 10 repetitions
- [x] Oracle allocator uses ground truth sufficiency
- [x] All 9 paper metrics implemented
- [x] Unit tests pass (14/14)
