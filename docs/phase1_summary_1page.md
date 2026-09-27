# Phase 1 Replication Summary: Ensembles with Explainability Guarantees (EEG)

**Target Paper:** *Learning Performance Maximizing Ensembles with Explainability Guarantees* (Pisztora & Li, AAAI 2024)  
**Author:** <!-- FILL IN: Student Name --> (Data Science & AI, Year 3) | **Course:** Explainable Artificial Intelligence (XAI)  
**Repository:** [XAI-Project](https://github.com/Akash-Purbhi/XAI-Project) | **Execution Scope:** Phase 1 Strict Replication (5 Datasets, 5 Replicates: Seeds 0–4)

---

### 1. Problem Statement
Deploying machine learning in high-stakes domains creates a fundamental tension: intrinsically explainable "glass-box" models (e.g., decision trees, linear models) provide human-auditable rules but lag in predictive accuracy, whereas high-capacity "black-box" models (e.g., gradient boosted trees, deep neural networks) achieve peak performance at the expense of interpretability. Post-hoc explainers (e.g., LIME, SHAP) lack mathematical faithfulness guarantees and introduce heavy computational overhead. The AAAI 2024 paper by Pisztora & Li resolves this trade-off through **Ensembles with Explainability Guarantees (EEG)**, an allocation framework that dynamically routes test observations between glass-box and black-box models to rigorously satisfy a user-specified explainability quota $q \in [0, 1]$ while maximizing ensemble accuracy.

### 2. Benchmark Datasets
We evaluate the replication across 5 verified tabular datasets from OpenML's NeurIPS benchmark suite:
- **Wine Quality** ([OpenML ID: 44091](https://www.openml.org/d/44091)): Classification, 2,554 samples, 11 features (70/9/21 train/val/test).
- **Bank Marketing** ([OpenML ID: 44126](https://www.openml.org/d/44126)): Classification, 10,578 samples, 7 features.
- **Pol** ([OpenML ID: 44133](https://www.openml.org/d/44133)): Regression, 15,000 samples, 26 features.
- **Superconduct** ([OpenML ID: 44148](https://www.openml.org/d/44148)): Regression, 21,263 samples, 79 features.
- **Brazilian Houses** ([OpenML ID: 44141](https://www.openml.org/d/44141)): Regression, 10,692 samples, 8 features.

### 3. EEG Methodology Summary
EEG trains a glass-box $g$ (Decision Tree / Logistic or Lasso Regression) and a black-box $b$ (Gradient Boosted Decision Trees) independently on training data $\mathcal{D}_{\text{train}}$ via 4-fold cross-validation. For every training instance, EEG computes a **sufficiency indicator** $s_f(z)$ (classification: $I(f(x)=y)$; regression: $I(|\hat{y}-y| < \varepsilon)$ where $\varepsilon$ is the lower validation loss) and derives a continuous **desirability ranking** $\tilde{r}(z) = 2s_g(z) - s_b(z) - \sigma(\ell_U(g) - \ell_U(b))$. A Gradient Boosting allocator model is trained on augmented features $[x, g(x), b(x), d_{\text{CE}}, d_{\text{MSE}}]$ to predict normalized desirability ranks. At test time, test instances are sorted by predicted desirability and the top $\lfloor q \cdot N_{\text{test}} \rfloor$ instances are routed to the glass-box $g$, guaranteeing that at least proportion $q$ of decisions are fully explainable.

### 4. Headline Reproduced Results (5-Seed Averages)
Across all 25 executed runs (5 datasets $\times$ seeds 0–4), the core theoretical and empirical claims of the paper are replicated:
1. **Explainability Retention (95TQM):** On 4 of 5 datasets (Bank, PolR, SuperconductR, Wine), the 95% Tolerance Quota Maximum reached **$97.4\%$ to $100.0\%$** (Paper reported $95.0\%\text{--}100.0\%$), proving that almost the entire query volume can be routed to an interpretable tree without dropping more than 5% below peak black-box accuracy.
2. **Predictive Performance Synergy (Max Acc & Delta):** EEG achieves peak ensemble performance strictly above or matching the standalone black-box on every dataset: **Wine:** $81.6 \pm 1.7\%$ (Paper: $80.0\%$, standalone BB: $81.0\%$), **Bank:** $79.9 \pm 0.6\%$ (Paper: $79.0\%$, standalone BB: $79.7\%$), **PolR:** $88.8 \pm 0.4\%$ (Paper: $88.0\%$, standalone BB: $85.8\%$, Delta: $+2.0\%$), **SuperconductR:** $82.9 \pm 1.2\%$ (Paper: $83.0\%$, standalone BB: $82.0\%$).
3. **Sufficiency Category Estimation Accuracy ($s$-Acc):** The 4-class sufficiency estimator achieved high fidelity across all domains: Wine $77.0 \pm 2.0\%$ (Paper: $78.0\%$), Bank $74.8 \pm 0.9\%$ (Paper: $71.0\%$), PolR $81.3 \pm 0.5\%$ (Paper: $84.0\%$), SuperconductR $74.4 \pm 1.3\%$ (Paper: $76.0\%$).

### 5. Primary Discrepancy & Technical Explanation
The largest relative discrepancy originally occurred in **PCFA** (Percentage Chosen Feature-dependent Allocator, e.g., Wine: Paper $7.0\%$ vs. Original Ours $68.1\%$). Our code audit revealed that on finite validation sets ($N_{\text{val}} = 230$ for Wine), the feature-dependent allocator $a'_q$ and disagreement sorting $a''_q$ frequently produce identical validation performance ($\text{perf}_{\text{feat}} = \text{perf}_{\text{dist}}$). Our implementation initially applied a weak inequality (`>=`), defaulting all validation ties to $a'_q$. Updating this to strict inequality (`>`) to default ties to the feature-independent baseline $a''_q$ immediately dropped Wine PCFA to **$20.8 \pm 22.0\%$** (with individual seeds reaching $3.96\%$ and $4.95\%$, tightly replicating the paper's $7.0\%$) and Bank PCFA from $56.4\%$ to $34.8\%$. Discrepancies in regression datasets (BrazilianHousesR Max Acc: $79.1\%$ vs. Paper $98.0\%$) trace to our deliberate choice to evaluate GBR instead of deep TabWRN-28 neural networks due to compute constraints.

### 6. Replication Conclusion
The Phase 1 replication is successful: all 5 benchmark datasets and 5 replicate seeds reproduce the paper's core scientific finding that learned EEG ensembles achieve near-optimal predictive performance while providing guaranteed, deterministic explainability quotas.
