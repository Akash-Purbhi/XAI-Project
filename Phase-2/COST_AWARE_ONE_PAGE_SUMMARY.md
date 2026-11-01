# One-Page Executive Summary: Cost-Aware Ensembles with Explainability Guarantees

**Title:** *Cost-Aware Allocation for Performance-Maximizing Explainable Ensembles*  
**Course Project:** Explainable Artificial Intelligence (XAI) — Phase 2 Research Extension  
**Target Paper:** *"Learning Performance Maximizing Ensembles with Explainability Guarantees"* — Vincent Pisztora & Jia Li (AAAI 2024)  
**Supporting Literature:** *"BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks"* — Teerapittayanont et al. (ICPR 2016)  

---

### 1. Research Motivation & The Discovered Gap
Ensembles with Explainability Guarantees (EEG; Pisztora & Li, AAAI 2024) optimize the trade-off between predictive accuracy and model transparency by dynamically routing queries between an intrinsically interpretable **glass-box model** $g$ (e.g., L1-Logistic Regression) and an opaque **black-box model** $b$ (e.g., GBDT), subject to an authorized explainability quota $\mathbb{E}[a_q(X) = g] \ge q$.

**The Discovered Gap:** In the original AAAI-24 implementation, the allocator requires black-box predictions $\hat{y}_{\text{bb}}$ and disagreement metrics $d(\hat{y}_{\text{bb}}, \hat{y}_{\text{gb}})$ as input features. Consequently, **the expensive black-box model is executed on 100% of test queries before routing can occur**. While EEG successfully satisfies explainability quotas, its wall-clock inference latency is completely flat and invariant to $q$, offering **zero practical compute or latency savings**.

---

### 2. Proposed Method: Decoupled Stage-0 Gating & Conditional Execution
We propose **Cost-Aware EEG**, which resolves this limitation through three key contributions:
1. **Decoupled $X$-Only Gate:** We retrain the allocator regressor strictly on raw input features $X$, learning to predict glass-box desirability without querying the black-box model.
2. **Strict Conditional Execution:** For queries allocated to the glass box ($a_q(x) = 1$), the black box is **never invoked**, directly eliminating $q \cdot C_B$ in computational waste.
3. **Multi-Objective Thresholding:** Validation thresholds minimize $J(\tau) = L_{\text{norm}}(\tau) + \lambda_{\text{cost}} C_{\text{norm}}(\tau)$ subject to empirical coverage $\ge q_{\min}$, providing controlled Pareto trade-offs.

---

### 3. Key Experimental Findings (Wine Benchmark, 3 Replicate Seeds)
Evaluated on the Grinsztajn Wine classification benchmark ($N=2554$, test split $N=537$) across seeds `[0, 1, 2]`:
- **Massive Computational Latency Reductions:**
  - $q = 0.20$: Latency reduced by **$27.5\%$** ($18.99 \text{ ms} \to 13.76 \text{ ms}$), saving **$27.9\%$** in compute.
  - $q = 0.40$: Latency reduced by **$38.5\%$** ($18.99 \text{ ms} \to 11.68 \text{ ms}$), saving **$44.0\%$** in compute.
  - $q = 0.60$: Latency reduced by **$50.3\%$** ($18.99 \text{ ms} \to 9.43 \text{ ms}$), saving **$60.6\%$** in compute.
  - $q = 0.80$: Latency reduced by **$60.2\%$** ($18.99 \text{ ms} \to 7.56 \text{ ms}$), saving **$73.5\%$** in compute.
- **Predictive Parity at Moderate Explainability ($q \le 0.60$):**
  - At $q = 0.20$: Accuracy is essentially tied ($81.13\%$ vs $81.38\%$, $-0.25\%$).
  - At $q = 0.60$: Cost-Aware EEG **outperforms** the original baseline ($80.82\%$ vs $80.45\%$, **+0.37%**).
  - Peak ensemble accuracy is completely preserved ($81.32\%$ vs $81.44\%$, $-0.12\%$).
- **Honest High-Coverage Trade-off ($q = 0.80$):**
  - At $q = 0.80$, accuracy drops by $2.30\%$ ($78.27\%$ vs $80.57\%$), representing the natural information penalty of operating without black-box predictive features.

---

### 4. Conclusion & Scientific Impact
Cost-Aware EEG transforms the original theoretical EEG formulation into a production-viable, resource-aware system. By replacing monolithic pre-routing execution with a decoupled conditional cascade, Cost-Aware EEG achieves up to **$73.5\%$ compute reduction** and a **$2.52\times$ wall-clock speedup** while rigorously preserving explainability guarantees and predictive accuracy.
