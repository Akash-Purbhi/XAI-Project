# Phase 2 One-Page Research Gap & Extension Summary

**Project Title:** Robustness of Explainable Model Allocation under Input Perturbations: A Stability-Aware Extension of Ensembles with Explainability Guarantees  
**Course:** DA351 – Explainable AI (Project Phase 2)  
**Reference Paper:** *Learning Performance Maximizing Ensembles with Explainability Guarantees* (Vincent Pisztora & Jia Li, AAAI 2024)

---

### 1. Problem Statement
Ensembles with Explainability Guarantees (EEG) allocate individual observations between an intrinsically interpretable glass-box model ($g$) and a high-capacity black-box model ($b$) to guarantee an explainability level $q \in [0, 1]$ while maximizing sufficient predictive performance. Predictions are allocated according to a continuous desirability ranking $r(z)$, with observations satisfying $\hat{r}(x) > 1 - q$ routed to $g$ and the remainder to $b$.

### 2. Research Gap
The original EEG framework assesses allocation quality, task performance, and trade-offs exclusively under static, clean inputs. It leaves an important operational vulnerability uninvestigated: **the robustness of model allocation decisions under small, realistic input perturbations**. Because allocation decisions rely on strict thresholding $\mathbb{I}\{\hat{r}(x) > 1 - q\}$, observations near the boundary frequently undergo "allocation flips"—switching from transparent explanations to opaque black-box predictions (or vice-versa) under negligible feature shifts, even when the original assignment was accurate and sufficient.

### 3. Supporting Literature
- **Routing & Deferral Instability:** Research in conditional computation and selective classification (Dai et al., ACL 2022; Mozannar & Sontag, ICML 2020; Verma & Nalisnick, ICLR 2023) demonstrates that gating functions without smoothness constraints suffer severe routing drift under input shift.
- **Tabular Fragility:** Systematic evaluations by Grinsztajn et al. (NeurIPS 2022) and Borisov et al. (IEEE TNNLS 2022) confirm that tabular predictors are vulnerable to tiny continuous feature perturbations.
- **Novelty:** No prior work in partial-explainability ensembles (EEG, HyRS, HybridCORELS) has audited or enforced allocation stability under test-time perturbations.

### 4. Research Question
> *"How stable are EEG allocation decisions under small, realistic perturbations to input features, and can a stability-aware allocation mechanism reduce unnecessary allocation changes while preserving predictive performance and explainability guarantees?"*

### 5. Proposed Improvement: Stability-Regularized EEG (SR-EEG)
We implement Stability-Regularized EEG (SR-EEG), augmenting allocator training with a local neighborhood consistency penalty on training data:
$$\mathcal{L}_{\text{SR}}(\theta) = \frac{1}{|D_{\text{train}}|} \sum_{i \in D_{\text{train}}} (A_\theta(\Phi(x_i)) - r(z_i))^2 + \lambda_{\text{stab}} \sum_{i \in D_{\text{train}}} (A_\theta(\Phi(x_i + \delta_i)) - r(z_i))^2$$
By Bishop's Theorem (1995), training with noise is mathematically equivalent to Tikhonov regularization, penalizing the Jacobian norm $\|\nabla_x A_\theta(x)\|^2$ and suppressing sharp decision boundary cliffs without exposing any test inputs during training.

### 6. Controlled Experimental Findings
We evaluated Original EEG vs. SR-EEG across multiple perturbation strengths ($\epsilon \in \{0.00, 0.01, 0.02, 0.05, 0.10\}$) over 3 random seeds on the `Wine` and `Phoneme` tabular benchmark datasets:
- **Baseline Allocation Fragility Confirmed:** On clean inputs, both methods achieve $\sim 80.6\%$ accuracy on Wine. However, at $\epsilon = 0.02$ ($q = 0.50$), Original EEG experiences an **8.07% Allocation Flip Rate (AFR)** on Wine and **9.45%** on Phoneme; at $\epsilon = 0.10$, flips surge to **27.93%** and **20.74%** respectively.
- **Substantial Stability Gains:** Proposed SR-EEG reduces AFR to **4.97%** at $\epsilon = 0.02$ on Wine (**38.4% relative reduction**) and to **5.50%** on Phoneme (**41.8% relative reduction**). At $q = 0.20$, flips drop by **73.5%** (from 12.17% to 3.23%).
- **Performance & Explainability Preserved:** SR-EEG maintains nominal explainability quotas ($\hat{q} \approx 0.46$) and achieves equal or superior accuracy under perturbation (79.39% vs. 78.77% on Wine at $\epsilon = 0.02$).

### 7. Contributions and Limitations
- **Contributions:** First empirical audit of allocation instability in partial-explainability ensembles; scientifically grounded SR-EEG formulation with proven Jacobian regularization; validated 35%–73% stability improvement.
- **Limitations:** Focuses on continuous quantitative feature perturbations; categorical perturbation operators and out-of-distribution shifts remain avenues for future research.
