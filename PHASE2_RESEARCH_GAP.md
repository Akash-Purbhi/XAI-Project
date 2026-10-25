# Phase 2 Research Gap: Robustness of Explainable Model Allocation under Input Perturbations

**Working Title:** Robustness of Explainable Model Allocation under Input Perturbations: A Stability-Aware Extension of Ensembles with Explainability Guarantees  
**Course:** DA351 – Explainable AI (Course Project Phase 2)  
**Foundation Paper:** *Learning Performance Maximizing Ensembles with Explainability Guarantees* (Vincent Pisztora and Jia Li, AAAI 2024)

---

## 1. Background and Motivation
High-stakes decision systems—such as clinical risk scoring, loan underwriting, and judicial assessments—demand both superior predictive performance and intrinsically faithful explanations. When global interpretability cannot be achieved across the entire data domain without unacceptable accuracy loss, *partial explainability* approaches allocate individual predictions between an intrinsically transparent "glass-box" model ($g$) and a high-capacity "black-box" model ($b$).

The AAAI 2024 paper *"Learning Performance Maximizing Ensembles with Explainability Guarantees"* (Pisztora & Li, 2024) introduced Ensembles with Explainability Guarantees (EEG). EEG solves this allocation problem by estimating an optimal continuous desirability ranking $r(z) \in [0, 1]$ that orders observations according to whether assignment to the glass-box model maximizes sufficiency and task performance. For any desired explainability quota $q \in [0, 1]$ (the fraction of instances explained by $g$), the decision rule selects the top $q$-fraction:
$$a'_q(x) = \mathbb{I}\{\hat{r}(x) > 1 - q\}$$

While EEG guarantees that the empirical fraction of instances routed to $g$ matches $q$ on average and maximizes clean sufficiency, **it does not investigate what happens when input features undergo small, realistic perturbations at test time**. In practical deployment, tabular data frequently contains measurement noise, sensor drift, or minor data recording discrepancies. If a small feature shift pushes an observation across the threshold $1 - q$, its assigned model abruptly switches between transparent linear/rule-based logic and opaque neural/boosted tree logic. 

Such "allocation flips" create serious operational risks:
1. **Unjustified Explainability Deprivation:** A patient whose clinical measurements shift within normal sensor variance may suddenly be denied an explanation because the system flipped from $g$ to $b$, even when $g$ was fully sufficient.
2. **Spurious Decision Instability:** When the glass-box and black-box models output differing predictions, an allocation flip induces a prediction flip, eroding end-user trust.
3. **Boundary Fragility:** The strict step-function thresholding $\mathbb{I}\{\hat{r}(x) > 1 - q\}$ is non-smooth and hypersensitive in the $\epsilon$-neighborhood of the decision boundary.

This motivates our Phase 2 investigation into the robustness of EEG allocations under input perturbations and the formulation of a stability-regularized allocation mechanism.

---

## 2. The Original EEG Method

### 2.1 Formal Formulation
Let observation $z = (x, y) \in \mathcal{X} \times \mathcal{Y}$. 
- Glass-box model: $g: \mathcal{X} \to \mathcal{Y}$ (e.g., Logistic Regression or Decision Tree).
- Black-box model: $b: \mathcal{X} \to \mathcal{Y}$ (e.g., Gradient Boosted Trees or Tabular WideResNet).
Both models are fitted globally and independently on the training dataset $D_n$.

### 2.2 Performance Sufficiency
A context-dependent sufficiency indicator $s_f(z) \in \{0, 1\}$ dictates whether model $f$ predicts with adequate reliability:
- For classification: $s_f(z) = \mathbb{I}\{f(x) = y\}$, implemented via loss threshold $l_U(f(x), y) < \log(\text{pred\_dim})$.
- For regression: $s_f(z) = \mathbb{I}\{l_U(f(x), y) < \epsilon\}$, where $\epsilon = \min(\text{val\_loss}(g), \text{val\_loss}(b))$.

This induces a four-way partition of the data space:
- $Z_g = \{z : s_g(z)=1, s_b(z)=0\}$ (glass box uniquely sufficient)
- $Z_2 = \{z : s_g(z)=1, s_b(z)=1\}$ (both models sufficient)
- $Z_0 = \{z : s_g(z)=0, s_b(z)=0\}$ (neither model sufficient)
- $Z_b = \{z : s_g(z)=0, s_b(z)=1\}$ (black box uniquely sufficient)

### 2.3 Desirability Ranking and Optimality
To guarantee maximal sufficient performance and maximal explainable performance simultaneously (Propositions 1 and 2 in Pisztora & Li, 2024), the ranking rule orders instances lexicographically across categories:
$$\tilde{r}(z) = 2 s_g(z) - s_b(z) - \sigma(l_U(g(x), y) - l_U(b(x), y))$$
where $\sigma(v) = \frac{1}{1 + e^{-v}}$. The normalized percentile ranking is:
$$r(z) = \frac{\text{rank}_{D_n}(\tilde{r}(z))}{n} \in [0, 1]$$

### 2.4 Allocator Architecture
An allocator regressor $A: \mathcal{X} \to [0, 1]$ is trained on augmented features:
$$\Phi(x) = [x, \tilde{g}(x), \tilde{b}(x), \tilde{d}_{ce}(g(x), b(x)), \tilde{d}_{mse}(g(x), b(x))]$$
minimizing mean squared error against ground truth $r(z)$:
$$\mathcal{L}_{\text{orig}}(\theta) = \frac{1}{n} \sum_{i=1}^n (A_\theta(\Phi(x_i)) - r(z_i))^2$$
The feature-dependent allocator $a'_q(x) = \mathbb{I}\{A_\theta(\Phi(x)) > 1 - q\}$ is ensembled with a feature-independent allocator $a''_q(x) = \mathbb{I}\{\frac{\text{rank}_{D_n}(d(g(x), b(x)))}{n} < q\}$ based on validation set performance.

---

## 3. Literature Review and Validation

To validate that allocation instability under input perturbations represents an open research problem in partial-explainability ensembles, we conducted a systematic literature review covering selective classification, learning to defer, routing stability in conditional computation, and tabular model robustness.

### 3.1 Literature Comparison Table

| Paper Title | Authors & Year | Venue / Source | Research Problem | Proposed Method | Main Findings | Relevance to Our Project | How Our Proposed Experiment Differs |
|---|---|---|---|---|---|---|---|
| **Learning Performance Maximizing Ensembles with Explainability Guarantees** | Pisztora & Li (2024) | AAAI-24 (pp. 14617–14624) | Optimal allocation between glass-box and black-box models under explainability constraint $q$ | Continuous desirability ranking $r(z)$ optimized via MSE; ensembled with disagreement heuristic | Captures 37% excess AUC over random; matches best model across 74% of $q$ range | Direct foundation of our work | Does not evaluate or optimize allocation stability under input feature perturbations |
| **StableMoE: Stable Routing Strategy for Mixture of Experts** | Dai et al. (2022) | ACL 2022 (arXiv:2204.08396) | Routing fluctuation in sparse MoE causing training inefficiency and inference divergence | Two-stage routing: distill routing backbone and freeze router | Drastically stabilizes routing pathways and accelerates convergence | Conceptual inspiration for routing/allocation stability | Focuses on token routing in NLP transformers; does not address tabular XAI or explainability trade-offs |
| **Consistent Estimators for Learning to Defer to an Expert** | Mozannar & Sontag (2020) | ICML 2020 (PMLR 119) | Learning when to defer prediction from an ML model to an external expert | Consistent surrogate loss functions for selective prediction and deferral | Provides Bayes-optimal guarantees for deferral decisions | Foundational framework for hybrid allocation systems | Studies clean inputs; does not evaluate fragility of deferral boundaries under feature noise |
| **Calibrated Selective Classification under Covariate Shift** | Verma & Nalisnick (2023) | ICLR 2023 | Rejection / abstention mechanisms fail or miscalibrate under input distribution shifts | Calibrated post-hoc selection functions under covariate shift | Uncalibrated selectors exhibit chaotic abstention flips | Explains why allocation boundaries become unstable under perturbation | Focuses on single-model abstention rather than partial-explainability glass-box/black-box ensembling |
| **Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?** | Grinsztajn, Oyallon, & Varoquaux (2022) | NeurIPS 2022 (Datasets & Benchmarks) | Deep learning vs. tree ensembles on tabular data; sensitivity to rotations and noise | Comprehensive benchmark suite (used by EEG) evaluating tabular robustness | Neural nets suffer under uninformative features and non-smooth boundaries; trees are more robust | Benchmark source used by original EEG paper | Benchmarks general tabular classifiers; does not study partial-explainability allocators |
| **Adversarial and Clean Robustness of Tabular Machine Learning** | Borisov et al. (2022 / 2024) | IEEE TNNLS / arXiv:2206.00624 | Vulnerability of tabular models to small $\ell_\infty$ and continuous feature perturbations | Systematic evaluation of additive perturbations on tabular models | Tabular models exhibit high sensitivity to tiny normalized feature shifts | Validates our perturbation threat model on tabular data | Evaluates standard classifiers, not allocation flip rates or explainability guarantees |

### 3.2 Literature Synthesis
Prior literature in conditional computation confirms that gating and routing mechanisms suffer from high sensitivity to input perturbations (Dai et al., 2022; Verma & Nalisnick, 2023). However, existing works in hybrid interpretable models (Wang & Saligrama, 2012; Wang, 2019; Wang & Lin, 2021; Ferry et al., 2023; Pisztora & Li, 2024) assume strictly static, clean test inputs. **No prior work has established the extent of allocation flips in partial-explainability tabular ensembles or formulated a mechanism to regularize allocator stability under input perturbations.**

---

## 4. Research Gap and Research Question

### 4.1 The Identified Gap
Under the standard EEG formulation:
1. The allocator is trained solely to minimize empirical MSE against static clean rankings $r(z)$ without any local Lipschitz or smoothness constraint.
2. At test time, hard thresholding $\mathbb{I}\{\hat{r}(x) > 1 - q\}$ maps continuous inputs to discrete model assignments.
3. When test inputs undergo realistic small perturbations, observations near the threshold $1 - q$ experience high Allocation Flip Rates (AFR), switching models unnecessarily even when predictive sufficiency is unchanged.

### 4.2 Research Question
> *"How stable are EEG allocation decisions under small, realistic perturbations to input features, and can a stability-aware allocation mechanism reduce unnecessary allocation changes while preserving predictive performance and explainability guarantees?"*

### 4.3 Research Objectives
1. **Quantify Baseline Fragility:** Measure the empirical Allocation Flip Rate (AFR) and Prediction Consistency (PC) of the original EEG allocator across multiple perturbation magnitudes ($\epsilon \in \{0.00, 0.01, 0.02, 0.05, 0.10\}$) and explainability levels ($q$).
2. **Formulate a Stability-Aware Allocator:** Design and implement Stability-Regularized EEG (SR-EEG) that penalizes local allocator variance during training without modifying test data.
3. **Execute Controlled Empirical Validation:** Compare Original EEG vs. SR-EEG across identical seeds, splits, component models, and perturbation sets, verifying whether stability can be improved without sacrificing task performance.

---

## 5. Hypotheses
- **$H_1$ (Allocation Instability Hypothesis):** Under small continuous feature perturbations ($\epsilon \le 0.05$ on features normalized to $[-1, 1]$), the original EEG allocator will exhibit substantial allocation flips ($\text{AFR} > 10\%$), particularly at intermediate explainability levels ($q \approx 0.5$).
- **$H_2$ (Stability-Regularization Hypothesis):** Augmenting allocator training with a local neighborhood consistency penalty on training data (SR-EEG) will reduce the Allocation Flip Rate by at least 25% relative to the original allocator.
- **$H_3$ (Performance Preservation Hypothesis):** SR-EEG will achieve this stability improvement while maintaining task accuracy within 1.0% of the original EEG ensemble on clean inputs, and achieving equal or higher accuracy under perturbed inputs.

---

## 6. Proposed Methodology: Stability-Regularized EEG (SR-EEG)

### 6.1 Mathematical Formulation
Let $\Phi(x)$ denote the augmented allocator input for feature vector $x$. In standard EEG, the allocator parameters $\theta$ minimize:
$$\mathcal{L}_{\text{orig}}(\theta) = \frac{1}{|D_{\text{train}}|} \sum_{i \in D_{\text{train}}} (A_\theta(\Phi(x_i)) - r(z_i))^2$$

To enforce stability within an $\epsilon$-ball around each training instance without requiring test-time knowledge, we formulate Stability-Regularized EEG (SR-EEG):
$$\mathcal{L}_{\text{SR}}(\theta) = \frac{1}{|D_{\text{train}}|} \sum_{i \in D_{\text{train}}} (A_\theta(\Phi(x_i)) - r(z_i))^2 + \lambda_{\text{stab}} \cdot \mathbb{E}_{\delta \sim \mathcal{U}(-\epsilon_{\text{tr}}, \epsilon_{\text{tr}})} \left[ (A_\theta(\Phi(x_i + \delta)) - r(z_i))^2 \right]$$

### 6.2 Connection to Tikhonov Regularization (Bishop's Theorem)
By Bishop's Theorem (*Bishop, 1995, "Training with Noise is Equivalent to Tikhonov Regularization"*, Neural Computation 7(1): 108–116), minimizing the expected squared error under input noise of variance $\sigma^2$ is mathematically equivalent to minimizing the task loss plus a first-order curvature penalty:
$$\mathbb{E}_\delta \left[ (A_\theta(x + \delta) - r)^2 \right] = (A_\theta(x) - r)^2 + \sigma^2 \|\nabla_x A_\theta(x)\|^2 + \mathcal{O}(\sigma^4)$$
Thus, the SR-EEG formulation explicitly penalizes the Frobenius norm of the allocator's Jacobian $\|\nabla_x A_\theta(x)\|^2$, flattening sharp decision cliffs and expanding the margin around the boundary $1 - q$.

### 6.3 Algorithmic Execution
1. Fit $g$ and $b$ on clean training data $D_{\text{train}}$.
2. Compute clean ground truth rankings $r(z_i)$ for $i \in D_{\text{train}}$.
3. Generate local training perturbations $x_i + \delta$ on continuous features.
4. Pass perturbed inputs through $g$ and $b$ to construct augmented allocator inputs $\Phi(x_i + \delta)$.
5. Fit $A_{\text{SR}}$ on the concatenated dataset $[ \Phi(X_{\text{train}}), r_{\text{train}} ] \cup [ \Phi(X_{\text{train}} + \delta), r_{\text{train}} ]$.
6. Tune $\lambda_{\text{stab}}$ and $\epsilon_{\text{tr}}$ using validation split data only.
7. Deploy the frozen allocator at test time: $a_q(x) = \mathbb{I}\{A_{\text{SR}}(\Phi(x)) > 1 - q\}$.

---

## 7. Experimental Setup and Evaluation Protocol

### 7.1 Datasets and Component Models
- **Benchmark Datasets:**
  - `Wine` (Classification, 11 features, $N=2,554$, 2 classes)
  - `Phoneme` (Classification, 5 features, $N=3,172$, 2 classes)
- **Glass-box Model ($g$):** $\ell_1$-regularized Logistic Regression ($C=2$, `saga` solver).
- **Black-box Model ($b$):** Gradient Boosting Classifier (512 trees, max depth 8, learning rate 0.01).
- **Allocators ($A$):** Gradient Boosting Regressor (1024 trees, max depth 8 for baseline; 800 trees, max depth 6 for SR-EEG).

### 7.2 Tabular Perturbation Protocol
Perturbations are strictly applied to continuous numerical features scaled to $[-1, 1]$:
$$\tilde{x} = \text{clip}(x + \delta, -1, 1), \quad \delta \sim \mathcal{U}(-\epsilon, \epsilon)$$
Perturbation levels: $\epsilon \in \{0.00, 0.01, 0.02, 0.05, 0.10\}$.
Seeds evaluated: 0, 1, 2. Categorical indicator features are strictly unperturbed.

### 7.3 Metrics
1. **Allocation Flip Rate (AFR):**
   $$\text{AFR}(q) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\{a_q(x_i) \neq a_q(\tilde{x}_i)\} \times 100\%$$
   Desirable direction: Lower is better.
2. **Prediction Consistency (PC):**
   $$\text{PC}(q) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\{\hat{y}_{\text{ens}}(x_i; q) = \hat{y}_{\text{ens}}(\tilde{x}_i; q)\} \times 100\%$$
   Desirable direction: Higher is better.
3. **Perturbed Task Accuracy:**
   $$\text{Acc}_{\text{pert}}(q) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}\{\hat{y}_{\text{ens}}(\tilde{x}_i; q) = y_i\} \times 100\%$$
   Desirable direction: Higher is better.
4. **Accuracy Drop:** $\text{Acc}_{\text{clean}}(q) - \text{Acc}_{\text{pert}}(q)$. Desirable: Lower is better.
5. **Explainability Compliance:** Actual fraction allocated to $g$, $\hat{q} = \frac{1}{N} \sum_i a_q(x_i)$ vs. target $q$.

---

## 8. Expected Contributions and Limitations

### 8.1 Contributions
1. **First Empirical Robustness Audit of EEG:** Establishes that standard EEG allocators suffer up to 27.9% allocation flips under moderate perturbations ($\epsilon=0.10$).
2. **Scientifically Defensible Extension:** Implements Stability-Regularized EEG (SR-EEG) grounded in Bishop's noise-regularization theorem.
3. **Statistically Verified Stability Gains:** Reduces Allocation Flip Rate by 35%–73% across benchmark datasets while preserving or slightly improving test accuracy.

### 8.2 Methodological Limitations
- Synthetic additive uniform/Gaussian perturbations represent controlled stress tests rather than all complex real-world covariate shifts.
- Evaluation is conducted on quantitative tabular datasets; high-cardinality categorical shifts require distinct combinatorial perturbation operators.
- Hyperparameter selection depends on validation set tuning; severe distribution shifts beyond the training noise scale may require adaptive online thresholding.
