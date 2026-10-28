# AI Usage Disclosure (AI_USAGE.md)

**Course:** DA351 – Explainable AI (Project Phase 2)  
**Project:** Robustness of Explainable Model Allocation under Input Perturbations: A Stability-Aware Extension of Ensembles with Explainability Guarantees  
**Academic Integrity Compliance:** In accordance with the course policy on AI tools, this document discloses all AI assistance utilized during the Phase 2 research project.

---

## 1. AI Tools Used
- **Tool:** Antigravity (Google DeepMind Advanced Agentic AI Assistant)
- **Environment:** Windows, PowerShell, Conda (`env_eeg` with Python 3.8.8, TensorFlow 2.3.0, Scikit-learn 1.0.2).

---

## 2. Purpose of Using the Tool
The AI assistant was used as a pair programming research engineer and experimental assistant to:
1. Conduct an audit of the original AAAI 2024 repository code structure.
2. Search and compile relevant academic literature on routing stability, selective classification, and tabular model robustness.
3. Formulate the mathematical optimization objective for Stability-Regularized EEG (SR-EEG).
4. Assist in implementing the evaluation metrics, unit tests, and perturbation harness.
5. Generate publication-quality 300 DPI comparative visualizations from saved experimental CSV outputs.
6. Assist in drafting academic reports conforming to formal scientific writing guidelines.

---

## 3. Important Prompts and Instructions Used
Key instruction prompts issued to the AI tool included:
- *"Audit the repository, inspect data loaders, models, and pipeline steps 1 to 4, and identify missing artifacts or hardcoded paths."*
- *"Formulate a literature comparison table validating the novelty of allocation stability under input perturbations in partial explainability ensembles."*
- *"Implement the missing EEG evaluation metrics (PPCR, PQEOM, 95TQM, Max Acc, s Acc) and create unit tests to verify calculations."*
- *"Design a tabular perturbation protocol that strictly perturbs continuous numerical features in [-1, 1] while preserving categorical indicators."*
- *"Implement Stability-Regularized EEG (SR-EEG) using training data perturbations only, ensuring zero test data leakage."*
- *"Run controlled before-versus-after experiments comparing Original EEG vs. SR-EEG across epsilons [0.00, 0.01, 0.02, 0.05, 0.10] and random seeds [0, 1, 2], and export summary CSVs and multi-panel figures."*

---

## 4. Code and Text Generated with AI Assistance
The following files were created or modified with AI collaboration:
1. **[utils.py](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/utils.py)**: Configured safe local results directory.
2. **[evaluate_eeg.py](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/evaluate_eeg.py)**: Module for original EEG metrics and allocation stability metrics (AFR, PC).
3. **[test_evaluate_eeg.py](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/test_evaluate_eeg.py)**: Unit test suite for evaluation metrics.
4. **[perturbation_utils.py](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/perturbation_utils.py)**: Scale-aware tabular perturbation engine.
5. **[experiment_stability.py](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/experiment_stability.py)**: End-to-end controlled before-versus-after experiment runner.
6. **[generate_combined_figures.py](file:///d:/python%20college/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees-main/generate_combined_figures.py)**: Multi-dataset comparative publication figure generator.
7. Documentation files: `PHASE2_RESEARCH_GAP.md`, `PHASE2_RESULTS.md`, `PHASE2_ONE_PAGE_SUMMARY.md`.

---

## 5. Review, Modification, Testing, and Human Validation
Every AI-assisted output was reviewed and validated as follows:
- **Code Execution & Error Debugging:** AI-suggested code was executed live in `env_eeg`. When bugs arose (such as missing `import os` in step 4 or directory scanning errors in step 3), the root cause was identified, fixed, and verified.
- **Mathematical Integrity:** The SR-EEG loss formulation was checked against Bishop's (1995) Tikhonov regularization theorem to guarantee theoretical validity.
- **Data Leakage Auditing:** Confirmed that training-time noise was applied solely to `X_train` and that all test-set evaluation was performed independently without tuning on test perturbations.
- **Metric Verification:** All evaluation formulas were cross-referenced with the AAAI 2024 paper definitions and tested using unit tests with analytic ground-truth values in `test_evaluate_eeg.py`.
- **Text Review:** All written sections were reviewed to ensure formal academic tone, original paraphrasing, and avoidance of unsupported claims.
