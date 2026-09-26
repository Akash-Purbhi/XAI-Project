# AI Usage & Academic Integrity Report

**Course:** Explainable Artificial Intelligence (XAI)  
**Project:** Phase 1 — Research Paper Replication  
**Target Paper:** *"Learning Performance Maximizing Ensembles with Explainability Guarantees"* (AAAI 2024)  
**Student:** 3rd-Year Data Science & AI Undergraduate  
**Date:** September 2026  

---

## 1. Declarative Statement of AI Utilization

In compliance with academic integrity guidelines and institutional policies governing artificial intelligence in university coursework, this document provides a comprehensive, transparent audit of the AI tools, models, prompts, and workflows utilized during Phase 1 of this project.

The development process utilized pair-programming and coding assistance from Google DeepMind's Antigravity agentic coding platform. The role of AI was strictly delimited to implementation assistance, automated boilerplate generation, refactoring, unit test generation, and computational execution management under direct student guidance and critical peer review.

---

## 2. Tools & Models Employed

| Tool / Platform | Model(s) | Role & Scope of Use |
|:----------------|:---------|:-------------------|
| **Antigravity IDE** | Claude 3.5 Sonnet / Gemini 1.5 Pro | Agentic code editing, terminal command execution, filesystem inspection |
| **Python Tooling** | Scikit-Learn 1.0.2, TensorFlow 2.3.0 | Underlying scientific computing libraries executed via Conda (`env_eeg`) |
| **OpenML Python API** | OpenML Benchmark Suite | Automated data retrieval of verified benchmarks (44091, 44126, 44133, 44148, 44141) |

---

## 3. Scope of AI Involvement vs. Human Oversight

```
+---------------------------------------------------------------------------------+
|                                WORKFLOW AUDIT                                   |
+---------------------------------------------------------------------------------+
| Project Stage                     | AI Contribution     | Human / Student Role  |
+-----------------------------------+---------------------+-----------------------+
| 1. Paper Analysis & Design        | Summarized math     | Verified equations    |
|                                   | and algorithm steps | against AAAI 2024 text|
+-----------------------------------+---------------------+-----------------------+
| 2. Modular Architecture Design    | Drafted directory   | Enforced strict       |
|                                   | & module hierarchy  | Phase 1 constraints   |
+-----------------------------------+---------------------+-----------------------+
| 3. Data Ingestion & Preprocessing | Implemented loaders | Audited for zero data |
|                                   | and split logic     | leakage across splits |
+-----------------------------------+---------------------+-----------------------+
| 4. Model & Allocator Code         | Scikit-learn & TF   | Reviewed TabWRN depth |
|                                   | wrapper classes     | & epsilon regression  |
+-----------------------------------+---------------------+-----------------------+
| 5. Metric Calculations (9 metrics)| Implemented math    | Hand-verified edge    |
|                                   | for AUC, PPCR, etc. | cases via unit tests  |
+-----------------------------------+---------------------+-----------------------+
| 6. Experiment Execution           | Ran background CLI  | Monitored GPU/CPU load|
|                                   | scripts across seeds| and verified outputs  |
+-----------------------------------+---------------------+-----------------------+
| 7. Documentation & Reporting      | Drafted initial     | Reviewed, formatted,  |
|                                   | Markdown structure  | and finalized text    |
+-----------------------------------+---------------------+-----------------------+
```

---

## 4. Prompt Engineering & Interaction Log Summary

Key prompt patterns and instructions directed at the agentic AI during development:

1. **Strict Methodological Boundary Enforcement:**
   > *"This is ONLY PHASE 1 of the project. DO NOT implement Phase 2. DO NOT propose or implement a research extension. DO NOT add unnecessary XAI techniques such as SHAP, LIME, PDP, Grad-CAM, etc. The selected research paper uses intrinsic explainability / glass-box models and an allocation mechanism called EEG."*
   - *Outcome:* Prevented scope creep, avoided hallucinated features, and maintained fidelity to the AAAI 2024 paper.

2. **Data Leakage & Preprocessing Rigor:**
   > *"Ensure all encoders and scalers fit ONLY on training sets and transform validation/test sets to prevent any data leakage. Verify all 5 OpenML dataset IDs match the Grinsztajn benchmark."*
   - *Outcome:* Ensured valid train/val/test splits (70/9/21) without statistical leakage.

3. **TabWRN Architecture Faithfulness:**
   > *"Faithfully replicate the 28-layer TabWRN architecture using Dense tabular layers instead of 2D convolutions, with pre-activation blocks (BatchNorm -> ReLU -> Dropout -> Dense)."*
   - *Outcome:* Generated a pure Keras/TensorFlow 2.3 implementation matching the paper's specification.

4. **Nine Evaluation Metrics Fidelity:**
   > *"Implement the exact 9 metrics reported in Table 2 of the paper: AUC, PPCR, PQEOM, PQOM, PCFA, 95TQM, Max_Acc, Argmax_q, and s_Acc."*
   - *Outcome:* Recreated the exact mathematical formulations with zero approximation shortcuts.

---

## 5. Verification & Validation Protocol

To ensure code correctness and eliminate AI hallucinations or subtle software bugs:

1. **Automated Unit Testing Suite:**
   - Authored 14 standalone unit tests (`tests/test_eeg_pipeline.py`) testing data loading, stratified splits, sufficiency calculations, continuous desirability rankings, allocator predictions, and all 9 metric functions.
   - Result: 100% pass rate (`14 passed in 3.12s`).
2. **Deterministic Seed Control:**
   - Fixed random seeds (0 through 4) across NumPy, Scikit-Learn, TensorFlow, and Python's `random` module to ensure exact reproducibility across runs.
3. **Execution Ground Truth:**
   - All results, figures, and comparison tables are generated directly from executed experiment artifacts in `results/raw/` and `results/processed/`. No values are hardcoded or fabricated.

---

## 6. Known Limitations & Corrective Human Interventions

During the project, several agent-generated drafts required explicit human correction:

1. **Grid Search Computational Explosion:**
   - *Issue:* The initial grid search configuration for Gradient Boosting Regressor spawned over 200 combinations per fold on large datasets (SuperconductR with 15k samples and 79 features), threatening CPU exhaustion.
   - *Correction:* Reduced hyperparameter grid bounds to standard representative values while retaining 4-fold cross-validation.
2. **Missing Dependency Guard:**
   - *Issue:* An initial draft attempted to import LightGBM, which was not strictly required by the core paper models and not installed in `env_eeg`.
   - *Correction:* Removed LightGBM references and standardized on `scikit-learn` GBT.
3. **Typographical Syntax Error:**
   - *Issue:* A typo `Tuple_df` occurred in `aggregation.py`.
   - *Correction:* Fixed import to `Tuple[pd.DataFrame, ...]` from `typing`.

---

## 7. Compliance Statement

I confirm that:
- I understand the underlying mathematics, code logic, and experimental outcomes of this replication.
- All code generated with AI assistance was audited, executed, and tested by me.
- The work presented adheres to university academic ethics, intellectual honesty, and responsible AI use guidelines.
