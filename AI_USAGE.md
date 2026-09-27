# AI Usage & Academic Integrity Report

**Course:** Explainable Artificial Intelligence (XAI)  
**Project:** Phase 1 — Research Paper Replication  
**Target Paper:** *"Learning Performance Maximizing Ensembles with Explainability Guarantees"* (AAAI 2024)  
**Student:** <!-- FILL IN: Student Name --> (3rd-Year Data Science & AI Undergraduate)  
**Institution:** <!-- FILL IN: University / Institution -->  
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

---

## 8. Session Log: SuperconductR Completion & Downstream Regeneration (September 27, 2026)

- **Tool / Model Used:** Antigravity IDE (Gemini / Claude via Antigravity Agentic Platform)
- **Prompt Summary:** Complete execution for remaining SuperconductR replicate seeds (seeds 2, 3, 4); address and explicitly document the TabWRN-28 black-box candidate status; regenerate all downstream aggregated artifacts (`phase1_summary.csv`, Tables 1–4, Figures 1–7) to eliminate all `NaN` values without altering other datasets' outputs; and preserve deterministic seed handling.
- **Actions Executed:**
  1. *NumPy 2.x Compatibility Fix:* Resolved `AttributeError: module 'numpy' has no attribute 'trapz'` by adding a backward- and forward-compatible trapezoidal integration helper (`np.trapezoid` / `np.trapz`) in `src/metrics/eeg_metrics.py` and `tests/test_allocator.py`. Verified all 14 unit tests pass.
  2. *SuperconductR Replicate Runs:* Executed seeds 2, 3, and 4 to completion using `run_superconductr_seeds.py` with deterministic seeds (2, 3, 4) from `src/utils/seed.py`, generating raw curve CSVs and summary JSONs in `results/raw/`.
  3. *TabWRN-28 Architecture Decision (Option B):* Verified that `src/models/tab_wrn.py` fully implements the paper's 28-layer Wide ResNet tabular architecture. Explicitly documented in `src/experiments/run_single.py` and `README.md` that TabWRN is excluded from Phase-1 experiment runs due to compute/time constraints, with `GradientBoosting` serving as the evaluated black-box model.
  4. *Downstream Artifact Regeneration:* Executed `src/experiments/generate_final_artifacts.py` over all 25 completed run summaries (5 datasets × 5 seeds), generating `results/processed/phase1_summary.csv`, `tables/table1_dataset_characteristics.csv`, `tables/table2_component_model_performance.csv`, `tables/table3_eeg_metrics_summary.csv`, `tables/table4_paper_vs_replication.csv`, and Figures 1–7 with zero `NaN` values.
- **Human Review & Verification:** Student audited execution logs, verified raw result files in `results/raw/`, and confirmed the before/after difference in `phase1_summary.csv` where standard deviations are now computed over all 5 replicate runs.

---

## 9. Session Log: Documentation Integrity & Technical Discrepancy Analysis (September 28, 2026)

- **Tool / Model Used:** Antigravity IDE (Gemini / Claude via Antigravity Agentic Platform)
- **Prompt Summary:** Fix documentation issues across `README.md`, `docs/phase1_replication_summary.md`, and `AI_USAGE.md`: replace local Windows paths with generic relative paths in the repo structure; verify actual test pass counts across the 4 test files (`test_allocator.py`, `test_metrics.py`, `test_preprocessing.py`, `test_sufficiency.py`); document TabWRN architectural implementation vs experimental exclusion honestly; add a Reference Implementation and Citation Policy subsection; replace name/institution placeholders with standard fill-in comments; and conduct an in-depth technical analysis into metric discrepancies in Table 4 (specifically PCFA and PQEOM).
- **Actions Executed:**
  1. *Repository Tree Normalization:* Updated `README.md` Section 7 to generic relative hierarchy `XAI-Project/` and reflected all 4 test files.
  2. *Automated Test Verification:* Executed `python -m unittest discover tests -v` to confirm 14 passing unit tests across the 4 test suites; updated `README.md` Section 8 installation instructions accordingly.
  3. *TabWRN-28 Scope Documentation:* Added explicit documentation in `README.md` clarifying that TabWRN-28 is architecturally implemented in `src/models/tab_wrn.py` but excluded from Phase-1 evaluation due to compute constraints, with Gradient Boosting serving as the sole evaluated black box.
  4. *Reference Implementation Section:* Added Section 1.1 in `README.md` citing the official repository (`VincentPisztora/Learning-Performance-Maximizing-Ensembles-with-Explainability-Guarantees`), highlighting independent reimplementation, zero verbatim code copying, and strict academic integrity compliance (<20% similarity threshold).
  5. *Placeholder Formatting:* Replaced literal placeholder strings in `docs/phase1_replication_summary.md` and `AI_USAGE.md` with explicit `<!-- FILL IN: ... -->` comments for human entry prior to final submission.
  6. *Technical Discrepancy Analysis:* Conducted an architectural and mathematical investigation of Table 4 discrepancies (notably PCFA and PQEOM). Isolated the PCFA variance to validation discretization and tie-breaking preference (`perf_feat_val >= perf_dist_val` vs strict `>`), verified component ceiling influences on BrazilianHousesR, analyzed Bank trade-off envelope dynamics, and mathematically re-verified all 9 metric definitions against Section 5 of the AAAI 2024 paper. Authored findings in Section 9 of `docs/phase1_replication_summary.md`.
- **Human Review & Verification:** Student audited all markdown changes, verified unit test outputs, and confirmed no changes were made to raw data, tables, or figure artifacts.

---

## 10. Session Log: Phase 1 Deliverables & Reproducibility Polish (September 28, 2026)

- **Tool / Model Used:** Antigravity IDE (Gemini / Claude via Antigravity Agentic Platform)
- **Prompt Summary:** Align repository with course Phase 1 rubric requirements: generate `docs/phase1_summary_1page.md` as a standalone, dense 1-page summary (~500–700 words); verify and pin all package dependencies in `requirements.txt`; add direct OpenML hyperlinks in `README.md`; provide full end-to-end command instructions for faculty evaluation (experiments, aggregation, and figure generation); and update replication comparison tables.
- **Actions Executed:**
  1. *1-Page Deliverable Creation:* Authored `docs/phase1_summary_1page.md` strictly structured to fit one printed page (covering problem statement, OpenML benchmark datasets, EEG methodology, headline reproduced metrics from 5-seed runs, technical explanation of primary PCFA/discrepancies, and overall replication conclusion).
  2. *Dependency Audit:* Cross-checked all imports in `src/` against `requirements.txt` (`numpy`, `scipy`, `pandas`, `scikit-learn`, `tensorflow`, `pyyaml`, `matplotlib`), confirming all packages have explicit pinned version bounds.
  3. *OpenML Dataset Links:* Updated the dataset table in `README.md` with direct hyperlinks to `https://www.openml.org/d/<id>` for all 5 benchmark datasets (44091, 44126, 44133, 44148, 44141).
  4. *Complete Faculty Reproduction Instructions:* Expanded `README.md` Section 9 with explicit step-by-step commands to run single datasets, execute the full 25-run pipeline, and regenerate all processed tables and figures from raw results via `python -m src.experiments.generate_final_artifacts`.
  5. *Replication Comparison Synchronization:* Synchronized `README.md` Section 10 comparison metrics with exact 5-seed means and standard deviations from `tables/table4_paper_vs_replication.csv`.
- **Human Review & Verification:** Student reviewed `docs/phase1_summary_1page.md` for word count and visual density, tested hyperlink destinations, verified `requirements.txt` completeness, and confirmed that raw results in `results/raw/` were unmodified.

---

## 11. Final Session Log: Pre-Submission QA & Four-Session Synthesis (September 28, 2026)

- **Tool / Platform Used:** Antigravity IDE (Gemini 3.8 Flash & Claude via Google DeepMind Antigravity Agentic Platform)
- **Prompt Summary:** Perform final rigorous pre-submission QA audit before September 29 deadline: execute `pytest tests/ -v` test suite, audit and re-verify end-to-end pipeline execution and artifact generation, inspect tables and figures for zero `NaN` occurrences and structural consistency, update `.gitignore` to prevent log noise, clean workspace, and provide a comprehensive multi-session synthesis.
- **Actions Executed:**
  1. *Test Suite Execution:* Verified all 14 tests across the 4 test files (`test_allocator.py`, `test_metrics.py`, `test_preprocessing.py`, `test_sufficiency.py`) pass 100% cleanly in 1.92s with `pytest tests/ -v`.
  2. *Pipeline & Artifact Regeneration Verification:* Confirmed end-to-end execution of `src.experiments.run_phase1` and verified downstream artifact regeneration via `src.experiments.generate_final_artifacts` over all 25 completed runs (5 datasets × 5 replicate seeds).
  3. *Table & Figure Consistency Audit:* Sanity-checked `tables/*.csv`, `results/processed/*.csv`, and `figures/`. Confirmed zero occurrences of `nan`, verified all 5 benchmark datasets appear in every table, and confirmed all 7 publication figures exist and match README documentation.
  4. *Gitignore Polish & Workspace Hygiene:* Updated `.gitignore` to exclude noisy execution logs and model checkpoints while explicitly ensuring graded deliverables (`results/`, `tables/`, `figures/`) remain tracked, and cleaned redundant scratch and temporary files.
- **Four-Session Synthesis & Declaration of Human Oversight:**
  - Across the 4 AI-assisted development sessions (September 25–28, 2026), the role of the AI was strictly delimited to pair-programming, automated execution management, debugging (e.g., resolving NumPy 2.x trapezoidal compatibility), documentation synthesis, and test scaffolding under continuous human direction.
  - No synthetic or fabricated data was introduced; all reported values originate from executed experiments across fixed deterministic seeds (0–4) on verified OpenML benchmarks.
  - All source code, mathematical formulations (desirability scoring, epsilon cutoff, dynamic allocation), and analytical findings were audited, verified, and understood by the student.




