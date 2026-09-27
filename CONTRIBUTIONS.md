# Team Contribution & Responsibility Log

**Course:** Explainable Artificial Intelligence (XAI)  
**Project:** Phase 1 — Research Paper Replication  
**Target Paper:** *"Learning Performance Maximizing Ensembles with Explainability Guarantees"* (AAAI 2024)  
**Date:** September 2026  

---

## 1. Team Members & Roles

| Member Name | Student ID | Academic Email | Primary Project Role |
|:------------|:-----------|:---------------|:---------------------|
| <!-- FILL IN: Member 1 Name --> | <!-- FILL IN: ID --> | <!-- FILL IN: Email --> | <!-- FILL IN: e.g., Model & Allocator Lead --> |
| <!-- FILL IN: Member 2 Name --> | <!-- FILL IN: ID --> | <!-- FILL IN: Email --> | <!-- FILL IN: e.g., Data Preprocessing & Validation Lead --> |
| <!-- FILL IN: Member 3 Name --> | <!-- FILL IN: ID --> | <!-- FILL IN: Email --> | <!-- FILL IN: e.g., Evaluation Metrics & Testing Lead --> |
| <!-- FILL IN: Member 4 Name --> | <!-- FILL IN: ID --> | <!-- FILL IN: Email --> | <!-- FILL IN: e.g., Documentation & Analysis Lead --> |

---

## 2. Division of Responsibilities

### Module A: Data Ingestion, Benchmark Curation & Preprocessing
- **Owner(s):** <!-- FILL IN: Team Member Name -->
- **Deliverables:**
  - Automated OpenML retrieval of the 5 Grinsztajn benchmark datasets (IDs: 44091, 44126, 44133, 44148, 44141).
  - Stratified 70/9/21 partitioning without cross-split data leakage.
  - Strict train-only fitting for numerical standard scaling and categorical one-hot encoders.

### Module B: Component Model Implementations & Tuning
- **Owner(s):** <!-- FILL IN: Team Member Name -->
- **Deliverables:**
  - Glass-box model wrappers (Logistic Regression, Lasso, Decision Trees).
  - Black-box GBT candidates and 4-fold cross-validation grid search tuning.
  - Implementation and architectural audit of TabWRN-28 in `src/models/tab_wrn.py`.

### Module C: Allocation Mechanism & Sufficiency Scoring
- **Owner(s):** <!-- FILL IN: Team Member Name -->
- **Deliverables:**
  - Mathematical formulation of classification and regression sufficiency indicators ($s_f(z)$ and $\varepsilon$ cutoff).
  - Continuous desirability scoring $\tilde{r}(z)$ and normalized ranking.
  - Feature augmentation $[x, g(x), b(x), d_{\text{CE}}, d_{\text{MSE}}]$ and dynamic allocator selection ($a'_q$ vs $a''_q$).

### Module D: Evaluation Metrics, Testing & Verification
- **Owner(s):** <!-- FILL IN: Team Member Name -->
- **Deliverables:**
  - Implementation of all 9 AAAI 2024 paper metrics (AUC, PPCR, PQEOM, PQOM, PCFA, 95TQM, Max Acc, Argmax $q$, $s$ Acc).
  - Automated unit test suite (14 passing tests across 4 test modules).
  - NumPy 2.x integration fix and cross-version compatibility.

### Module E: Replication Reporting & Academic Integrity
- **Owner(s):** <!-- FILL IN: Team Member Name -->
- **Deliverables:**
  - 1-page standalone replication deliverable (`docs/phase1_summary_1page.md`).
  - Comprehensive replication analysis and discrepancy breakdown (`docs/phase1_replication_summary.md`).
  - Academic integrity compliance auditing and session logging (`AI_USAGE.md`).

---

## 3. Team Sign-off & Verification

We confirm that all team members contributed actively to the codebase, reviewed experimental findings, and understand the replicated mathematical formulations.

- **Signature / Date:** <!-- FILL IN: Signature and Date -->
