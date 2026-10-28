# -*- coding: utf-8 -*-
"""
Script 06: Meta-Explainability of the EEG Allocator using SHAP (Gap 6)
"Who explains the Explainer?"

Computes TreeSHAP attributions on the EEG Allocator to answer:
1. Feature Attribution: Which features drive sample routing to Glass-Box vs Black-Box?
2. Confidence vs. Domain Features: Does the allocator rely more on intrinsic raw features (X)
   or model disagreement / uncertainty (d_gb_bb, y_hat)?
3. Robustness Shift (Gap 3 + Gap 6 Synergy): How does Stability-Robust training (SR-EEG)
   shift the allocator's feature importance compared to Standard EEG?
4. Local Case Studies: Generates Waterfall explanations for:
   - High-confidence Glass-box routing
   - Boundary contention routing
   - High-confidence Black-box routing
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shap
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from data import get_dataset, scale_to_minus_one_to_one
from utils_2 import get_percentiles_of_a_wrt_b
from perturbation_utils import generate_tabular_perturbations

# Import model factory from script 03
import importlib.util
spec3 = importlib.util.spec_from_file_location("stab_exp", str(PROJECT_ROOT / "scripts" / "03_run_stability_experiment.py"))
stab_exp = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(stab_exp)
build_allocator_features = stab_exp.build_allocator_features
train_tree_model = stab_exp.train_tree_model


def get_allocator_feature_names(feature_cols, problem_type='classification', pred_dim=2):
    """
    Constructs human-interpretable names for all augmented allocator input features.
    """
    names = list(feature_cols)
    if problem_type == 'classification':
        for c in range(pred_dim):
            names.append(f"GB_proba_cls_{c}")
        for c in range(pred_dim):
            names.append(f"BB_proba_cls_{c}")
        names.append("Disagreement_CrossEntropy")
        names.append("Disagreement_ProbMSE")
    else:
        names.append("GB_pred_y")
        names.append("BB_pred_y")
        names.append("Disagreement_MSE")
        names.append("Disagreement_AbsDiff")
    return names


def run_shap_analysis(dataset_name="Wine", use_gpu=True, max_display=12):
    print("=" * 75)
    print(f"RUNNING GAP 6 META-EXPLAINABILITY (SHAP) ON ALLOCATOR: {dataset_name}")
    print("=" * 75)

    out_dir = Path(f"results/shap/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    pred_dim, ds_dict = get_dataset(dataset_name=dataset_name)
    problem_type = ds_dict.get('problem_type')
    cat_indicator = ds_dict.get('categorical_features_indicator_dict', {}).get('categorical_features_indicator_list', [])

    ds_sklearn = ds_dict['ds_sklearn']
    X_train_df = ds_sklearn['ds_train']['x_train']
    y_train_raw = ds_sklearn['ds_train']['y_train']
    X_val_df = ds_sklearn['ds_val']['x_val']
    y_val_raw = ds_sklearn['ds_val']['y_val']
    X_test_df = ds_sklearn['ds_test']['x_test']
    y_test_raw = ds_sklearn['ds_test']['y_test']

    feature_cols = list(X_train_df.columns)
    alloc_feature_names = get_allocator_feature_names(feature_cols, problem_type, pred_dim)

    # 2. Fit Component Models
    print("\n[Step 1] Fitting Component Models g & b...")
    if problem_type == 'classification':
        gb_model = LogisticRegression(C=2, solver='saga', penalty='l1', max_iter=5000, random_state=0)
        gb_model.fit(X_train_df, y_train_raw)

        bb_model = train_tree_model('classifier', 'classification', use_gpu, lr=0.01, max_depth=8, n_estimators=512, subsample=0.8)
        bb_model.fit(X_train_df, y_train_raw)

        y_hat_gb_train = gb_model.predict_proba(X_train_df)
        y_hat_gb_test = gb_model.predict_proba(X_test_df)
        y_hat_bb_train = bb_model.predict_proba(X_train_df)
        y_hat_bb_test = bb_model.predict_proba(X_test_df)

        y_train_onehot = np.eye(pred_dim, dtype=np.float32)[y_train_raw.astype(int)]
        loss_gb_tr = np.sum(-y_train_onehot * np.log(np.maximum(np.minimum(y_hat_gb_train, 1 - 1e-7), 1e-7)), axis=1, keepdims=True)
        loss_bb_tr = np.sum(-y_train_onehot * np.log(np.maximum(np.minimum(y_hat_bb_train, 1 - 1e-7), 1e-7)), axis=1, keepdims=True)
        loss_cutoff = np.log(pred_dim)

        gb_c = (loss_gb_tr < loss_cutoff).astype(np.float32)
        bb_c = (loss_bb_tr < loss_cutoff).astype(np.float32)
        gb_desirability = 2.0 * gb_c - bb_c - 1.0 / (1.0 + np.exp(loss_bb_tr - loss_gb_tr))
    else:
        gb_model = Lasso(alpha=0.01, max_iter=5000, random_state=0)
        gb_model.fit(X_train_df, y_train_raw)

        bb_model = train_tree_model('regressor', 'regression', use_gpu, lr=0.01, max_depth=8, n_estimators=512, subsample=0.8)
        bb_model.fit(X_train_df, y_train_raw)

        y_hat_gb_train = gb_model.predict(X_train_df).reshape(-1, 1)
        y_hat_gb_test = gb_model.predict(X_test_df).reshape(-1, 1)
        y_hat_bb_train = bb_model.predict(X_train_df).reshape(-1, 1)
        y_hat_bb_test = bb_model.predict(X_test_df).reshape(-1, 1)

        loss_gb_tr = (y_train_raw.reshape(-1, 1) - y_hat_gb_train) ** 2
        loss_bb_tr = (y_train_raw.reshape(-1, 1) - y_hat_bb_train) ** 2
        cutoff_reg = np.median(loss_bb_tr)

        gb_c = (loss_gb_tr < cutoff_reg).astype(np.float32)
        bb_c = (loss_bb_tr < cutoff_reg).astype(np.float32)
        gb_desirability = 2.0 * gb_c - bb_c - (loss_gb_tr - loss_bb_tr) / (np.std(loss_bb_tr) + 1e-7)

    r_train = get_percentiles_of_a_wrt_b(b=gb_desirability, a=None, distr_min=-2.0, distr_max=2.0)

    # Scalers
    _, mn_gb, mx_gb = scale_to_minus_one_to_one(y_hat_gb_train)
    _, mn_bb, mx_bb = scale_to_minus_one_to_one(y_hat_bb_train)
    if problem_type == 'classification':
        d_ce = -np.sum(y_hat_bb_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        d_mse = np.sum((y_hat_bb_train - y_hat_gb_train) ** 2, axis=1, keepdims=True)
        d_init = np.hstack([d_ce, d_mse])
    else:
        d_mse = (y_hat_bb_train - y_hat_gb_train) ** 2
        d_abs = np.abs(y_hat_bb_train - y_hat_gb_train)
        d_init = np.hstack([d_mse, d_abs])
    _, mn_d, mx_d = scale_to_minus_one_to_one(d_init)

    X_alloc_train, _ = build_allocator_features(X_train_df, y_hat_gb_train, y_hat_bb_train, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)
    X_alloc_test, _ = build_allocator_features(X_test_df, y_hat_gb_test, y_hat_bb_test, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)

    # 3. Fit Standard Allocator & SR-EEG Allocator (using original paper's GBR for 100% TreeExplainer compatibility)
    print("\n[Step 2] Training Baseline and Stability-Robust Allocators (GBR)...")
    n_feats = len(feature_cols)
    if n_feats > 100:
        # High-dimensional feature spaces (e.g., Isolet_R with 617 features)
        n_est_orig, depth_orig = 150, 5
        n_est_sr, depth_sr = 150, 4
        sample_size = min(150, len(X_alloc_test))
    else:
        n_est_orig, depth_orig = 400, 8
        n_est_sr, depth_sr = 400, 6
        sample_size = min(500, len(X_alloc_test))

    alloc_orig = GradientBoostingRegressor(learning_rate=0.01, max_depth=depth_orig, n_estimators=n_est_orig, subsample=0.25, random_state=0)
    alloc_orig.fit(X_alloc_train, r_train.ravel())

    X_tr_pert_1 = generate_tabular_perturbations(X_train_df, cat_indicator, epsilon=0.02, seed=10)
    X_tr_pert_2 = generate_tabular_perturbations(X_train_df, cat_indicator, epsilon=0.04, seed=20)
    if problem_type == 'classification':
        y_gb_p1 = gb_model.predict_proba(X_tr_pert_1)
        y_bb_p1 = bb_model.predict_proba(X_tr_pert_1)
        y_gb_p2 = gb_model.predict_proba(X_tr_pert_2)
        y_bb_p2 = bb_model.predict_proba(X_tr_pert_2)
    else:
        y_gb_p1 = gb_model.predict(X_tr_pert_1).reshape(-1, 1)
        y_bb_p1 = bb_model.predict(X_tr_pert_1).reshape(-1, 1)
        y_gb_p2 = gb_model.predict(X_tr_pert_2).reshape(-1, 1)
        y_bb_p2 = bb_model.predict(X_tr_pert_2).reshape(-1, 1)

    X_alloc_p1, _ = build_allocator_features(X_tr_pert_1, y_gb_p1, y_bb_p1, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)
    X_alloc_p2, _ = build_allocator_features(X_tr_pert_2, y_gb_p2, y_bb_p2, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)
    X_alloc_train_sr = np.vstack([X_alloc_train, X_alloc_p1, X_alloc_p2])
    r_train_sr = np.concatenate([r_train.ravel(), r_train.ravel(), r_train.ravel()])

    alloc_sr = GradientBoostingRegressor(learning_rate=0.01, max_depth=depth_sr, n_estimators=n_est_sr, subsample=0.50, random_state=0)
    alloc_sr.fit(X_alloc_train_sr, r_train_sr)

    # 4. Compute SHAP Values on Test Observations
    print(f"\n[Step 3] Computing TreeSHAP Values on Test Set (evaluating {sample_size} samples)...")
    # Wrap test data into a clean DataFrame with column names
    X_alloc_test_df = pd.DataFrame(X_alloc_test, columns=alloc_feature_names)
    X_eval = X_alloc_test_df.iloc[:sample_size]

    explainer_orig = shap.TreeExplainer(alloc_orig)
    shap_vals_orig = explainer_orig(X_eval)

    explainer_sr = shap.TreeExplainer(alloc_sr)
    shap_vals_sr = explainer_sr(X_eval)

    # 5. Global Meta-Explanations: Beeswarm & Bar Plots
    print("\n[Step 4] Generating Global Summary & Beeswarm Visualizations...")
    # Summary Beeswarm Plot - Baseline
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_vals_orig, max_display=max_display, show=False)
    plt.title(f"EEG Allocator SHAP Summary (Original) - {dataset_name}", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm_original.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Summary Beeswarm Plot - SR-EEG
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_vals_sr, max_display=max_display, show=False)
    plt.title(f"SR-EEG Allocator SHAP Summary (Robust) - {dataset_name}", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm_sreeg.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Comparative Feature Importance: Standard vs SR-EEG Attribution Shift
    mean_abs_orig = np.mean(np.abs(shap_vals_orig.values), axis=0)
    mean_abs_sr = np.mean(np.abs(shap_vals_sr.values), axis=0)

    df_importance = pd.DataFrame({
        'Feature': alloc_feature_names,
        'Original_Mean_SHAP': mean_abs_orig,
        'SREEG_Mean_SHAP': mean_abs_sr,
        'Attribution_Shift': mean_abs_sr - mean_abs_orig
    }).sort_values(by='Original_Mean_SHAP', ascending=False)
    df_importance.to_csv(out_dir / "shap_importance_comparison.csv", index=False)

    # Comparative Bar Plot
    top_k = df_importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_k))
    width = 0.35

    ax.barh(y_pos - width/2, top_k['Original_Mean_SHAP'], width, label='Original Allocator', color='#d9534f', alpha=0.85)
    ax.barh(y_pos + width/2, top_k['SREEG_Mean_SHAP'], width, label='SR-EEG Allocator (Robust)', color='#2b6cb0', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_k['Feature'], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Allocation Routing)", fontsize=11)
    ax.set_title(f"Top Routing Drivers: Original vs. SR-EEG - {dataset_name}", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance_shift.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Local Case Studies (Waterfall Explanations for 3 Canonical Decisions)
    print("\n[Step 5] Generating Local Case Studies (Waterfall Explanations)...")
    alloc_scores = alloc_orig.predict(X_eval.to_numpy())
    sorted_indices = np.argsort(alloc_scores)

    # Case A: Strong Glass-Box assignment (lowest score)
    idx_gb = sorted_indices[0]
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_vals_orig[idx_gb], max_display=8, show=False)
    plt.title(f"Case A: High-Confidence Glass-Box Routing (Score={alloc_scores[idx_gb]:.3f})", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / "case_study_A_glass_box.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Case B: Boundary decision (score closest to 0.5)
    idx_boundary = np.argmin(np.abs(alloc_scores - 0.5))
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_vals_orig[idx_boundary], max_display=8, show=False)
    plt.title(f"Case B: Boundary Contention Routing (Score={alloc_scores[idx_boundary]:.3f})", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / "case_study_B_boundary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Case C: Strong Black-Box assignment (highest score)
    idx_bb = sorted_indices[-1]
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_vals_orig[idx_bb], max_display=8, show=False)
    plt.title(f"Case C: High-Confidence Black-Box Routing (Score={alloc_scores[idx_bb]:.3f})", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(out_dir / "case_study_C_black_box.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Meta-Feature Share Analysis: Domain vs Uncertainty Disagreement
    # Split features into Domain (X) vs Meta-Confidence (GB/BB predictions + Disagreement)
    domain_mask = [f in feature_cols for f in alloc_feature_names]
    meta_mask = [not m for m in domain_mask]

    domain_importance_pct = float(np.sum(mean_abs_orig[domain_mask]) / np.sum(mean_abs_orig) * 100.0)
    meta_importance_pct = float(np.sum(mean_abs_orig[meta_mask]) / np.sum(mean_abs_orig) * 100.0)

    domain_sr_pct = float(np.sum(mean_abs_sr[domain_mask]) / np.sum(mean_abs_sr) * 100.0)
    meta_sr_pct = float(np.sum(mean_abs_sr[meta_mask]) / np.sum(mean_abs_sr) * 100.0)

    summary_meta = {
        'Dataset': dataset_name,
        'Problem_Type': problem_type,
        'Original_Domain_Feature_Share': domain_importance_pct,
        'Original_Disagreement_Confidence_Share': meta_importance_pct,
        'SREEG_Domain_Feature_Share': domain_sr_pct,
        'SREEG_Disagreement_Confidence_Share': meta_sr_pct,
        'Top_Driver_1': df_importance.iloc[0]['Feature'],
        'Top_Driver_2': df_importance.iloc[1]['Feature'],
        'Top_Driver_3': df_importance.iloc[2]['Feature']
    }
    df_meta = pd.DataFrame([summary_meta])
    df_meta.to_csv(out_dir / "meta_share_summary.csv", index=False)

    print("\n" + "=" * 75)
    print(f"SHAP ANALYSIS COMPLETE FOR {dataset_name}!")
    print(f" -> Allocator relies {meta_importance_pct:.1f}% on Model Disagreement / Predictions and {domain_importance_pct:.1f}% on Raw Domain Features.")
    print(f" -> Top 3 Routing Drivers: {df_importance.iloc[0]['Feature']}, {df_importance.iloc[1]['Feature']}, {df_importance.iloc[2]['Feature']}")
    print(f" -> Plots saved to: {out_dir}")
    print("=" * 75)
    return summary_meta


def main():
    parser = argparse.ArgumentParser(description="Analyze Allocator via SHAP (Gap 6)")
    parser.add_argument('--dataset_name', type=str, default='Wine', help="Dataset to analyze")
    parser.add_argument('--use_gpu', action='store_true', default=True, help="Use GPU acceleration")
    parser.add_argument('--no_gpu', dest='use_gpu', action='store_false', help="Force CPU mode")
    args = parser.parse_args()

    run_shap_analysis(dataset_name=args.dataset_name, use_gpu=args.use_gpu)


if __name__ == "__main__":
    main()
