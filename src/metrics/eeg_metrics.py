"""
EEG evaluation metrics strictly implementing the AAAI 2024 paper definitions:

1. AUC: Area under the performance-vs-explainability curve
2. PPCR: Percentage Performance Captured over Random
3. PQEOM: Percent Q Equal or Over Max
4. PQOM: Percent Q Over Max
5. PCFA: Percent Contribution of Feature-dependent Allocator
6. 95TQM: 95% Threshold Q Max
7. Max Acc / Max Performance: Maximum performance across all q
8. Argmax q: Highest q at which maximum performance is maintained
9. Sufficiency Accuracy (s Acc): Accuracy of estimating the 4 sufficiency categories
"""

from typing import Dict, List, Optional
import numpy as np


def compute_auc(perf_curve: np.ndarray, q_grid: np.ndarray) -> float:
    """
    Compute area under the curve using trapezoidal rule.
    Returns value in [0, 1].
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(perf_curve, q_grid))
    return float(np.trapz(perf_curve, q_grid))


def compute_eeg_metrics(
    perf_eeg: np.ndarray,
    perf_random: np.ndarray,
    perf_oracle: np.ndarray,
    perf_gb: float,
    perf_bb: float,
    q_grid: np.ndarray,
    pcfa_choices: List[float],
    sufficiency_accuracy: float,
) -> Dict[str, float]:
    """
    Compute all 9 EEG metrics reported in Table 2 of the paper.
    All returned values are formatted as percentages (0 to 100).
    
    Args:
        perf_eeg: Array of EEG sufficient performance on test set across q_grid
        perf_random: Array of random baseline performance across q_grid
        perf_oracle: Array of oracle upper-bound performance across q_grid
        perf_gb: Glass-box model standalone test performance
        perf_bb: Black-box model standalone test performance
        q_grid: Array of q values in [0, 1]
        pcfa_choices: Binary list indicating whether a'_q was chosen for each q
        sufficiency_accuracy: Accuracy of predicting sufficiency categories (in [0, 100])
        
    Returns:
        metrics_dict: Dictionary containing all 9 metrics as percentages
    """
    q_arr = np.asarray(q_grid, dtype=np.float32)
    eeg = np.asarray(perf_eeg, dtype=np.float32)
    rand = np.asarray(perf_random, dtype=np.float32)
    orac = np.asarray(perf_oracle, dtype=np.float32)

    # 1. AUC (as percentage)
    auc_eeg = compute_auc(eeg, q_arr)
    auc_rand = compute_auc(rand, q_arr)
    auc_orac = compute_auc(orac, q_arr)

    # 2. PPCR
    denom = auc_orac - auc_rand
    if denom > 1e-7:
        ppcr = float((auc_eeg - auc_rand) / denom * 100.0)
    else:
        ppcr = 0.0

    # 3. PQEOM: % of q values where EEG >= max(perf(g), perf(b))
    max_indiv = max(perf_gb, perf_bb)
    pqeom = float(np.mean(eeg >= (max_indiv - 1e-5)) * 100.0)

    # 4. PQOM: % of q values where EEG > max(perf(g), perf(b))
    pqom = float(np.mean(eeg > (max_indiv + 1e-5)) * 100.0)

    # 5. PCFA: % contribution of feature-dependent allocator
    pcfa = float(np.mean(pcfa_choices) * 100.0) if pcfa_choices else 0.0

    # 6. 95TQM: highest q where EEG >= 0.95 * max(perf(g), perf(b))
    thresh_95 = 0.95 * max_indiv
    valid_q_95 = q_arr[eeg >= (thresh_95 - 1e-5)]
    if len(valid_q_95) > 0:
        tqm95 = float(np.max(valid_q_95) * 100.0)
    else:
        tqm95 = 0.0

    # 7. Max Acc / Max Performance
    max_perf = float(np.max(eeg) * 100.0)

    # 8. Argmax q: highest q at which maximum performance is achieved
    max_val = np.max(eeg)
    max_indices = np.where(np.isclose(eeg, max_val, atol=1e-5))[0]
    argmax_q = float(q_arr[max_indices[-1]] * 100.0) if len(max_indices) > 0 else 0.0

    return {
        "AUC": float(round(auc_eeg * 100.0, 2)),
        "PPCR": float(round(ppcr, 2)),
        "PQEOM": float(round(pqeom, 2)),
        "PQOM": float(round(pqom, 2)),
        "PCFA": float(round(pcfa, 2)),
        "95TQM": float(round(tqm95, 2)),
        "Max_Acc": float(round(max_perf, 2)),
        "Argmax_q": float(round(argmax_q, 2)),
        "s_Acc": float(round(sufficiency_accuracy, 2)),
        "GB_Perf": float(round(perf_gb * 100.0, 2)),
        "BB_Perf": float(round(perf_bb * 100.0, 2)),
    }
