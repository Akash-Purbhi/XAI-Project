"""Build the Phase-2 experimental-results comparison PDF.

The report deliberately separates:
1. Published results copied from the original EEG paper.
2. Results obtained by this repository's local reproduction of original EEG.
3. Results obtained by the Phase-2 Cost-Aware EEG implementation.
"""

from pathlib import Path
import os
import shutil
import textwrap

import fitz
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PAPER = REPO_ROOT / "XAI Paper.pdf"
RESULTS = ROOT / "results" / "cost_aware_extension"
OUTPUT = ROOT / "phase2_experimental_results_comparison.pdf"

BLUE = "#1a365d"
MID_BLUE = "#2b6cb0"
TEXT = "#2d3748"
GREEN = "#276749"
LIGHT_BLUE = "#ebf8ff"
LIGHT_GRAY = "#f7fafc"


def mean_std(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def pct(value):
    return f"{value * 100:.1f}%"


def add_header(fig, title, subtitle=None):
    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=14, fontweight="bold", color=BLUE)
    if subtitle:
        fig.text(0.5, 0.938, subtitle, ha="center", va="top",
                 fontsize=9, style="italic", color="#4a5568")
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.915, 0.915],
                                transform=fig.transFigure, color="#cbd5e0", lw=1.2))


def add_wrapped_text(fig, x, y, text, width=105, fontsize=9, line_spacing=1.35,
                     color=TEXT, va="top"):
    fig.text(x, y, "\n".join(textwrap.wrap(text, width=width)),
             ha="left", va=va, fontsize=fontsize, color=color,
             linespacing=line_spacing)


def render_paper_pages():
    if not PAPER.exists():
        raise FileNotFoundError(f"Original paper not found: {PAPER}")
    document = fitz.open(PAPER)
    rendered = {}
    for pdf_index in (4, 5, 6):
        page = document[pdf_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False)
        path = ROOT / "assets" / f"_paper_extract_pdf_page_{pdf_index + 1}.png"
        pixmap.save(path)
        rendered[pdf_index + 1] = path
    document.close()
    return rendered


def add_table(fig, data, columns, bbox, fontsize=7.5, header_color=MID_BLUE,
              row_colors=None):
    table = fig.add_axes(bbox)
    table.axis("off")
    rendered = table.table(cellText=data, colLabels=columns, cellLoc="center",
                           loc="center", bbox=[0, 0, 1, 1])
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(fontsize)
    for (row, col), cell in rendered.get_celld().items():
        cell.set_edgecolor("#cbd5e0")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row_colors:
            cell.set_facecolor(row_colors[(row - 1) % len(row_colors)])
        elif row % 2 == 0:
            cell.set_facecolor(LIGHT_GRAY)
    return rendered


def build_report():
    comparison = pd.read_csv(RESULTS / "comparison_results.csv")
    baseline = pd.read_csv(RESULTS / "baseline_results.csv")
    cost_aware = pd.read_csv(RESULTS / "cost_aware_results.csv")
    timing = pd.read_csv(RESULTS / "inference_timing.csv")
    paper_pages = render_paper_pages()

    q_values = sorted(comparison["target_coverage"].unique())
    with PdfPages(OUTPUT) as pdf:
        # Page 1: connection between gap, solution, implementation, and evidence.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "PHASE 2 EXPERIMENTAL RESULTS", 
                   "Cost-Aware Allocation for Performance-Maximizing Explainable Ensembles")
        fig.text(0.08, 0.885, "1. Research question and evidence chain", fontsize=11,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.852,
            "Phase 2 investigates three distinct limitations of the original EEG framework: routing instability "
            "under perturbations, the opacity of the allocator itself, and unnecessary pre-routing computation.",
            width=105)
        fig.text(0.08, 0.755, "Identified gap → proposed solution → measurable prediction", fontsize=11,
                 fontweight="bold", color=BLUE)
        chain = [
            ("Gap 3", "Small feature perturbations can flip model assignments and explanations."),
            ("Gap 6", "The allocator is opaque; users cannot see why it routed an observation."),
            ("Cost-aware", "The original gate needs black-box outputs before routing, wasting computation."),
            ("Solutions", "SR-EEG stabilizes routing; TreeSHAP explains routing; Cost-Aware EEG enables conditional execution."),
            ("Evidence", "Separate results are reported for the 24-dataset stability, SHAP, and Wine cost benchmarks.")
        ]
        y = 0.715
        for label, body in chain:
            rect = plt.Rectangle((0.08, y - 0.055), 0.18, 0.047,
                                 transform=fig.transFigure,
                                 facecolor=LIGHT_BLUE, edgecolor="#3182ce", lw=0.8)
            fig.patches.append(rect)
            fig.text(0.17, y - 0.031, label, ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color=MID_BLUE)
            add_wrapped_text(fig, 0.29, y - 0.013, body, width=75, fontsize=8.5)
            y -= 0.092
        fig.text(0.08, 0.285, "Experimental design", fontsize=11,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.255,
            "Gap 3 and Gap 6 use 24 datasets, seeds {0, 1, 2}, five perturbation levels, and q values "
            "from 0.1 to 0.9. The cost-aware experiment uses Wine (N=537), q ∈ {0.20, 0.40, 0.60, 0.80}, "
            "validation-only thresholds, and 50 timing repetitions after 10 warm-up runs.",
            width=105, fontsize=9)
        fig.patches.append(plt.Rectangle((0.08, 0.08), 0.84, 0.095,
                                         transform=fig.transFigure,
                                         facecolor="#f0fff4", edgecolor="#68d391", lw=1))
        fig.text(0.10, 0.145, "Reporting integrity", fontsize=10, fontweight="bold", color=GREEN)
        add_wrapped_text(
            fig, 0.10, 0.123,
            "Published values are labeled Published. Local baseline values are labeled Reproduced Original EEG. "
            "Only Cost-Aware EEG values are presented as Phase-2 results.",
            width=96, fontsize=8.5)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: original paper extracts.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "PUBLISHED EVIDENCE FROM THE ORIGINAL PAPER",
                   "Extracts are reproduced from the supplied XAI Paper.pdf; they are not our experimental results")
        fig.text(0.08, 0.885,
                 "Extract A — Results discussion and metric definitions (original paper printed p. 14621; PDF p. 5)",
                 fontsize=9.5, fontweight="bold", color=BLUE)
        img = mpimg.imread(paper_pages[5])
        ax = fig.add_axes([0.09, 0.49, 0.82, 0.37])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.465,
                 "The source text reports cross-dataset PPCR = 37%, PQEOM = 74%, 95TQM = 94%, "
                 "maximum accuracy = 84%, and sufficiency accuracy = 76%. These are published claims.",
                 fontsize=8.5, color=TEXT)
        fig.text(0.08, 0.425,
                 "Extract B — Original paper Table 2 (original paper printed p. 14622; PDF p. 6)",
                 fontsize=9.5, fontweight="bold", color=BLUE)
        img = mpimg.imread(paper_pages[6])
        ax = fig.add_axes([0.09, 0.075, 0.82, 0.33])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.045,
                 "The Wine row in the published table reports AUC 79, PPCR 21, PQEOM 71, "
                 "95TQM 98, Max Acc 80, Argmax q 70, and s Acc 78 (all percentages as defined by the paper).",
                 fontsize=8.2, color=TEXT)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: feature-set extract and why the improvement changes the gate.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "PUBLISHED FEATURE-SET RESULT AND DESIGN RATIONALE",
                   "Original paper printed p. 14623 (PDF p. 7) is used as supporting evidence")
        fig.text(0.08, 0.885,
                 "Extract C — Feature-set ablation (original paper printed p. 14623; PDF p. 7)",
                 fontsize=9.5, fontweight="bold", color=BLUE)
        img = mpimg.imread(paper_pages[7])
        ax = fig.add_axes([0.09, 0.51, 0.82, 0.34])
        ax.imshow(img)
        ax.axis("off")
        add_wrapped_text(
            fig, 0.08, 0.475,
            "The paper reports that the all-feature set x, g, b, dce, dmse was most consistently strong, "
            "while the unaugmented x-only allocator was consistently worst in its benchmark. This creates the "
            "deployment tension addressed in Phase 2: the strongest allocation features include model outputs, "
            "but those outputs require model execution before routing.",
            width=105, fontsize=8.7)
        fig.text(0.08, 0.345, "How Phase 2 uses this evidence", fontsize=11,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.315,
            "The proposed change does not claim that X-only allocation is universally more accurate. It changes "
            "the deployment objective by removing pre-routing black-box execution. The experiments therefore "
            "measure the explicit trade-off between computational savings and predictive performance, rather than "
            "hiding the information loss.",
            width=105, fontsize=9)
        fig.patches.append(plt.Rectangle((0.08, 0.11), 0.84, 0.13,
                                         transform=fig.transFigure,
                                         facecolor="#fffaf0", edgecolor="#ed8936", lw=1))
        fig.text(0.10, 0.205, "Important distinction", fontsize=10, fontweight="bold", color="#9c4221")
        add_wrapped_text(
            fig, 0.10, 0.182,
            "Pages 14621–14623 are source-paper extracts. They document the published EEG benchmark and "
            "motivate the gap; they are not evidence generated by this Phase-2 implementation.",
            width=96, fontsize=8.5, color="#744210")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Gap 3 stability results.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "GAP 3 — PERTURBATION STABILITY",
                   "Original EEG vs. Stability-Robust EEG (SR-EEG), 24 datasets")
        fig.text(0.08, 0.885, "Research gap and proposed improvement", fontsize=11,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.852,
            "The original EEG allocator is trained on clean rankings and then applies a hard threshold. "
            "Small normalized feature perturbations can move an observation across the routing boundary. "
            "SR-EEG adds neighborhood perturbation augmentation and regularization so nearby inputs receive "
            "more consistent allocation decisions.",
            width=105, fontsize=9)
        stability = pd.read_csv(RESULTS.parent / "stability_master.csv")
        stab = stability[stability["Epsilon"] == 0.05].groupby("Method").agg(
            afr=("AFR", "mean"),
            pc=("Prediction_Consistency", "mean"),
            clean=("Clean_Perf", "mean"),
            perturbed=("Perturbed_Perf", "mean"),
        ).reset_index()
        stab_rows = [[row.Method, f"{row.afr:.2f}%",
                      f"{row.pc:.2f}%", f"{row.clean:.2f}%",
                      f"{row.perturbed:.2f}%"]
                     for row in stab.itertuples()]
        fig.text(0.08, 0.735, "Table 3. Cross-dataset mean at ε=0.05 (24 datasets, all q and seeds).",
                 fontsize=9.5, fontweight="bold", color=BLUE)
        add_table(fig, stab_rows,
                  ["Method", "AFR (lower)", "Prediction consistency\n(higher)",
                   "Clean performance", "Perturbed performance"],
                  [0.08, 0.57, 0.84, 0.13], fontsize=7.8)
        fig.text(0.08, 0.515, "Representative benchmark findings", fontsize=10.5,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.482,
            "At ε=0.05, documented reductions include CPU_R: 27.74% → 7.77% (72.0%), "
            "California: 21.84% → 9.67% (55.7%), BrazilianHouses_R: 40.47% → 18.66% (53.9%), "
            "and Wine: 14.68% → 8.56% (41.7%). These values are from results/stability_master.csv "
            "and PHASE2_RESULTS_GAP3.md.",
            width=105, fontsize=8.8)
        img = mpimg.imread(ROOT / "results" / "figures" / "gap3_cross_dataset_stability.png")
        ax = fig.add_axes([0.08, 0.08, 0.84, 0.29])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.045,
                 "Gap 3 conclusion: SR-EEG improves routing stability while preserving the task objective; "
                 "it addresses robustness, not inference cost.",
                 fontsize=8.3, style="italic", color="#4a5568")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Gap 6 meta-explainability results.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "GAP 6 — META-EXPLAINABILITY",
                   "Explaining the EEG allocator with TreeSHAP")
        fig.text(0.08, 0.885, "Research gap and proposed improvement", fontsize=11,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.852,
            "EEG routes observations for explainability, but its gradient-boosted allocator is itself opaque. "
            "Gap 6 asks: who explains the explainer? TreeSHAP quantifies whether routing is driven by raw domain "
            "features or by disagreement and confidence between component models.",
            width=105, fontsize=9)
        shap = pd.read_csv(RESULTS.parent / "shap_meta_summary.csv")
        mean_orig_domain = shap["Original_Domain_Feature_Share"].mean()
        mean_orig_model = shap["Original_Disagreement_Confidence_Share"].mean()
        mean_sr_domain = shap["SREEG_Domain_Feature_Share"].mean()
        mean_sr_model = shap["SREEG_Disagreement_Confidence_Share"].mean()
        shap_rows = [
            ["Original EEG", f"{mean_orig_model:.2f}%", f"{mean_orig_domain:.2f}%",
             "Disagreement/model confidence"],
            ["SR-EEG", f"{mean_sr_model:.2f}%", f"{mean_sr_domain:.2f}%",
             "Disagreement remains dominant; shift is dataset-dependent"],
        ]
        fig.text(0.08, 0.735, "Table 4. Mean TreeSHAP attribution shares across 24 datasets.",
                 fontsize=9.5, fontweight="bold", color=BLUE)
        add_table(fig, shap_rows,
                  ["Allocator", "Disagreement + outputs", "Raw domain features", "Interpretation"],
                  [0.08, 0.50, 0.84, 0.18], fontsize=7.8)
        add_wrapped_text(
            fig, 0.08, 0.455,
            "The SHAP summary shows that original EEG routing is primarily driven by component disagreement "
            "and predictive confidence rather than domain features. This makes individual routing decisions "
            "auditable: users can inspect whether an observation was sent to the black-box because the two "
            "component models disagreed.",
            width=105, fontsize=8.8)
        img = mpimg.imread(ROOT / "results" / "figures" / "gap6_meta_explainability_summary.png")
        ax = fig.add_axes([0.08, 0.10, 0.84, 0.29])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.08, 0.06,
                 "Gap 6 conclusion: TreeSHAP makes the routing mechanism inspectable; it addresses "
                 "meta-explainability, not perturbation stability or compute cost.",
                 fontsize=8.3, style="italic", color="#4a5568")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: published vs local original vs proposed metrics.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "BEFORE-AND-AFTER RESULTS",
                   "Published EEG values, local Original_EEG reproduction, and Phase-2 CostAware_EEG")
        published = [
            ["AUC", "79", "Published Table 2, p. 14622"],
            ["PPCR", "21%", "Published Table 2, p. 14622"],
            ["PQEOM", "71%", "Published Table 2, p. 14622"],
            ["95TQM", "98%", "Published Table 2, p. 14622"],
            ["Max Acc", "80%", "Published Table 2, p. 14622"],
            ["Argmax q", "70%", "Published Table 2, p. 14622"],
            ["Sufficiency Acc", "78%", "Published Table 2, p. 14622"],
        ]
        add_table(fig, published, ["Published Wine metric", "Value", "Source/status"],
                  [0.08, 0.70, 0.84, 0.17], fontsize=8)
        fig.text(0.08, 0.665, "Table 1. Published source-paper values (not generated by this project).",
                 fontsize=9.5, fontweight="bold", color=BLUE)

        rows = []
        for q in q_values:
            b = baseline[baseline["target_coverage"] == q]
            c = cost_aware[(cost_aware["target_coverage"] == q) &
                           (cost_aware["mode"] == "matched_coverage")]
            rows.append([
                f"q={q:.2f}", mean_std(b["accuracy"]) + "%",
                mean_std(c["accuracy"]) + "%",
                mean_std(c["relative_latency_reduction_pct"]) + "%",
                mean_std(c["cost_savings_pct"]) + "%",
            ])
        fig.text(0.08, 0.625,
                 "Table 2. Local experimental comparison (mean ± standard deviation across seeds [0, 1, 2]).",
                 fontsize=9.5, fontweight="bold", color=BLUE)
        add_table(fig, rows,
                  ["Quota", "Original_EEG\naccuracy", "CostAware_EEG\naccuracy",
                   "Latency reduction", "Cost saved"],
                  [0.08, 0.36, 0.84, 0.23], fontsize=8)
        add_wrapped_text(
            fig, 0.08, 0.315,
            "The original implementation results above are obtained from baseline_results.csv using the "
            "Original_EEG path. The proposed results are obtained from cost_aware_results.csv using the "
            "matched_coverage CostAware_EEG path. The published paper values are not substituted for either "
            "local experiment.",
            width=105, fontsize=8.7)
        fig.text(0.08, 0.235, "Component timing evidence", fontsize=10.5,
                 fontweight="bold", color=BLUE)
        component = (timing[timing["category"] == "Component"]
                     [["pipeline_step", "mean_latency_ms"]]
                     .drop_duplicates("pipeline_step", keep="first"))
        comp_rows = [[row.pipeline_step.replace("_", " "), f"{row.mean_latency_ms:.3f} ms"]
                     for row in component.itertuples()]
        add_table(fig, comp_rows, ["Measured component", "Mean latency (N=537)"],
                  [0.08, 0.08, 0.48, 0.13], fontsize=8)
        add_wrapped_text(
            fig, 0.61, 0.185,
            "The black-box component is approximately 241× slower than the glass-box component in the "
            "stored benchmark timing. This asymmetry explains why conditional execution is a meaningful "
            "computational intervention.",
            width=38, fontsize=8.5)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: charts from stored empirical results.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "VISUAL BEFORE-AND-AFTER COMPARISON",
                   "All plotted values are computed from Phase-2 result CSVs")
        grouped_b = baseline.groupby("target_coverage").agg(
            accuracy=("accuracy", "mean"), latency=("mean_latency_ms", "mean"))
        grouped_c = cost_aware[cost_aware["mode"] == "matched_coverage"].groupby(
            "target_coverage").agg(accuracy=("accuracy", "mean"),
                                   latency=("mean_latency_ms", "mean"),
                                   savings=("cost_savings_pct", "mean"))
        ax1 = fig.add_axes([0.10, 0.59, 0.82, 0.26])
        ax1.plot(q_values, [grouped_b.loc[q, "accuracy"] for q in q_values],
                 "o-", label="Original EEG", color="#718096", lw=2)
        ax1.plot(q_values, [grouped_c.loc[q, "accuracy"] for q in q_values],
                 "o-", label="Cost-Aware EEG (ours)", color=MID_BLUE, lw=2)
        ax1.set_title("Predictive accuracy vs. explainability quota", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Target glass-box quota q")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(alpha=0.25)
        ax1.legend(frameon=False)
        ax1.set_xticks(q_values)

        ax2 = fig.add_axes([0.10, 0.25, 0.37, 0.23])
        ax2.plot(q_values, [grouped_b.loc[q, "latency"] for q in q_values],
                 "o-", label="Original EEG", color="#718096", lw=2)
        ax2.plot(q_values, [grouped_c.loc[q, "latency"] for q in q_values],
                 "o-", label="Cost-Aware EEG", color=GREEN, lw=2)
        ax2.set_title("Latency scaling", fontsize=10, fontweight="bold")
        ax2.set_xlabel("q")
        ax2.set_ylabel("Latency (ms)")
        ax2.grid(alpha=0.25)
        ax2.set_xticks(q_values)

        ax3 = fig.add_axes([0.56, 0.25, 0.37, 0.23])
        ax3.bar([str(q) for q in q_values],
                [grouped_c.loc[q, "savings"] for q in q_values],
                color="#48bb78")
        ax3.set_title("Realized compute savings", fontsize=10, fontweight="bold")
        ax3.set_xlabel("q")
        ax3.set_ylabel("Savings (%)")
        ax3.grid(axis="y", alpha=0.25)
        for i, q in enumerate(q_values):
            ax3.text(i, grouped_c.loc[q, "savings"] + 1,
                     f"{grouped_c.loc[q, 'savings']:.1f}%", ha="center", fontsize=8)
        add_wrapped_text(
            fig, 0.08, 0.17,
            "Interpretation: the proposed method produces a monotonic latency and compute benefit as q "
            "increases, while the original pipeline remains approximately flat because the black-box model "
            "is evaluated before routing. The accuracy cost is small at moderate quotas and larger at q=0.80.",
            width=105, fontsize=8.8)
        fig.text(0.08, 0.10,
                 "Figure note: these are implementation results, not screenshots or re-plots of the source paper.",
                 fontsize=8, style="italic", color="#4a5568")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: literature support, conclusion, reproducibility.
        fig = plt.figure(figsize=(8.5, 11), dpi=220)
        add_header(fig, "INTERPRETATION, SUPPORTING LITERATURE, AND REPRODUCIBILITY")
        fig.text(0.08, 0.885, "Connection to the proposed improvement", fontsize=11,
                 fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.852,
            "BranchyNet supports the central design principle used here: easy cases should exit through a "
            "cheaper computation path, while difficult cases receive more expensive processing. Cost-Aware EEG "
            "adapts that principle to a tabular explainable ensemble. Unlike BranchyNet's neural early exits, "
            "this implementation routes between a glass-box and black-box model and retains an explicit "
            "explainability quota.",
            width=105, fontsize=9)
        fig.text(0.08, 0.735, "Main findings", fontsize=11, fontweight="bold", color=BLUE)
        findings = [
            "Gap 3: SR-EEG reduces allocation flips under controlled perturbations across the 24-dataset benchmark.",
            "Gap 6: TreeSHAP shows that routing is primarily driven by component disagreement and confidence, making the allocator auditable.",
            "At q=0.20–0.60, the proposed method remains close to the reproduced original accuracy while reducing measured latency and realized cost.",
            "At q=0.80, the accuracy decrease is reported rather than hidden; it is the observed price of removing model-output features from the pre-routing gate.",
            "The three extensions address different failure modes: robustness, auditability, and deployment cost.",
        ]
        y = 0.70
        for item in findings:
            fig.text(0.10, y, "•", fontsize=13, color=MID_BLUE)
            add_wrapped_text(fig, 0.125, y + 0.004, item, width=92, fontsize=8.5)
            y -= 0.065
        fig.text(0.08, 0.345, "References", fontsize=11, fontweight="bold", color=BLUE)
        refs = (
            "[1] Pisztora, V., and Li, J. (2024). Learning Performance Maximizing Ensembles with "
            "Explainability Guarantees. Proceedings of AAAI-24, 38(13), 14594–14602. "
            "Source extracts used: printed pp. 14621–14623 in the supplied PDF.\n\n"
            "[2] Dai, D. et al. (2022). StableMoE: Stable Routing Strategy for Mixture of Experts. ACL. "
            "Supporting literature for routing stability and the Gap 3 motivation.\n\n"
            "[3] Teerapittayanont, S., McDanel, B., and Kung, H. T. (2016). BranchyNet: Fast Inference "
            "via Early Exiting from Deep Neural Networks. ICPR, 2464–2469. "
            "DOI: 10.1109/ICPR.2016.7900001.\n\n"
            "[4] Lundberg, S. M., and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. "
            "NeurIPS. Supporting literature for TreeSHAP-based meta-explainability.\n\n"
            "[5] Bishop, C. M. (1995). Training with Noise is Equivalent to Tikhonov Regularization. "
            "Neural Computation, 7(1), 108–116. (Methodological context for regularization and robustness.)"
        )
        add_wrapped_text(fig, 0.08, 0.312, refs, width=105, fontsize=7.2, line_spacing=1.1)
        fig.text(0.08, 0.175, "Reproducibility record", fontsize=11, fontweight="bold", color=BLUE)
        add_wrapped_text(
            fig, 0.08, 0.142,
            "Dataset: Wine / OpenML benchmark; test batch N=537; seeds 0, 1, 2; validation-only threshold "
            "selection; timing repetitions: 50 with 10 warm-ups. Source files: experiment_config.json, "
            "baseline_results.csv, cost_aware_results.csv, comparison_results.csv, inference_timing.csv, "
            "and the Phase-2 implementation scripts. Generated report: phase2_experimental_results_comparison.pdf.",
            width=105, fontsize=8.7)
        pdf.savefig(fig)
        plt.close(fig)

    # Keep generated paper extracts out of the committed deliverable.
    for path in paper_pages.values():
        path.unlink(missing_ok=True)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    build_report()
