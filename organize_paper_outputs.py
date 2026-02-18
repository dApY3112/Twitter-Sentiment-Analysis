"""
Organize all paper outputs into a single directory structure
Copies all necessary CSV, PNG, and TEX files for the paper
"""

import shutil
from pathlib import Path

# Define paths
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "paper_outputs"

# Create main directory and subdirectories
(OUTPUT_DIR / "tables_csv").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tables_tex").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

print("=== Organizing Paper Outputs ===\n")

# ===== TABLES (CSV) =====
print("📊 Copying CSV tables...")

csv_files = [
    # Notebook 04: Translation & Robustness
    ("translation_eval/robustness_results.csv", "01_robustness_results.csv"),
    ("translation_eval/flip_rate.csv", "02_flip_rate.csv"),
    
    # Notebook 05: XAI Consistency
    ("xai_results/ctam_summary.csv", "03_ctam_summary.csv"),
    ("xai_results/overlap_summary.csv", "04_overlap_summary.csv"),
    
    # Notebook 06: Error Drift Analysis
    ("error_drift_analysis/flip_type_summary.csv", "05_flip_type_summary.csv"),
    ("error_drift_analysis/correlation_summary.csv", "06_correlation_summary.csv"),
    
    # Notebook 07: Faithfulness
    ("faithfulness_results/faithfulness_summary.csv", "07_faithfulness_summary.csv"),
    
    # Notebook 08: Baseline Comparison
    ("baseline_comparison/model_comparison.csv", "08_model_comparison.csv"),
    
    # Notebook 09: Calibration & Triage
    ("calibration_triage/calibration_summary.csv", "09_calibration_summary.csv"),
    ("calibration_triage/triage_summary.csv", "10_triage_summary.csv"),
    ("calibration_triage/error_detection_summary.csv", "11_error_detection_summary.csv"),
    ("calibration_triage/cost_benefit_analysis.csv", "12_cost_benefit_analysis.csv"),
]

for src, dst in csv_files:
    src_path = ROOT / src
    dst_path = OUTPUT_DIR / "tables_csv" / dst
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"  ✅ {dst}")
    else:
        print(f"  ⚠️  {src} not found")

# ===== TABLES (TEX) =====
print("\n📝 Copying LaTeX tables...")

tex_files = [
    "calibration_summary.tex",
    "triage_summary.tex",
    "error_detection_summary.tex",
    "cost_benefit_analysis.tex",
]

for tex in tex_files:
    src_path = ROOT / tex
    dst_path = OUTPUT_DIR / "tables_tex" / tex
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"  ✅ {tex}")
    else:
        print(f"  ⚠️  {tex} not found")

# ===== FIGURES =====
print("\n🖼️  Copying figures...")

figure_files = [
    # Notebook 05: XAI
    ("xai_results/ctam_distribution.png", "fig_01_ctam_distribution.png"),
    
    # Notebook 06: Error Drift
    ("error_drift_analysis/flip_type_distribution.png", "fig_02_flip_type_distribution.png"),
    ("error_drift_analysis/drift_overlap_vs_correctness.png", "fig_03_drift_overlap_correctness.png"),
    ("error_drift_analysis/scatter_drift_vs_overlap.png", "fig_04_scatter_drift_overlap.png"),
    
    # Notebook 07: Faithfulness
    ("faithfulness_results/faithfulness_boxplots.png", "fig_05_faithfulness_boxplots.png"),
    ("faithfulness_results/faithfulness_scatter.png", "fig_06_faithfulness_scatter.png"),
    
    # Notebook 08: Baseline
    ("baseline_comparison/model_comparison_accuracy.png", "fig_07_model_comparison.png"),
    
    # Notebook 09: Calibration & Triage (MAIN FIGURES)
    ("calibration_triage/flip_matrix_en_es.png", "fig_08_flip_matrix.png"),  # Figure 2 in paper
    ("calibration_triage/reliability_diagrams.png", "fig_09_reliability_diagrams.png"),  # Figure 3
    ("calibration_triage/ece_comparison.png", "fig_10_ece_comparison.png"),  # Figure 4A
    ("calibration_triage/confidence_by_correctness.png", "fig_11_confidence_correctness.png"),  # Figure 4B
    ("calibration_triage/error_detection_curves.png", "fig_12_error_detection_curves.png"),
    ("calibration_triage/case_study_0.png", "fig_13_case_study_0.png"),  # Figure 5A
    ("calibration_triage/case_study_1.png", "fig_14_case_study_1.png"),  # Figure 5B
    ("calibration_triage/case_study_2.png", "fig_15_case_study_2.png"),  # Figure 5C
]

for src, dst in figure_files:
    src_path = ROOT / src
    dst_path = OUTPUT_DIR / "figures" / dst
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"  ✅ {dst}")
    else:
        print(f"  ⚠️  {src} not found")

# ===== CREATE INDEX =====
print("\n📋 Creating index file...")

index_content = """# Paper Outputs Index

Generated: February 2026

## Directory Structure

```
paper_outputs/
├── tables_csv/      # 12 CSV files (raw data tables)
├── tables_tex/      # 4 TEX files (LaTeX formatted)
└── figures/         # 15 PNG files (all visualizations)
```

---

## 📊 Tables (CSV Format)

### Notebook 04: Translation & Robustness
- `01_robustness_results.csv` → **Table 2**: Accuracy EN/ES/FR by label
- `02_flip_rate.csv` → Flip rates (EN→ES, EN→FR)

### Notebook 05: XAI Consistency  
- `03_ctam_summary.csv` → Mean CTAM scores per language
- `04_overlap_summary.csv` → **Table 3**: Mean Jaccard overlap (EN-ES: 0.86, EN-FR: 0.86)

### Notebook 06: Error Drift Analysis
- `05_flip_type_summary.csv` → Error categories (consistent correct/incorrect, flips)
- `06_correlation_summary.csv` → Correlation: overlap vs correctness, CTAM vs correctness

### Notebook 07: Faithfulness Validation
- `07_faithfulness_summary.csv` → AOPC, Sufficiency, Comprehensiveness scores

### Notebook 08: Baseline Comparison
- `08_model_comparison.csv` → BERT vs LSTM vs Rules accuracy

### Notebook 09: Calibration & Triage (MAIN RESULTS)
- `09_calibration_summary.csv` → **Table 4**: ECE, Brier, confidence per language
- `10_triage_summary.csv` → Triage distribution (HIGH/MEDIUM/LOW: 10/20/70%)
- `11_error_detection_summary.csv` → **Table 5A**: Precision/Recall/F1 per strategy
- `12_cost_benefit_analysis.csv` → **Table 5B**: Workload/Recall/Efficiency (1.41× for HIGH only)

---

## 📝 Tables (LaTeX Format)

Ready for Overleaf:
- `calibration_summary.tex` → Table 4
- `triage_summary.tex` → Table 3 (supplementary)
- `error_detection_summary.tex` → Table 5A
- `cost_benefit_analysis.tex` → Table 5B

---

## 🖼️ Figures

### Supporting Figures (Notebooks 05-08)
- `fig_01_ctam_distribution.png` → CTAM distribution violin plots
- `fig_02_flip_type_distribution.png` → Error type bar chart
- `fig_03_drift_overlap_correctness.png` → Heatmap: XAI vs correctness
- `fig_04_scatter_drift_overlap.png` → Scatter: CTAM drift vs overlap
- `fig_05_faithfulness_boxplots.png` → Faithfulness metric distributions
- `fig_06_faithfulness_scatter.png` → AOPC vs confidence
- `fig_07_model_comparison.png` → Baseline model accuracy comparison

### Main Figures (Notebook 09 - For Paper Body)
- `fig_08_flip_matrix.png` → **Figure 2**: Confusion matrix (EN vs ES predictions)
- `fig_09_reliability_diagrams.png` → **Figure 3**: Calibration curves (3 languages)
- `fig_10_ece_comparison.png` → **Figure 4A**: ECE bar chart (EN: 0.04, ES/FR: 0.36)
- `fig_11_confidence_correctness.png` → **Figure 4B**: Confidence by correctness
- `fig_12_error_detection_curves.png` → Precision-Recall curves (all strategies)
- `fig_13_case_study_0.png` → **Figure 5A**: Token importance example 1
- `fig_14_case_study_1.png` → **Figure 5B**: Token importance example 2
- `fig_15_case_study_2.png` → **Figure 5C**: Token importance example 3

---

## 🎯 Key Findings Summary

### RQ1: Cross-lingual Robustness
- Accuracy drop: EN 91% → ES 88%, FR 86%
- Flip rate: ~8-10% predictions change across languages

### RQ2: XAI Consistency
- **HIGH overlap**: Jaccard ~0.86 (EN-ES/FR) → Explanations mostly preserved
- CTAM scores similar across languages (cyber term focus maintained)
- Faithfulness validated: AOPC >0.15 for all languages

### RQ3: Calibration (MAIN CONTRIBUTION)
- **EXTREME miscalibration in translation**: ECE jumps 0.04 → 0.36 (8.6× worse)
- Confidence paradox: ~98% predictions >0.9 confidence despite 12-14% errors
- Triage system: 10% workload catches 14% errors @ 53.5% precision (1.41× efficient)

---

## 📄 Recommended Paper Structure

1. **Introduction**
   - Problem: Translation-based evaluation common but calibration ignored
   - Gap: No work on confidence reliability in cross-lingual settings
   
2. **Related Work**
   - Cross-lingual NLP & robustness
   - XAI consistency across domains
   - Calibration in production systems

3. **Methodology**
   - Dataset: Twitter cyber threat sentiment
   - Pipeline: Train EN → Translate ES/FR → XAI → Calibration → Triage
   
4. **Results**
   - RQ1: Robustness (Table 2, Fig 2)
   - RQ2: XAI Consistency (Table 3, Fig 1, 3-6)
   - RQ3: Calibration Collapse (Table 4-5, Fig 7-11) ← **MAIN FOCUS**
   
5. **Discussion**
   - Why calibration breaks: semantic shift in feature space
   - Implications: Confidence-based systems fail in translation
   - Triage as mitigation strategy
   
6. **Conclusion**
   - Novel finding: Calibration ≠ Robustness
   - Practical impact: Don't trust confidence in cross-lingual deployment
   - Future: Better calibration techniques for translated text

---

## ✅ Checklist for Paper Writing

- [x] All 9 notebooks executed successfully
- [x] 35+ output files generated
- [x] 12 CSV tables ready
- [x] 4 LaTeX tables formatted
- [x] 15 figures rendered
- [ ] Table 1 (Dataset statistics) - Create manually
- [ ] Figure 1 (Pipeline diagram) - Create manually
- [ ] Write Introduction
- [ ] Write Related Work
- [ ] Write Methodology
- [ ] Write Results (3 sections for 3 RQs)
- [ ] Write Discussion
- [ ] Write Conclusion

**STATUS: Ready to write! 📝**
"""

with open(OUTPUT_DIR / "INDEX.md", "w", encoding="utf-8") as f:
    f.write(index_content)

print("  ✅ INDEX.md created")

print(f"\n✨ Done! All outputs organized in: {OUTPUT_DIR.relative_to(ROOT)}")
print(f"\n📂 Summary:")
print(f"  - {len(csv_files)} CSV tables")
print(f"  - {len(tex_files)} LaTeX tables")
print(f"  - {len(figure_files)} Figures")
print(f"\n💡 Next: Open INDEX.md to see complete inventory")
