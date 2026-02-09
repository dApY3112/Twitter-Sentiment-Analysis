# Cross-lingual Explanation Consistency for Sentiment Analysis

Code repository cho paper: **"Cross-lingual Explanation Consistency for Cyber Threat Sentiment on X: A Lightweight Translation-based Multilingual XAI Study"**

**Note:** Paper này làm được trên dataset sentiment chung, không nhất thiết phải cyber-specific. Nếu dataset của bạn không có cyber terms nhiều, đổi thành **"Cross-lingual Explanation Consistency for Sentiment on X"** và thay CTAM bằng "Domain Cue Attribution Mass" (cues = emotion words, negation, etc.).

---

## 📁 Project Structure

```
Twitter Sentiment Analysis Dataset/
├── 01_explore_twitter_sentiment.ipynb  # EDA + language audit
├── 02_sample_split.ipynb               # Stratified sampling + train/val/test split
├── 03_train_model.ipynb                # Train distilbert-multilingual
├── 04_translate_eval.ipynb             # Translation EN→ES/FR + robustness eval
├── 05_xai_consistency.ipynb            # XAI (IG) + CTAM metrics
├── utils.py                             # Cyber patterns, CTAM, overlap metrics
├── twitter_sentiment_dataset.csv       # Dataset gốc (large)
├── data_splits/                         # Train/val/test CSVs
├── model_output/                        # Trained model + checkpoints
├── translation_eval/                    # Translated texts + predictions
└── xai_results/                         # IG attributions + CTAM results
```

---

## 🚀 Workflow (chạy tuần tự)

### **Step 0: Environment setup**

```bash
pip install pandas numpy scikit-learn matplotlib
pip install torch transformers datasets accelerate
pip install captum  # for XAI
pip install langid  # for language detection audit (optional)
```

**GPU note:** Code optimized cho GTX 1650 (4GB VRAM). Nếu OOM, giảm `BATCH_SIZE` và `SAMPLE_SIZE` trong notebooks.

---

### **Step 1: EDA + Language Audit**

Notebook: `01_explore_twitter_sentiment.ipynb`

**Mục tiêu:**
- Explore dataset: columns, missing, duplicates, label distribution
- **Language audit** (quan trọng cho paper): chứng minh metadata `language` noisy (en/fr/es) nhưng text thực tế EN-dominant

**Output:**
- Summary statistics (số mẫu, label dist, text length)
- **Language mismatch rate** (metadata vs detected lang) → đưa vào paper Section 4 (Dataset)

---

### **Step 2: Stratified Sampling + Split**

Notebook: `02_sample_split.ipynb`

**Mục tiêu:**
- Stratified sampling từ dataset lớn (memory-efficient chunking)
- Split 70/15/15 train/val/test

**Config cần điều chỉnh:**
```python
TEXT_COL = 'clean_text'  # or 'text', based on your dataset
LABEL_COL = 'sentiment'  # or 'label'
TARGET_PER_LABEL = 15000  # reduce to 5000-10000 if memory tight
```

**Output:**
- `data_splits/train.csv`
- `data_splits/val.csv`
- `data_splits/test.csv`

---

### **Step 3: Train Sentiment Model**

Notebook: `03_train_model.ipynb`

**Mục tiêu:**
- Fine-tune `distilbert-base-multilingual-cased` trên EN train set
- Early stopping, fp16, batch size 8

**Config tối ưu GTX 1650:**
```python
MAX_LENGTH = 128
BATCH_SIZE = 8 
EPOCHS = 5
FP16 = True  # if GPU
```

**Output:**
- `model_output/final_model/` (model + tokenizer)
- `model_output/label_map.json` (label encoding)
- Validation metrics (accuracy, F1-macro)

**Paper section:** Results (baseline performance)

---

### **Step 4: Translation + Robustness Evaluation**

Notebook: `04_translate_eval.ipynb`

**Mục tiêu:**
- Translate val/test EN→ES, EN→FR (MarianMT)
- Evaluate model on EN, ES, FR
- Measure:
  - Accuracy/F1 drop
  - Label flip rate (prediction changes)

**Config:**
```python
EVAL_SPLIT = 'test'  # or 'val'
SAMPLE_SIZE = 2000  # reduce to 1000 if translation too slow
```

**Output:**
- `translation_eval/test_translated.csv` (with text_es, text_fr)
- `translation_eval/test_with_predictions.csv` (with pred_en, pred_es, pred_fr)
- `translation_eval/robustness_results.csv` (summary table)
- `translation_eval/flip_rate.csv`

**Paper section:** Results (Table 1: Robustness metrics)

---

### **Step 5: XAI + Explanation Consistency**

Notebook: `05_xai_consistency.ipynb`

**Mục tiêu:**
- Compute Integrated Gradients (IG) attributions cho EN, ES, FR
- Measure:
  - **CTAM (CyberTerm Attribution Mass)**: tỉ lệ attribution từ cyber/domain terms
  - **Top-k Cyber Token Overlap**: Jaccard overlap của cyber tokens trong top-k important tokens
- Case studies: high drift examples

**Config:**
```python
XAI_SAMPLE_SIZE = 200  # reduce to 100 if too slow
TOP_K = 10  # for overlap metric
```

**Output:**
- `xai_results/xai_full_results.csv` (per-sample CTAM + overlap)
- `xai_results/ctam_summary.csv` (mean CTAM drift)
- `xai_results/overlap_summary.csv` (mean Jaccard)
- `xai_results/ctam_distribution.png` (visualization)

**Paper section:** 
- Results (Table 2: Explanation consistency metrics)
- Discussion (case studies, Figure 2: CTAM drift distribution)

---

## 📊 Key Metrics cho Paper

### **Prediction Robustness (from Notebook 04)**

| Metric | EN | ES | FR | Drop (ES) | Drop (FR) |
|--------|----|----|----|-----------|-----------| 
| Accuracy | X.XXX | X.XXX | X.XXX | -X.XX% | -X.XX% |
| F1-macro | X.XXX | X.XXX | X.XXX | -X.XX% | -X.XX% |
| Label flip rate | - | X.XX% | X.XX% | - | - |

### **Explanation Consistency (from Notebook 05)**

| Metric | EN→ES | EN→FR |
|--------|-------|-------|
| Mean CTAM drift | ±X.XXX | ±X.XXX |
| Mean Jaccard overlap (top-10) | X.XXX | X.XXX |

---

## 🎯 Contributions cho Paper (Section 1: Introduction)

**3 contributions rõ ràng:**

1. **Translation-based multilingual benchmark**
   - Audit language metadata noise (quantified mismatch rate)
   - Create silver translations (EN→ES/FR) for robustness evaluation
   - Report prediction stability across languages

2. **Explanation consistency metrics** (novel, measurable)
   - **CTAM**: measures if model focuses on domain/cyber cues consistently
   - **Top-k overlap**: measures rationale similarity across languages
   - Both metrics don't require word alignment (lightweight)

3. **Case study taxonomy** (qualitative insights)
   - Explanation drift patterns: hashtags, entities, cyber terms, negation
   - Implications for trust & interpretability in security/CTI contexts

---

## 📝 Paper Structure (8 sections)

### 1. Abstract (150-250 words)
- Problem: sentiment + robustness + explainability
- Dataset: X-based sentiment (EN-dominant, noisy metadata)
- Method: train EN, translate-based multilingual eval, explanation consistency
- Results: robustness drop + CTAM drift + overlap metrics

### 2. Introduction
- Cyber discourse on X important for CTI
- Challenge: noisy metadata, multilingual needs
- Gap: few papers measure **cross-lingual stability of explanations**
- 3 contributions (listed above)

### 3. Related Work
- Cyber NLP / social media mining
- Multilingual robustness via translation/perturbation
- XAI for text (IG, LIME) + stability
- Gap: explanation consistency in domain-specific, cross-lingual setting

### 4. Dataset & Preprocessing
- Dataset summary (size, labels, source)
- **Language audit** (metadata vs detected lang, mismatch rate) ← from notebook 01
- Text cleaning (minimal, keep cyber patterns)
- Split (train/val/test, stratified)

### 5. Methodology
- **5.1 Sentiment model**: distilbert-multilingual, training details
- **5.2 Translation-based multilingual setup**: EN→ES/FR silver translations
- **5.3 XAI + Explanation Consistency**:
  - Integrated Gradients
  - CTAM metric definition
  - Top-k overlap metric definition

### 6. Experiments
- Settings: max_length=128, batch=8, epochs=3, early stopping
- Metrics: Accuracy, F1-macro, label flip rate, CTAM drift, Jaccard overlap

### 7. Results & Discussion
- **Table 1**: Prediction robustness (EN vs ES/FR)
- **Table 2**: Explanation consistency (CTAM + overlap)
- **Figure 1**: CTAM distribution + drift
- **Case studies** (3-5 examples of high drift)
- Discussion:
  - Why drift happens? (translation artifacts, tokenization, domain terms lost)
  - Which labels/patterns drift most?
  - Implications for security/CTI applications

### 8. Limitations, Ethics, Conclusion
- Silver translation limitations (no human eval)
- Dataset ToS (non-commercial, restricted use)
- Future: real multilingual data, human evaluation, consistency regularization

---

## ⚠️ Important Notes

### **Nếu dataset KHÔNG cyber-heavy:**

1. **Title**: đổi thành "Cross-lingual Explanation Consistency for Sentiment on X"
2. **CTAM**: đổi thành "Domain Cue Attribution Mass" (DCuAM)
   - Replace `CYBER_PATTERNS` trong `utils.py` bằng:
     - Emotion words: happy, sad, angry, excited, ...
     - Negation: not, never, no, don't, ...
     - Intensifiers: very, really, extremely, ...
     - Hashtags/mentions (domain-agnostic social media cues)
3. **Framing**: focus on "social media sentiment robustness" thay vì "cyber threat"

### **Nếu muốn làm cyber-heavy:**

- Filter dataset bằng cyber keywords (utils.py có `has_cyber_cue()`)
- Create cyber subset trước khi sampling (trong notebook 02)
- Phải justify coverage đủ lớn để train model

---

## 🔧 Troubleshooting

### **OOM (Out of Memory)**

1. Reduce `BATCH_SIZE` (notebook 03, 04, 05)
2. Reduce `SAMPLE_SIZE` / `XAI_SAMPLE_SIZE`
3. Reduce `MAX_LENGTH` (128 → 96 hoặc 64)
4. Use `gradient_accumulation_steps=2` (notebook 03)

### **Translation too slow**

1. Reduce `SAMPLE_SIZE` (2000 → 1000 hoặc 500)
2. Increase `BATCH_SIZE` for translation (16 → 32)
3. Use CPU if GPU memory tight

### **IG too slow**

1. Reduce `XAI_SAMPLE_SIZE` (200 → 100)
2. Reduce `n_steps` in IG (50 → 20)
3. Remove FR, only do EN→ES

---

## 📚 Citation

```bibtex
@inproceedings{yourname2026crosslingual,
  title={Cross-lingual Explanation Consistency for Sentiment on X: A Lightweight Translation-based Multilingual XAI Study},
  author={Your Name},
  booktitle={Conference Name},
  year={2026}
}
```

---

## 📧 Contact

For questions about this code or paper, contact: [your email]

---

**Good luck với paper!** 🚀

Key advantages của approach này:
- Nhẹ (chạy được trên GTX 1650)
- Novel (explanation consistency chưa nhiều paper đo)
- Measurable (2 metrics rõ ràng: CTAM + overlap)
- Practical (insights cho security/CTI applications)
