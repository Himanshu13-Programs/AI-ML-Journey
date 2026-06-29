# Credit Card Fraud Detection
### Week 4 — Naive Bayes & Random Forest on Real-World Imbalanced Data

**Domain:** Fintech / Fraud Detection  
**Algorithms:** Gaussian Naive Bayes, Random Forest  
**Dataset:** [Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Problem Statement

Given anonymised credit card transaction data, predict whether a transaction is **legitimate (0)** or **fraudulent (1)**.

This is a binary classification problem with extreme class imbalance — only **0.17% of transactions are fraud**. A model that predicts "legitimate" for every single transaction achieves 99.83% accuracy while catching **zero fraud cases**. This makes accuracy a completely useless metric here, and motivates the use of Precision, Recall, F1, ROC-AUC, and — most importantly — the **Precision-Recall curve and AUPRC**.

The primary business objective is minimising **False Negatives** (missed fraud) — a missed fraud case costs the bank the full transaction amount and damages customer trust. False Positives (legitimate transactions wrongly flagged) only cause temporary inconvenience.

---

## Dataset

- **Source:** ULB Machine Learning Group via Kaggle
- **Size:** 284,807 transactions (283,726 after removing 1,081 duplicates)
- **Period:** September 2013, European cardholders, 2 days
- **Features:** 30 total
  - `V1–V28`: PCA-transformed anonymised features (pre-scaled by Kaggle)
  - `Amount`: transaction amount in EUR (scaled by us)
  - `Hour`: hour of day derived from `Time` (0–23)
- **Target:** `Class` — 0 = Legitimate, 1 = Fraud
- **Class balance:** 99.83% legitimate, 0.17% fraud (~492 fraud cases)

**Note:** The raw dataset file (`creditcard.csv`) is not included in this repository due to size. Download it from the Kaggle link above and update the file path in Cell 2 of the notebook before running.

---

## Key EDA Findings

**Transaction Amount:**
Both fraud and legitimate transactions are concentrated in the 0–500 EUR range. However, legitimate transactions spread up to 25,000+ EUR, while fraudulent transactions almost never exceed 1,500 EUR — suggesting fraudsters deliberately keep amounts small to avoid triggering detection thresholds.

**Time of Day:**
Legitimate transactions follow a clear daily cycle — dipping around night hours, peaking during business hours. Fraudulent transactions are distributed uniformly across all hours — fraud happens regardless of time of day, unlike legitimate spending behaviour.

---

## Data Preprocessing

### Feature Engineering
- `Time` converted to `Hour` (hour of day, 0–23) using:
  `Hour = (Time % 86400) // 3600`
  This preserves the time-of-day signal without the raw cumulative seconds
  that carry no meaningful pattern.

### Scaling
`StandardScaler` applied to all features **after** the train/test split — fitted on training data only, then used to transform the test set.

> Fitting the scaler on the full dataset would leak test set statistics (mean and standard deviation) into the model — a form of data leakage that produces dishonestly optimistic evaluation results.

### Train/Test Split
80/20 split with `stratify=y` to preserve the 0.17% fraud ratio in both training and test sets.

### SMOTE — Training Set Only
SMOTE (Synthetic Minority Over-sampling TEchnique) applied to the training set to balance the 0.17% / 99.83% class split. SMOTE generates synthetic fraud samples by interpolating between existing fraud cases.

> SMOTE is applied ONLY to the training set — never to the test set. Applying it to test data would introduce artificial samples that don't represent real-world distribution, making evaluation metrics dishonestly  optimistic.

---

## Models & Hyperparameter Choices

### Gaussian Naive Bayes
```python
GaussianNB(var_smoothing=1e-9)
```

Chosen as a fast, interpretable baseline. Assumes all features are conditionally independent given the class — an assumption that is **partially violated here**: PCA features (V1–V28) are independent by construction, but `Amount` and `Hour` correlate with each other and with the PCA features.

### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
```

### Random Forest — Hyperparameter Choices

`n_estimators=100` — 100 trees, sufficient for stable predictions on this dataset.

`class_weight='balanced'` — assigns higher loss weight to fraud samples during training. Note: since SMOTE already balanced the training set to 50/50, this parameter has minimal additional effect here. Both mechanisms address class imbalance — SMOTE by generating synthetic samples, class_weight by adjusting the loss function. Using both is slightly redundant but harmless.

`oob_score=True` — provides a free cross-validation estimate using the ~1/3 of bootstrap samples each tree never saw during training.

`random_state=42` — ensures reproducible results across runs.
`n_jobs=-1` — trains all 100 trees in parallel across all CPU cores.
---

## Results

### Gaussian Naive Bayes

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Legitimate (0) | ~100% |  ~98% | ~99% |
| **Fraud (1)** | **5%** | **81%** | **10%** |

**ROC-AUC: 94% | AUPRC: 8%**

Despite 81% Recall, Precision of only 5% means 19 out of every 20 fraud flags are false alarms — legitimate customers being wrongly blocked. Operationally unacceptable despite high Recall.

The inflated ROC-AUC (94%) is misleading — the massive True Negative pool (99.83% legitimate transactions) keeps FPR artificially low, making the ROC curve look optimistic. The AUPRC of 8% tells the real story.

---

### Random Forest

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Legitimate (0) | ~100% | ~100% | ~100% |
| **Fraud (1)** | **90%** | **77%** | **83%** |

**ROC-AUC: 95% | AUPRC: 81%**

Note: ~100% accuracy is misleading for the same reason as NB — the test set is 99.83% legitimate. Even misclassifying all fraud would give 99.83% accuracy. We focus on fraud class metrics only.

---

## Model Comparison

| Metric | Gaussian NB | Random Forest |
|--------|-------------|---------------|
| Precision (Fraud) | 5% | **90%** |
| Recall (Fraud) | **81%** | 77% |
| F1-Score (Fraud) | 10% | **83%** |
| ROC-AUC | 94% | **95%** |
| **AUPRC** | 8% | **81%** |

*All metrics measured on the positive (Fraud) class.*

---

## Conclusion & Recommendation

**Random Forest is the production recommendation.**

AUPRC is the definitive metric on this dataset — with only 0.17% positive class, it is the only metric not distorted by the massive legitimate transaction pool. Random Forest's AUPRC of 81% vs Naive Bayes's 8% makes the choice clear.

While Naive Bayes has slightly higher Recall (81% vs 77%), its 5% Precision makes it operationally unviable — it would block 19 legitimate customers for every 1 real fraud case it correctly catches, generating massive customer complaints and operational overhead.

Random Forest achieves 90% Precision with 77% Recall — a far more balanced and deployable tradeoff. When it flags a transaction as fraud, it is right 9 times out of 10. And it still catches 77% of all real fraud cases — meaning only ~1 in 4 fraud cases slips through.

**The business logic:** A missed fraud (False Negative) costs the bank the full transaction amount and damages customer trust — a direct, unrecoverable loss. A false alarm (False Positive) blocks one legitimate transaction temporarily — a far smaller, recoverable cost. Minimising False Negatives is the priority, and Random Forest achieves this while maintaining operationally viable Precision.

---

## Feature Importance

The Random Forest identified the following as the most predictive features (based on Gini importance across all 100 trees):

Top features by importance were PCA components V4, V11, V12, V14, V17 alongside `Amount` — suggesting the PCA-transformed anonymised features carry the most discriminative signal for fraud detection, while raw transaction amount also contributes meaningfully.

---

## Notebook Structure

```
Introduction
└── Problem statement, dataset overview, business objective

EDA
├── Transaction amount distribution (fraud vs legitimate)
├── Time-of-day distribution (fraud vs legitimate)
├── Duplicate detection and removal
└── Class imbalance quantification

Data Preprocessing
├── Hour feature engineering from Time
├── Train/test split with stratification
├── StandardScaler (fit on train, transform test)
└── SMOTE oversampling (training set only)

Model Training & Evaluation
├── Gaussian Naive Bayes — confusion matrix, metrics, ROC, PR curve
└── Random Forest — confusion matrix, metrics, ROC, PR curve, feature importance

Model Comparison
└── Side-by-side metric table, AUPRC analysis

Conclusion
└── Recommendation with business justification
```

---

## How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Upload to Google Colab or mount your Google Drive
3. Update the file path in **Cell 2** of the notebook to match your file location
4. Run all cells in order (`Runtime → Run all`)

**Requirements:** All standard libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) are pre-installed in Google Colab.
Install `imbalanced-learn` if not available:
```python
!pip install imbalanced-learn
```

---

## What I Would Do Next

This project used Naive Bayes and Random Forest as a baseline comparison. To push performance further on this dataset, the natural next steps are:

- **XGBoost / LightGBM** — gradient boosting typically outperforms Random Forest on tabular fraud data (coming in Week 9)
- **Threshold tuning** — lowering the classification threshold from 0.5   to 0.3 would increase Recall at the cost of some Precision, which may be the right tradeoff depending on the bank's risk appetite
- **Cost-sensitive learning** — assigning explicit dollar costs to FP and
  FN rather than using class weights

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*