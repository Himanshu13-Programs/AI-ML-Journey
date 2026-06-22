# Loan Default Classifier
### Week 3 — Decision Tree & KNN on LendingClub Data

**Domain:** Fintech / Credit Risk  
**Algorithms:** Decision Tree, K-Nearest Neighbors  
**Dataset:** [LendingClub Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## Problem Statement

Given a loan application with borrower financial information, predict whether the loan will be **fully repaid (0)** or **default / charged off (1)**.

This is a binary classification problem on a heavily imbalanced dataset — approximately 80% of loans are fully paid and 20% default. The primary challenge is not achieving high accuracy (a model that predicts "Fully Paid" for everything scores 80% accuracy while being completely useless), but achieving meaningful **Recall on the minority class** — catching actual defaulters before they are approved.

---

## Dataset

- **Source:** LendingClub accepted loans (2007–2018), downloaded from Kaggle
- **Size:** ~1.1 million loan records after filtering for deterministic outcomes
- **Target:** `loan_status` mapped to binary — `Fully Paid` → 0, `Charged Off / Default` → 1
- **Class balance:** ~80% Fully Paid, ~20% Charged Off

**Note:** The raw dataset file (`accepted_2007_to_2018Q4.csv`) is not included in this repository due to its size. Download it from the Kaggle link above and update the file path in Cell 3 of the notebook before running.

---

## Features Selected

10 features were selected based on availability at loan origination — no post-outcome data was included to avoid data leakage:

| Feature | Description | Type |
|---------|-------------|------|
| `loan_amnt` | Loan amount requested by borrower | Numeric |
| `int_rate` | Interest rate assigned by LendingClub | Numeric |
| `annual_inc` | Self-reported annual income | Numeric |
| `dti` | Debt-to-income ratio | Numeric |
| `fico_range_low` | Lower bound of borrower's FICO credit score | Numeric |
| `grade` | LendingClub risk grade (A–G) | Ordinal |
| `emp_length` | Employment length in years | Ordinal |
| `term` | Loan term (36 or 60 months) | Ordinal |
| `purpose` | Stated reason for the loan | Categorical |
| `home_ownership` | Borrower's housing status | Categorical |

**Data leakage note:** Post-outcome columns (`total_pymnt`, `recoveries`, `out_prncp`, etc.) were explicitly excluded — these only exist after the loan has concluded and cannot be known at application time.

---

## Data Preparation

- **Missing values:** Numeric columns (`annual_inc`, `dti`) imputed with median; `emp_length` imputed with mode
- **Ordinal encoding:** `grade` (A→0, G→6), `emp_length` (< 1 year→0, 10+ years→10), `term` (extracted as integer 36/60)
- **One-hot encoding:** `purpose` and `home_ownership` via `sklearn.preprocessing.OneHotEncoder` (note: `pd.get_dummies()` is an equivalent alternative)
- **Scaling:** `StandardScaler` applied before KNN only — Decision Trees are threshold-based and do not require scaling
- **Split:** 80/20 train/test with `stratify=y` to preserve class ratio across both sets

---

## Models & Hyperparameter Choices

### Decision Tree
```python
DecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    class_weight='balanced',
    random_state=42
)
```

`max_depth=8` was selected via 5-fold cross-validation sweep across depths 1–20. Test F1 peaked at depth 8 before declining — the classic bias-variance sweet spot. Beyond depth 8, training F1 continued climbing toward 0.82 while test F1 declined to 0.32, confirming overfitting.

`class_weight='balanced'` was added after observing that an unweighted tree at depth=1 achieved 0% F1 on the minority class — it simply predicted "Fully Paid" for everything. Balancing weights each defaulted loan ~4x more heavily than a fully paid one during training, since the class ratio is roughly 80/20.

### K-Nearest Neighbors
```python
KNeighborsClassifier(
    n_neighbors=11,
    weights='uniform'
)
```

`n_neighbors=11` was chosen as a practical compromise — a full CV sweep across k=1–30 was computationally prohibitive on this dataset size (~1.1M rows). 11 was selected as an odd number (avoids tie votes in binary classification) in the range where KNN typically generalises reasonably.

`weights='uniform'` was used since `weights='distance'` showed training times that were prohibitive on this dataset scale.

**Note:** KNN has no `class_weight` parameter, meaning class imbalance cannot be corrected the same way it was for the Decision Tree. This is a known limitation reflected in KNN's results below.

---

## Results

### Decision Tree (max_depth=8, class_weight='balanced')

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fully Paid (0) | 0.88 | 0.64 | 0.74 | 215,748 |
| Default (1) | 0.31 | 0.66 | 0.42 | 53,872 |
| **Weighted Avg** | **0.77** | **0.64** | **0.68** | **269,620** |

**ROC-AUC: 0.704**

### KNN (n_neighbors=11, weights='uniform')

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fully Paid (0) | 0.81 | 0.96 | 0.88 | 215,748 |
| Default (1) | 0.42 | 0.13 | 0.19 | 53,872 |
| **Weighted Avg** | **0.74** | **0.79** | **0.74** | **269,620** |

**ROC-AUC: 0.653**

---

## Model Comparison & Recommendation

| Metric | Decision Tree | KNN | Better Model |
|--------|--------------|-----|--------------|
| Recall (Default) | **0.66** | 0.13 | Decision Tree |
| Precision (Default) | 0.31 | **0.42** | KNN |
| F1 (Default) | **0.42** | 0.19 | Decision Tree |
| ROC-AUC | **0.704** | 0.653 | Decision Tree |

**Why Recall matters more than Precision here:**

A False Negative (approving a loan that defaults) means the lender loses the principal amount — a direct, unrecoverable financial loss. A False Positive (rejecting a loan that would have been repaid) means a missed lending opportunity — a smaller, recoverable cost. Since the cost of missing a real defaulter significantly exceeds the cost of a false alarm, **Recall on the Default class is the primary metric for this use case.**

**Recommendation:** The Decision Tree with `class_weight='balanced'` is the stronger model for a lender — it catches 66% of actual defaulters versus KNN's 13%, and has a higher AUC (0.704 vs 0.653). KNN's higher Precision (0.42 vs 0.31) is a secondary advantage that does not offset its dramatically lower Recall.

Both models still leave meaningful room for improvement — neither achieves strong enough Recall for production use on this imbalanced dataset.

---

## Key Learnings

- **Accuracy is a misleading metric on imbalanced data.** A model predicting "Fully Paid" for all 1.1M loans scores ~80% accuracy while catching zero defaults. Always evaluate imbalanced classification with Recall, F1, and AUC.
- **`class_weight='balanced'` is a simple, high-impact fix for class imbalance** in tree-based models — it took Recall from 0% to 66% without any changes to data or model architecture.
- **KNN has no class imbalance correction mechanism**, making it structurally weaker than Decision Trees for heavily imbalanced problems unless combined with resampling techniques.
- **Data leakage is the most dangerous trap in this dataset.** Post-outcome columns like `total_pymnt` and `recoveries` must be explicitly excluded — including them produces suspiciously high accuracy by letting the model look at evidence of what already happened.
- **The bias-variance tradeoff is real and visible.** The depth sweep clearly showed train F1 → 0.82 while test F1 → 0.32 at high depth — a textbook overfitting signal, not just a theoretical concept.

---

## Notebook Structure

```
Introduction
└── Problem statement, dataset overview, objectives

Data Preparation
├── Loading selected columns with usecols
├── Filtering deterministic loan outcomes
├── Missing value handling
├── Target encoding (loan_status → binary)
├── Ordinal and one-hot encoding
└── Train/test split with stratification

Model Training & Evaluation
├── Decision Tree — depth sweep (CV), final model, confusion matrix, metrics
└── KNN — final model, confusion matrix, metrics, ROC curve comparison

Conclusion
└── Model comparison, recommendation, business reasoning
```

---

## How to Run

1. Download `accepted_2007_to_2018Q4.csv` from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Upload to Google Colab or mount your Google Drive
3. Update the file path in **Cell 3** of the notebook to match your file location
4. Run all cells in order (`Runtime → Run all`)

**Requirements:** All libraries used (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) are pre-installed in Google Colab — no additional installation needed.

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*