# IPO Subscription Predictor
### Week 9 — Classification on Real IPO Subscription Data, and What Data Leakage Actually Costs You

**Domain:** Fintech / Capital Markets
**Algorithms:** SVC, Random Forest, XGBoost (compared via `GridSearchCV`)
**Dataset:** BSE & NSE Mainboard IPO Subscription Data, manually compiled (2025–2026)

---

## Problem Statement

Predict whether an upcoming Indian Mainboard IPO will be **highly oversubscribed** (subscription multiple ≥ 10x) using only information genuinely available **before** an IPO's subscription window opens — not information that only exists once bidding is already underway or closed.

That last constraint turned out to be the entire story of this project: an earlier version of this notebook reached ~89% accuracy using subscription-category data (QIB/NII/Retail multiples, total applications) — numbers that only exist *because* the subscription window has already happened. Once those were correctly identified as data leakage and removed, the real, honest answer to "can we predict this in advance?" turned out to be **no, not with the data available here** — and that negative result is the actual finding of this project.

---

## Dataset

- **Source:** Real IPO subscription records scraped from BSE/NSE listings (Chittorgarh-style source), covering Jan 2025–Aug 2026
- **Size:** 135 IPOs after cleaning
- **Features used (final, leakage-free set):** `Issue_Amount_Cr`, `Has_Emp_Quota`, `Year`, `month`
- **Scope:** Mainboard IPOs only (the data source itself is Mainboard-specific) — `market_cap_category` (SME vs. Mainboard) was intentionally excluded, since it would be a constant column here, carrying zero information

**Note on missing columns from the original task spec:** the original task asked for `sector`, `price_band_upper`, `lot_size`, and `listing_gain_pct`. None of these were present in the raw scraped source and would require a second data pull (or manual tagging, in the case of `sector`) to add. This is real-world engineering reality: the columns you want and the columns a free public source actually gives you are rarely the same — the project adapted the feature set to what was genuinely available rather than fabricating the rest.

---

## Data Preparation

Several real-world data quality issues had to be resolved, each with a different reason and a different fix:

| Issue | Rows Affected | Fix | Reasoning |
|---|---|---|---|
| Junk separator rows (`============`) from concatenating two source CSVs | 2 | Dropped | Not real data, leftover formatting artifact |
| Still-open IPOs (`[O]` tag, closing date in the future) | 2 | Dropped | Their subscription numbers are live/partial snapshots, not final outcomes |
| `Others (x)`, `Shareholder (x)` | 98.5% / 95.6% missing | Dropped columns | Near-constant if filled — provide no usable signal, not worth keeping |
| `ISIN`, `BSE Script Code`, `NSE Symbol` | ~1 unique value per row | Dropped columns | Pure identifiers, same information as `Company`; unique-per-row columns are an overfitting risk if kept as features |
| `Employee (x)` | 54.1% missing | Kept, split into `Has_Emp_Quota` (flag) + 0-filled value | Missingness here is **structural** (many IPOs simply don't offer an employee quota), not random — confirmed by checking that both large (₹10,000+ cr) and small IPOs appear on both sides of the split. A flag preserves this distinction instead of erasing it |
| 2 REIT/InvIT rows (`Cube Highways Trust`, `Bagmane Prime Office REIT`) | 2 | Dropped | Structurally different financial instruments — they don't report `Retail`/`sNII`/`bNII` the same way ordinary equity IPOs do, so patching would mean fabricating values for a category that doesn't apply |
| Numeric columns stored as text | All numeric columns | Converted via `pd.to_numeric` | Required before any calculation or plotting |

---

## The Central Issue: Data Leakage

Before finalizing the feature set, every subscription-related column was checked for **timing leakage** — whether the value would genuinely be known *before* the event being predicted.

**Test performed:** fit a simple linear regression predicting `Total (x)` (the source of the target variable) from candidate features alone.

| Features Tested | R² Predicting `Total (x)` | Verdict |
|---|---|---|
| `QIB (x)`, `NII (x)`, `Retail (x)` | **0.979** | Severe leakage — these near-perfectly reconstruct the target's source column, since `Total (x)` is essentially a weighted average of them |
| `Applications` alone | **0.538** | Moderate leakage — real predictive-looking signal, but only known once subscription is underway/closing, not beforehand |

Both leaky groups were removed from the final feature set. This is the single most important methodological decision in this project — see **Key Learnings** below for what removing them actually did to model performance.

---

## Methodology & Parameter Choices

Three classifiers were compared via `GridSearchCV` (5-fold CV, `scoring='roc_auc'`):

```python
models = {
    'SVC': {'model': SVC(random_state=42),
             'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}},
    'RandomForest': {'model': RandomForestClassifier(random_state=42),
             'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5]}},
    'XGBoost': {'model': XGBClassifier(random_state=42, eval_metric='logloss'),
             'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}}
}
```

Features were standardized via `StandardScaler` (fit on train only, applied to test) before modeling — necessary for SVC, harmless for the tree-based models. `roc_auc` was chosen over plain accuracy as the CV scoring metric, since it's threshold-independent and better suited to a moderately-balanced binary classification problem (75 vs. 60 — a 55.6%/44.4% split).

---

## Results

### Model Comparison (5-fold CV)

| Model | Best CV ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **SVC** | **0.687** | 0.600 | 0.800 | **0.686** |
| RandomForest | 0.651 | 0.526 | 0.667 | 0.588 |
| XGBoost | 0.632 | 0.550 | 0.733 | 0.629 |

### Final Model (XGBoost) — Test Set

```
              precision    recall  f1-score   support
           0       0.43      0.25      0.32        12
           1       0.55      0.73      0.63        15
    accuracy                           0.52        27
```

**Majority-class baseline accuracy: 55.6%** (always predicting "highly oversubscribed"). The final model's **52% test accuracy is at or below this baseline** — meaning, with genuinely pre-listing-only information, the model does not reliably outperform simply guessing the majority class.

### Feature Importance (Final, Leakage-Free Model)

| Feature | Importance |
|---|---|
| `Year` | 42.8% |
| `month` | 25.7% |
| `Issue_Amount_Cr` | 22.1% |
| `Has_Emp_Quota` | 9.3% |

`Year` and `month` together account for ~70% of the model's decisions — most likely reflecting broad IPO-market-cycle sentiment differences between 2025 and 2026 rather than any property of an individual company or issue.

---

## Key Learnings

- **The headline finding is a negative result, and that's the actual value of this project.** Issue size, employee quota status, and listing timing alone do not meaningfully predict IPO oversubscription. The ~89% accuracy from an earlier iteration was driven almost entirely by subscription-window data (`Applications`, `QIB`/`NII`/`Retail` multiples) — information that only exists once the thing being predicted has already started happening.
- **Leakage doesn't always look like "the exact target column got left in."** The QIB/NII/Retail case (R²=0.979) was severe and mathematical — those columns literally compose `Total (x)`. The `Applications` case (R²=0.538) was subtler: not mathematically derived from the target, but still only knowable at the same time as the outcome itself. Both needed the same fix, for slightly different reasons.
- **"Too good to be true" model performance is a real, checkable signal.** A RandomForest hitting 100% test accuracy in an earlier iteration wasn't a modeling win — it was the first clue that leakage was present, well before the R² check confirmed it.
- **A model failing to beat its baseline is informative, not a bug to hide.** It tells you the available features genuinely don't carry the signal needed — which directly points to what data would actually be required for this problem to be solvable (see Next Steps).

---

## Limitations

- **Small dataset** — 135 IPOs is a limited sample for a 4-feature model; results should be treated as directional, not robust.
- **No sector, listing-gain, or grey market premium (GMP) data** — all plausible sources of genuine pre-listing signal, none available from this data source.
- **`Year`/`month` dominance suggests the model may be partly capturing this specific 2025–2026 market window**, rather than a generalizable pattern that would hold in a different market cycle.
- **No macro/market-sentiment features** (e.g., broader index performance around the listing date) were included, despite `Year`/`month`'s dominance hinting they might matter.

## Next Steps

- Incorporate **grey market premium (GMP)** trends in the days before listing — a genuinely pre-listing, if informal, signal.
- Add **anchor investor data** (which institutions participated, at what price) — typically disclosed ~1 day before public bidding opens, so still legitimately "pre-listing."
- Manually tag or scrape **sector** classification per company, to test the original task's sector-based hypothesis.
- Consider a **reframed problem**: instead of predicting before the window opens, predict the *final* outcome using only **Day 1** partial subscription figures — a "nowcasting" version of this task that's a legitimately different (and more solvable) question than the one attempted here.

---

## Notebook Structure

```
Step 0: Importing Libraries

Step 1: Data Collection and Preprocessing
├── Load & concatenate source CSVs, drop junk rows
├── Filter still-open IPOs, clean company names
├── Missing value inspection and column-specific handling
├── Drop identifier columns, Employee flag + fill
└── Drop REIT/InvIT rows, dtype conversion

Step 2: EDA
└── KDE distributions of all numeric features, split by target

Step 3: Model Tuning & Comparison
└── SVC / RandomForest / XGBoost via GridSearchCV (5-fold CV, ROC-AUC)

Step 4: Final Model Training (Best XGBoost Params)

Step 5: Feature Importance Analysis
├── Native (gain-based) importance
├── Permutation importance
└── SHAP values

Feature Importance Analysis & Business Logic (revised, leakage-free interpretation)
Final Business Insights (revised, leakage-free interpretation)
```

---

## How to Run

1. Obtain the source CSVs (`Live_Mainboard_IPOs_Subscription_Status_from_BSE_and_NSE_...csv`) or use the pre-combined `Combined_IPOs_Subscription_Status.csv`
2. Upload to Google Colab or mount your Google Drive
3. Update the file path(s) in **Step 1** of the notebook to match your file location
4. Run all cells in order (`Runtime → Run all`)

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (pre-installed in Colab) plus `xgboost` and `shap` — run `!pip install xgboost shap` if not already available.

---

*Part of the AI/ML fintech roadmap — Week 9.*
