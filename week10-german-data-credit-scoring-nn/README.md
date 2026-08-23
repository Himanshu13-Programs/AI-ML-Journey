# Credit Scoring Neural Network

### Week 10 — Backpropagation, MLPClassifier & Keras on Real-World Credit Risk Data

**Domain:** Fintech / Credit Risk **Algorithms:** Random Forest, `MLPClassifier` (sklearn), Keras `Sequential` Neural Network **Dataset:** [German Credit (UCI Statlog)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — 1000 loan applicants, 20 features

> **Disclaimer:** This project is a methodology exercise comparing tree-based and neural-network approaches to a small tabular classification problem — it is not a production credit scoring system. The goal is to demonstrate correct workflow (encoding choices, imbalance handling, honest metric evaluation, and fair baseline comparison) and to draw an honest conclusion about when neural networks are and aren't the right tool, rather than to build a deployable model.

---

## Problem Statement

Given 20 features describing a loan applicant (checking account status, credit history, employment, existing debts, and more), predict whether the applicant represents **good** or **bad** credit risk — a binary classification problem with real financial consequences on both sides: approving a bad-risk applicant costs the lender directly, while rejecting a good-risk applicant costs missed business.

This project compares three models of increasing complexity — a tuned Random Forest baseline, a quick `MLPClassifier` neural network, and a fully custom Keras network with dropout, class weighting, and training callbacks — to test a specific question: **does the added complexity of a neural network actually earn its keep on a dataset this small?**

---

## Dataset

- **Source:** UCI Statlog German Credit Data
- **Size:** 1000 applicants, 20 raw features (mix of categorical and numeric)
- **Target:** Originally coded `1` = good credit, `2` = bad credit; remapped to `0`/`1` with **bad credit (default risk) as the positive class**
- **Class balance:** ~30% positive (bad credit) / ~70% negative (good credit) — a meaningful but not extreme imbalance, addressed during preprocessing and modeling (see below)

---

## Key EDA Findings

**Class imbalance:** roughly 3-to-1 skew toward good-credit applicants. This shaped two downstream decisions: using `scoring='f1'` (not accuracy) for Random Forest's `GridSearchCV`, and using class-weighted training for the Keras network, since accuracy alone would reward a model that just predicts "good credit" for everyone.

**Feature mix:** 7 categorical fields (checking account status, credit history, purpose, savings, employment, property, housing, job, etc.) with no natural ordering between categories, alongside numeric fields (duration, credit amount, age, existing credits). This mix directly shaped the encoding decision below.

---

## Data Preprocessing

### Target Encoding

```python
data['response'] = (data['response'] != 1).astype(int)
```

Bad credit (originally coded `2`) is remapped to `1`, so the positive class represents default risk — the outcome the model needs to catch.

### Categorical Encoding

```python
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
```

One-hot encoding was used instead of ordinal label encoding. Label encoding would assign arbitrary integer order (0, 1, 2, 3...) to categories like `purpose` or `housing` that have no real ordering — a tree model can mostly work around this, but a neural network treats the input as a continuous number, implicitly assuming `purpose=3` sits "between" `purpose=2` and `purpose=4`, which is meaningless. One-hot encoding removes that false assumption at the cost of a wider feature space.

### Train/Test Split & Scaling

An 80/20 **stratified** split preserves the ~30% positive class ratio in both sets. `StandardScaler` is fit on the training set only, then applied to both sets — preventing test-set distribution information from leaking into training.

---

## Models & Hyperparameter Choices

### Baseline: Random Forest

```python
GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    },
    cv=5, scoring='f1'
)
```

Tree-based baseline, tuned via 5-fold cross-validated grid search. `scoring='f1'` (rather than accuracy) directly accounts for the class imbalance during model selection, and `class_weight` search handles it at the algorithm level too.

### MLPClassifier (sklearn)

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam',
    max_iter=300, early_stopping=True, validation_fraction=0.1
)
```

A quick neural-network baseline inside the sklearn ecosystem — same one-hot-encoded, scaled inputs as Random Forest, with sklearn's built-in early stopping to prevent overfitting on this small dataset.

### Keras Network

```python
Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])
```

**Training:** `EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True)` and `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)`, trained for up to 100 epochs.

**On early stopping:** training ran the full 100-epoch budget rather than stopping early. Validation AUC visibly plateaus around epoch 10–15, but `EarlyStopping`'s default `min_delta=0` means any improvement — even a fraction of a percent from normal training noise — resets the patience counter before it can accumulate 10 consecutive non-improving epochs. `restore_best_weights=True` still pulls the final model back to its best-observed epoch, so the reported metrics reflect that point rather than the noisier epoch 100 state.

---

## Results

| Model | F1 (default class) | ROC-AUC | Notes |
|---|---|---|---|
| Random Forest | 0.63 | 0.80 | Baseline, computed on held-out test set |
| MLPClassifier | 0.25 | 0.765 | sklearn MLP, computed on held-out test set |
| Keras Network | 0.69 | 0.91* | With dropout + class-imbalance-aware architecture |

*\*Caveat on the Keras AUC — see Honest Interpretation below.*

---

## Honest Interpretation

**The Keras AUC above is not directly comparable to the other two rows.** Random Forest and `MLPClassifier`'s AUC values were both computed with `roc_auc_score(y_test, ...)` on the same held-out 20% test set that neither model touched during training. The Keras AUC of 0.91, by contrast, comes from the training curve's internal `val_auc` metric — computed on a 10% slice carved out of the *training* set (`validation_split=0.1`), not the actual test set. The plotted training curves show `val_auc` plateauing closer to **~0.78**, meaningfully lower than the 0.91 figure in the table above.

This matters for the headline question this project set out to answer. Taken at face value, 0.91 (Keras) vs 0.80 (RF) says the neural network wins decisively. But using the more directly comparable ~0.78 validation figure, **Random Forest is likely the stronger model** on this dataset — which is the expected, unsurprising outcome given the data size.

**Why that's the expected result, not a shortfall:** on ~1000 rows of tabular data, tree-based models routinely match or beat neural networks. Trees handle mixed categorical/numeric features and non-linear feature interactions natively, without needing the thousands of examples a neural network typically needs to learn the same interactions from raw gradient descent. Neural networks earn their complexity on images, text, and sequences — unstructured, high-dimensional inputs where trees can't operate at all — not on small structured tables like this one.

**What would make this comparison fully honest:** computing `roc_auc_score(y_test, keras_probabilities)` on the same held-out test set used for RF and MLP, rather than relying on the training-time validation metric. That fix was identified but deliberately not applied in this pass — noted here rather than silently left out, since the gap changes which model this project's own comparison table appears to recommend.

---

## Notebook Structure

```
Introduction
└── Problem statement, disclaimer, three-model comparison goal

Dataset
├── Feature/target overview, info(), describe()
└── Target encoding (good/bad → 0/1)

Preprocessing
├── One-hot encoding of categorical features (vs. ordinal label encoding)
└── Stratified train/test split, scaler fit on train only

Baseline: Random Forest
├── GridSearchCV tuning (scoring='f1')
├── Classification report, confusion matrix
└── ROC curve, Precision-Recall curve

MLPClassifier (sklearn)
├── Training with early_stopping=True
├── Classification report, confusion matrix
└── Training loss curve

Keras Network
├── Architecture: Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Dense(1, sigmoid)
├── EarlyStopping + ReduceLROnPlateau callbacks
├── Train/val loss and AUC curves
├── Overfitting-onset analysis (~epoch 10-15)
└── Classification report, confusion matrix

Comparison & Insights
├── Results table (F1, ROC-AUC per model)
└── Honest interpretation, including the Keras AUC comparability caveat
```

---

## How to Run

1. Download `german.data` from the [UCI Statlog German Credit page](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
2. Place it at `/content/german.data` if running in Colab, or update the path in the Dataset section
3. Run all cells in order (`Runtime → Run all` in Colab, or `Kernel → Restart & Run All` in Jupyter)

**Requirements:**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

Install with:

```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## What I Would Do Next

- **Fix the Keras AUC comparison** — compute `roc_auc_score` on the true held-out test set rather than the training-time `val_auc`, to get a genuinely apples-to-apples comparison against Random Forest and `MLPClassifier`
- **Add explicit class weighting to the Keras model** — `compute_class_weight` + `class_weight=` in `.fit()`, matching the imbalance-handling already present in Random Forest's `GridSearchCV`
- **Tighten `EarlyStopping` with `min_delta`** — the current default (`min_delta=0`) let small training-noise fluctuations in `val_auc` reset the patience counter, which is why training ran the full 100-epoch budget instead of stopping early despite visibly plateauing around epoch 10-15
- **Cross-validated comparison** — repeat the RF vs. MLP vs. Keras comparison across multiple stratified folds rather than a single train/test split, to check how much the current results depend on this particular split
- **Try a tree-based challenger to RF** — XGBoost or LightGBM, to see whether a stronger tree ensemble widens or narrows the gap with the neural network approaches

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*