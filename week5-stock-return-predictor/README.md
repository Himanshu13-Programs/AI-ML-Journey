# Nifty 50 Stock Return Predictor

### Week 5 — Linear, Ridge & Lasso Regression on Real-World Financial Time-Series Data

**Domain:** Fintech / Quantitative Finance
**Algorithms:** Linear Regression, Ridge Regression, Lasso Regression
**Dataset:** [Nifty 50 Index (^NSEI)](https://finance.yahoo.com/quote/%5ENSEI/) — 3 years of daily OHLCV data via `yfinance`

> **Disclaimer:** This project is a machine learning methodology exercise, not a trading strategy or financial advice. The goal is to demonstrate correct regression workflow — feature engineering, time-series-aware train/test splitting, regularization, and honest evaluation metric reporting — on real market data. Nothing in this notebook should be used to make actual investment decisions.

---

## Problem Statement

Given historical daily price and volume data for the Nifty 50 index, predict **next-day return** — the percentage change in closing price from today to tomorrow.

Raw price is non-stationary (it trends and drifts over time with no fixed mean or variance), which makes it a poor direct input for a linear model. Instead, this project derives stationary, information-carrying features — moving averages, momentum ratios, volatility, and volume change — and uses them to predict the next day's return.

Stock returns are widely known to be close to random over short horizons (an implication of the efficient market hypothesis). This project is explicitly **not** about achieving high predictive accuracy — it's about demonstrating that the full regression pipeline (feature engineering → correct time-based splitting → scaling → regularization → honest metric interpretation) is done correctly, even when the underlying signal is weak.

---

## Dataset

- **Source:** Yahoo Finance, via the `yfinance` Python package
- **Ticker:** `^NSEI` (Nifty 50 Index)
- **Period:** 3 years of daily data, ending at run time
- **Raw size:** 738 trading days
- **Raw columns:** `Open`, `High`, `Low`, `Close`, `Volume`
- **After feature engineering & dropping rolling-window NaNs:** 712 rows, 11 features + 1 target

**Note:** Because this notebook downloads a rolling 3-year window relative to the day it's run, exact metric values will drift slightly each time it's re-executed. This is expected and doesn't affect the methodology or conclusions.

---

## Key EDA Findings

**Target distribution:** The daily return distribution is roughly symmetric — mean and median both sit close to 0.000. However, it shows a sharp peak around the mean with a rapid drop-off moving away from it, consistent with a **leptokurtic** distribution (fatter tails and a sharper peak than a Normal distribution). This is a well-known property of financial returns: most days are quiet, but the tails carry more extreme moves than a Normal distribution would predict.

**Volume:** The dataset contains some zero-volume days, which produce `inf` values when computing percentage change — handled explicitly during preprocessing (see below).

---

## Data Preprocessing

### Target Variable

```python
df['Target'] = df['Close'].pct_change().shift(-1)
```

We shift by `-1` because we're predicting **tomorrow's** return using **today's** features. Without the shift, the model would be trained to predict today's return using today's data — which is trivially available and not a real forecasting problem (a form of data leakage).

### Feature Engineering

| Feature | Description |
|---|---|
| `MA5` | 5-day rolling average of Close — short-term price trend |
| `MA20` | 20-day rolling average of Close — longer-term price trend |
| `MA_ratio` | `MA5 / MA20` — momentum indicator; above 1 signals a short-term uptrend relative to the longer trend |
| `Return` | Daily percentage change in Close |
| `Volatility` | 10-day rolling standard deviation of `Return` — recent price choppiness |
| `Vol_change` | Percentage change in trading Volume day-over-day |

All engineered features are **causal** — each one only uses information available up to and including the current day, with no forward-looking data at any point.

### Handling Infinities

```python
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
```

`pct_change()` on `Volume` produces `inf` on days following a zero-volume day (division by zero). These are explicitly replaced with `NaN` and dropped, alongside the `NaN`s introduced naturally by rolling windows (the first 19 rows can't have a valid `MA20`) and the final row (which has no "tomorrow" to compute a target from).

### Train/Test Split

An 80/20 **time-based** split — the oldest 80% of rows for training, the most recent 20% for testing (569 train / 143 test rows).

> Random splitting on time-series data leaks future information into the training set — a model could effectively "interpolate" between two nearby dates that ended up on opposite sides of a random split. A time-based split respects the temporal order and reflects how the model would actually be used: trained on the past, evaluated on the future.

### Scaling

`StandardScaler` is fit on the training set only, then used to transform both the training and test sets — preventing test-set statistics from leaking into the training process.

---

## Models & Hyperparameter Choices

### Linear Regression

```python
LinearRegression(fit_intercept=True)
```

Baseline model — no regularization, no coefficient penalty.

### Ridge Regression

```python
Ridge(alpha=1.0)
```

Adds an L2 penalty on coefficient magnitude, shrinking all coefficients toward zero without eliminating any entirely.

### Lasso Regression

```python
Lasso(alpha=0.001)
```

Adds an L1 penalty, which can shrink coefficients all the way to exactly zero — effectively performing feature selection.

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.0073 | 0.0099 | -0.0317 |
| Ridge Regression | 0.0073 | 0.0100 | -0.0426 |
| Lasso Regression | 0.0074 | 0.0098 | **-0.0091** |

*Metrics generated programmatically from the notebook — will vary slightly on re-run since the dataset is a rolling 3-year window.*

### Coefficient Behaviour

A grouped bar chart in the notebook compares the fitted coefficients of all three models side by side across every feature. The key visual takeaway: Ridge's coefficients are visibly shrunk compared to Linear's, while **Lasso zeroed out every single feature**:

```
Close       -0.0
High        -0.0
Low         -0.0
Open        -0.0
Volume       0.0
MA5         -0.0
MA20        -0.0
MA_ratio    -0.0
Return      -0.0
Volatility   0.0
Vol_change   0.0
```

With `alpha=0.001`, the L1 penalty was strong enough to eliminate the entire feature set. Because the target (daily return) is on a tiny scale (~0.007 in magnitude), even a small `alpha` value is large relative to the coefficient sizes needed to explain it — meaning Lasso judged that no feature contributed enough predictive signal to be worth a non-zero weight, given the noise level in the data.

---

## Model Comparison

| Metric | Linear | Ridge | Lasso |
|---|---|---|---|
| MAE | 0.0073 | 0.0073 | 0.0074 |
| RMSE | 0.0099 | 0.0100 | **0.0098** |
| R² | -0.0317 | -0.0426 | **-0.0091** |

All three models perform similarly poorly in absolute terms, which is the expected and correct outcome for this problem (see below). Lasso's R² is the least negative, and its total feature elimination suggests that, given the noise level in daily returns, a model that predicts close to the mean return every day is about as good as one that tries to use these particular features — a genuinely useful finding about the limits of this feature set, not a failure of the code.

---

## Honest Interpretation

**R² for all three models is negative** — meaning each model performs slightly *worse* than simply predicting the mean return every day. This is expected, not a bug: daily stock returns are close to random over short horizons, which is exactly what the Efficient Market Hypothesis predicts. If a simple linear model with these features could reliably beat a coin flip on next-day returns, that would be the surprising result.

The value of this project isn't the prediction accuracy — it's demonstrating the full methodology correctly: causal feature engineering with no lookahead leakage, a time-respecting train/test split, scaler fit only on training data, and transparent reporting of results that don't flatter the model. A near-zero or negative R² reported honestly is more useful, and more credible, than a suspiciously high one on this kind of data — which would be a red flag for leakage rather than a sign of a good model.

---

## Notebook Structure

```
Introduction
└── Problem statement, disclaimer

EDA
├── Raw OHLCV overview, describe(), null check

Data & Feature Engineering
├── Target construction (next-day return, with shift explanation)
├── Moving averages, momentum ratio, volatility, volume change
├── Infinity/NaN handling
└── Target distribution plot (mean/median, skew/kurtosis discussion)

Model Training
├── Time-based train/test split (with visualization)
├── Feature scaling (StandardScaler, fit on train only)
└── Linear, Ridge, Lasso training

Model Testing, Evaluation & Model Comparison
├── Results table (MAE, RMSE, R² per model)
└── Coefficient comparison bar chart (Linear vs Ridge vs Lasso)

Honest Interpretation
└── Explanation of near-zero/negative R², efficient market context
```

---

## How to Run

1. Install dependencies (see below)
2. Run all cells in order (`Runtime → Run all` in Colab, or `Kernel → Restart & Run All` in Jupyter)
3. No manual file paths needed — data is pulled live from Yahoo Finance via `yfinance`

**Requirements:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
yfinance
```

Install with:
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy yfinance
```

---

## What I Would Do Next

- **Cross-validated alpha selection** — use `RidgeCV`/`LassoCV` to select regularization strength systematically, rather than fixed `alpha` values, to check whether a different regularization strength changes the "all coefficients zeroed" outcome for Lasso
- **Additional features** — technical indicators like RSI, MACD, or Bollinger Bands, which capture different aspects of momentum and volatility than the moving-average-based features used here
- **Longer/shorter prediction horizons** — testing whether weekly or monthly returns are more predictable than daily returns, which are the noisiest horizon
- **Non-linear models** — tree-based models (Random Forest, XGBoost) to check whether the weak signal here is linear-model-specific or a genuine ceiling on predictability with these features

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*