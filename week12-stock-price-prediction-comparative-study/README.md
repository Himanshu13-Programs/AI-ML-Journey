# Stock Return Prediction: LSTM vs GRU vs 1D-CNN vs LSTM + Company Embeddings

### Can Deep Learning Predict Tomorrow's Return of an Indian Large-Cap Stock?

**Domain:** Finance / Time-Series Forecasting **Algorithms:** LSTM, GRU, 1D-CNN, LSTM + Company Embedding (all in PyTorch) **Data:** Daily closing prices of 8 Nifty 50 stocks from Yahoo Finance (`yfinance`), January 2015 to 30 September 2026

> **Disclaimer:** This project is a methodology exercise on testing whether a prediction model has *real* skill. It is not a trading system and not financial advice. The most important result is a negative one: none of the neural networks beat a simple "always guess the average" baseline, and the project explains why that is a valid and useful finding.

---

## Problem Statement

Given the **previous 30 days of returns** of a stock, can a neural network predict the **return on the next day**?

Four PyTorch models are compared:

1. **LSTM**: a recurrent network that reads the 30 days in order and keeps a memory of what it has seen
2. **GRU**: a lighter version of the LSTM (fewer gates, fewer parameters)
3. **1D-CNN**: a convolutional network that looks for short local patterns instead of reading in order
4. **LSTM + company embedding**: the LSTM plus a small learned "fingerprint" for each company

Every model is judged against two **"do nothing" baselines** that involve no learning at all:

- **`naive_up`**: always predict the average daily return seen in training (a small positive number)
- **`zero`**: always predict 0%

The question this project really asks is not "which network is best?" but: **does any network beat a constant guess, and how do we know the answer is not just luck?** That is why the notebook uses baselines, a paired significance test, and repeated runs with different random seeds.

---

## Dataset

- **Source:** Yahoo Finance through the `yfinance` library
- **Period:** January 2015 to 30 September 2026 (daily closing prices)
- **Adjusted prices:** `auto_adjust=True` corrects prices for stock splits, bonus issues and dividends, so a split does not show up as a fake 50% crash
- **Missing values:** none (`isna().sum()` is 0 for every stock). `dropna()` is kept as a safety net so all stocks share exactly the same dates

| Role | Companies |
|---|---|
| **Training (6)** | RELIANCE, TCS, HDFCBANK, ITC, LT, MARUTI |
| **Held-out (2)** | INFY, SBIN |

The six training companies come from different sectors (energy and retail, IT, banking, FMCG, engineering, autos). The two held-out companies are **never used in training, not even to compute the scaling**. They test whether a model learned something general about how stocks move, or only something specific to the six companies it saw.

**Target:** the daily return, calculated as `(price_today - price_yesterday) / price_yesterday`.

**Why returns and not prices?** Prices drift upward over the years, and today's price is already an excellent guess for tomorrow's price, so *any* model looks accurate on a price chart. Returns have no trend, sit on a similar scale for every stock, and are what traders and risk managers actually care about.

---

## Key EDA Findings

**Daily returns are tiny compared to their noise.** The average daily return is only about 0.03% to 0.08%, while a typical day's swing (standard deviation) is 1.4% to 2.0%. So the average is only about **2% to 5% of one day's swing**. Any predictable part of tomorrow's return is roughly 20 to 50 times smaller than the random part.

**Returns have fat tails.** Most days are small, but a few days are huge (single-day moves of 10% to 17% occur, and SBIN has one day of +27.7%). This matters because MSE (the main error metric) is dominated by a handful of extreme days, so small gaps between models can be moved by just a few observations. The notebook prints the largest single-day moves as a **data-quality check**, since each one should match a real market event and not a data glitch.

**Stocks move together.** On the price history chart (rebased to 100, log scale), all eight stocks rise and fall broadly together, and every one drops sharply in March 2020 (the COVID-19 crash), which lies inside the training period.

**The direction of returns has almost no memory, but the size of moves does.** The autocorrelation plots show that past returns barely predict future returns, while squared returns (a simple volatility measure) show *volatility clustering*: calm days follow calm days and wild days follow wild days. This set the expectation before any training: predicting the direction of tomorrow's return is very hard, while predicting how *volatile* tomorrow will be is much easier. That idea comes back in the next-steps section.

---

## Data Preprocessing

### Windows: Turning a Return History into Training Examples

```python
WINDOW = 30
# input  = returns on days 1..30
# target = return on day 31
# then slide forward by one day and repeat
```

Each example is **the previous 30 daily returns (input)** and **the next day's return (target)**. This is done separately for each stock, so a window never mixes two companies. A window may reach back into the previous period (the first 2022 windows use late-2021 returns as input), which is fine because it only uses the past.

### Scaling

```python
mu  = returns.loc[train_mask, train_tickers].values.mean()
std = returns.loc[train_mask, train_tickers].values.std()
scaled = (returns - mu) / std
```

Returns are standardised using **one mean and one standard deviation computed from the training period of the six training stocks only**.

- Using only training data avoids **leakage**: if test-period numbers helped set the scale, test information would sneak into training
- Using **one shared scale** (instead of one per stock) keeps differences in volatility visible to the model
- Predictions are converted back to real returns with `* std + mu`, so all final metrics are in real return units

### Chronological Split (No Shuffling)

| Split | Period | Stocks | Samples | Used for |
|---|---|---|---|---|
| Train | Jan 2015 to Dec 2021 | 6 training stocks | 10,182 | Learning the weights |
| Validation | 2022 | 6 training stocks | 1,488 | Deciding when to stop training |
| Test, "seen" | Jan 2023 to Sep 2026 | 6 training stocks | 5,574 | Same companies, future dates |
| Test, "held-out" | Jan 2023 to Sep 2026 | INFY, SBIN | 1,858 | New companies, same future dates |

**Why chronological and not a random shuffle:** a shuffled split lets the model train on days that come *after* some of its test days, which is looking into the future.

**Why two test sets:** both cover the same dates, so any difference between "seen" and "held-out" comes only from the company. This separates "does it generalise in time?" from "does it generalise to new companies?"

---

## Models & Hyperparameter Choices

All four networks share the same shape: **encoder → dropout → one linear layer → one number** (the predicted next-day scaled return). Only the *encoder*, the part that reads the 30 days, differs. This keeps the comparison fair.

| Model | How the encoder works | Approx. parameters |
|---|---|---|
| **LSTM** | Reads days in order, with a forget gate, input gate and output gate controlling a hidden state and a long-term cell state | ~17,200 |
| **GRU** | Same idea with two gates and no separate cell state | ~12,900 |
| **1D-CNN** | Slides 3-day-wide filters across the window to detect short patterns, two conv layers, then averages over time | ~6,400 |
| **LSTM + Embedding** | LSTM plus a learned 4-number vector per company, joined to the 64-number summary (64 + 4 = 68) | ~17,000 |

### The Network Code

```python
class Net(nn.Module):
    def __init__(self, kind, hidden=64, dropout=0.2):
        super().__init__()
        self.kind = kind
        if kind in ("LSTM", "GRU"):
            self.body = {"LSTM": nn.LSTM, "GRU": nn.GRU}[kind](1, hidden, batch_first=True)
        else:
            self.body = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv1d(32, hidden, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
        self.drop, self.fc = nn.Dropout(dropout), nn.Linear(hidden, 1)
```

### Training Settings

| Setting | Value |
|---|---|
| Loss | MSE (mean squared error) |
| Optimiser | Adam, learning rate 0.001 |
| Batch size | 64 |
| Maximum epochs | 50 |
| Early stopping | Patience 5, on validation loss |
| Hidden size / dropout | 64 / 0.2 |
| Random seed | 42 (then seeds 0 to 4 for the robustness check) |

**How early stopping works here:** after each epoch the model is tested on the 2022 validation data. If the validation loss improves, a copy of the weights is saved. If it fails to improve 5 epochs in a row, training stops and the **best saved weights are reloaded**. The model that gets evaluated is therefore the best one on validation, not the last one trained.

**Training result:** validation loss was almost identical for every model (about 0.851 to 0.855 in scaled units) and barely moved between the first epoch and the best one. The LSTM, for example, went from 0.8546 to 0.8512. Training stopped after 14 epochs for the LSTM (best at epoch 9), 11 for the GRU (best at epoch 6), 7 for the 1D-CNN (best at epoch 2) and 12 for LSTM + Embedding (best at epoch 7). A network that found real structure would show the loss falling clearly for several epochs. Here the models learn the average almost immediately, then have nothing more to learn.

### The Company Embedding Variant

An **embedding** is a small lookup table that maps each company ID to a vector of 4 learned numbers. The problem: the held-out companies (INFY, SBIN) have no ID in training. The fix is an extra **UNKNOWN** ID. During training, 20% of examples have their true ID randomly replaced by UNKNOWN (`p_unk=0.2`), so the model learns a sensible UNKNOWN vector. Held-out companies always get UNKNOWN at test time.

### The Two Baselines

```python
P["naive_up"] = {s: np.full(len(T[s]), max(mu, 1e-6)) for s in ["val", "seen", "held"]}
P["zero"]     = {s: np.zeros(len(T[s])) for s in ["val", "seen", "held"]}
```

---

## How the Models Are Evaluated

| Metric | What it tells you |
|---|---|
| **Directional accuracy (`dir_acc`)** | Share of days where the predicted sign (up or down) matches the real one. The bar to beat is the always-up baseline (49.28% seen, 50.27% held-out), not 50% |
| **RMSE** | Typical size of the error in return units (0.0134 means 1.34 percentage points) |
| **`pct_up`** | Share of days the model predicted "up". Close to 100% means the model says almost the same thing every day |
| **MSE and skill score** | `Skill = (1 - MSE_model / MSE_naive_up) × 100%`. Positive means better than the constant guess, 0 means the same, negative means worse than doing nothing |
| **Paired t-test vs `naive_up`** | For every date, the average squared-error difference between the model and the baseline. A t-stat above 2 in absolute value is unlikely (about 5%) to be pure chance |
| **Information Coefficient (IC)** | Spearman rank correlation between predicted and actual returns. 0 means no relationship. Pure noise gives about ±0.013 (seen) or ±0.023 (held-out) |
| **Multi-seed robustness** | Retrain all four networks with 5 other seeds and report mean ± standard deviation, so a gap only counts if it is clearly bigger than the run-to-run spread |

---

## Results

All numbers below use the test period (January 2023 onwards), which was never touched during training or early stopping.

### Single Run (seed 42)

| Model | MSE seen (×1e-4) | Skill seen | MSE held-out (×1e-4) | Skill held-out | Dir. accuracy seen / held-out |
|---|---|---|---|---|---|
| `zero` baseline | 1.7851 | +0.13% | 2.4346 | +0.07% | n/a |
| 1D-CNN | 1.7850 | +0.13% | 2.4350 | +0.05% | 49.37% / 50.27% |
| `naive_up` baseline | 1.7874 | 0.00% | 2.4363 | 0.00% | 49.28% / 50.27% |
| GRU | 1.7893 | -0.11% | 2.4389 | -0.11% | 49.16% / 50.27% |
| LSTM + Embedding | 1.7909 | -0.20% | 2.4429 | -0.27% | 48.78% / 50.22% |
| LSTM | 1.7966 | -0.52% | 2.4483 | -0.49% | 49.25% / 50.48% |

### Paired Test Against `naive_up`

Only **one of ten** rows passes the t-stat 2 mark: the LSTM on seen companies (t = 2.10), and it is in the **wrong direction** (worse than `naive_up`). Every other row has |t| below 1.6, meaning no clear difference from the baseline.

### Information Coefficient

| Model | IC seen | IC held-out |
|---|---|---|
| LSTM | -0.0042 | 0.0108 |
| GRU | -0.0053 | 0.0060 |
| 1D-CNN | -0.0014 | -0.0046 |
| LSTM + Embedding | -0.0062 | 0.0117 |

Every value sits inside the noise band, and the sign flips between seen and held-out for three of the four models.

### Five-Seed Robustness Check (mean ± std)

| Model | MSE seen (×1e-4) | MSE held-out (×1e-4) |
|---|---|---|
| LSTM | 1.7910 ± 0.0008 | 2.4420 ± 0.0010 |
| GRU | 1.7899 ± 0.0015 | 2.4391 ± 0.0014 |
| 1D-CNN | 1.7867 ± 0.0013 | 2.4373 ± 0.0011 |
| LSTM + Embedding | 1.7888 ± 0.0019 | 2.4413 ± 0.0048 |
| `naive_up` baseline | 1.7874 | 2.4363 |
| `zero` baseline | 1.7851 | 2.4346 |

---

## Honest Interpretation

**No neural network beats the constant baseline, and the LSTM is slightly but consistently *worse* than doing nothing.** Over five seeds, the 1D-CNN, GRU and LSTM + Embedding are all within about 2 run-to-run spreads of `naive_up`, so they cannot be told apart from it. The LSTM is worse than `naive_up` by about 4.5 spreads (seen) and 5.7 spreads (held-out). That is consistent across seeds, but tiny: about 0.2% of the MSE.

**The networks learned the average return and little else.** Three pieces of evidence point the same way:

- Validation loss was flat from the first epoch
- `pct_up` was 96% to 100%, so the models predict "up" on almost every day
- RMSE matched the baseline to four decimals (1.34% seen, 1.56% held-out). The 1D-CNN's directional accuracy has a spread of 0.00 and equals the baseline exactly, meaning it predicts "up" on every single test day

**A misleading chart:** the price-level plot looks excellent for every model. That is not evidence of skill. A single day's price change is only 1% to 2%, so "tomorrow's price = today's price" already looks perfect. Looking only at that plot, or only at an RMSE of about 1.3%, would have suggested success.

**Why this is the expected result, not a shortfall:**

1. **Very low signal-to-noise.** The average daily return is only 2% to 5% of a typical day's swing, and past returns show almost no autocorrelation
2. **The models collapsed to a constant** because there was nothing more to learn from 30 past returns
3. **More parameters means more room to fit noise.** The MSE ranking follows model size (LSTM about 17,000, GRU about 13,000, CNN about 6,400 parameters). With only four models this is a consistent pattern, not a proof

This agrees with the efficient-market idea that yesterday's information is already in today's price. It does **not** prove that nothing can be predicted. It shows that the previous 30 daily returns alone contain no usable signal for next-day returns on these stocks.

**Two small caveats worth knowing:**

- Ten paired comparisons were made, and each has a roughly 1-in-20 chance of crossing t = 2 by luck, so one borderline hit is not surprising
- Seed 42 was an unlucky seed for the LSTM (seen MSE 1.7966 vs 1.7910 ± 0.0008 for seeds 0 to 4), so the multi-seed table is the more trustworthy summary

**What protected this conclusion:** the evaluation design. Baselines, a paired test and repeated seeds are what showed that nothing was learned. A clearly reported negative result is still a valid result.

---

## Notebook Structure

```
Introduction
└── Question, four models, two baselines

Data
├── Yahoo Finance download (8 Nifty 50 stocks, adjusted prices)
└── Training vs held-out company split

Daily Returns
└── Returns vs prices, summary statistics

Exploratory Analysis
├── EDA 1: Price history (rebased, log scale)
├── EDA 2: Return distribution, fat tails, largest single-day moves
└── EDA 3: Autocorrelation of returns and squared returns

Preparing the Data
├── 30-day windows
├── Scaling (fit on training period only)
├── Chronological train / validation / test split
└── First baseline: naive_up

PyTorch Setup
└── Device selection, tensor helper

The Models
├── LSTM, GRU, 1D-CNN (shared Net class)
├── Training loop with early stopping (fit_predict)
└── Variant: LSTM + company embedding (UNKNOWN ID for held-out stocks)

Evaluation
├── Directional accuracy, RMSE, pct_up
├── Actual-vs-predicted plots (return, price level, rolling accuracy)
├── MSE and skill score vs baselines (adds the zero baseline)
├── Paired test against naive_up
├── Per-company MSE
└── Information coefficient + confidence-interval chart

Robustness
├── Five-seed retraining (mean ± std)
└── Verdict table and conclusion

Limitations and Next Steps
```

---

## How to Run

1. Make sure you have an internet connection, because the notebook downloads prices from Yahoo Finance when it runs
2. Open `stock_price_prediction_comparative_study.ipynb` in Jupyter or Google Colab
3. Run all cells in order (`Kernel → Restart & Run All` in Jupyter, or `Runtime → Run all` in Colab)

**Notes:**

- A GPU is used automatically if available, but the models are small and run fine on CPU
- The five-seed robustness section retrains all four networks five times, so it is the slowest part
- Because Yahoo Finance data can be revised and the end date is fixed, small differences from the numbers above are possible

**Requirements:**

```
numpy
pandas
matplotlib
scipy
torch
yfinance
```

Install with:

```
pip install numpy pandas matplotlib scipy torch yfinance
```

---

## What I Would Do Next

- **Add more input features:** the models only see past returns. Try volume, the Nifty 50 index return, volatility measures and calendar effects
- **Predict volatility instead of direction:** EDA 3 shows that the *size* of moves persists from day to day. Predicting next-day volatility is the natural follow-up, and it is what risk teams in finance actually model
- **Try longer horizons:** the next-day horizon is where noise is largest. Weekly or monthly returns are worth testing
- **Use walk-forward evaluation:** retrain on a rolling window instead of a single train / validation / test split, for a more robust estimate
- **Widen and de-bias the stock universe:** eight stocks picked from today's Nifty 50 are all companies that survived and did well (survivorship bias), and the six training stocks move together, so the 10,182 training samples carry less independent information than the count suggests

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*
