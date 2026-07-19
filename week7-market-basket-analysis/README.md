# Grocery Basket Analysis
### Week 7 — Apriori & FP-Growth Association Rule Mining on Real Transaction Data

**Domain:** Retail / Market Basket Analysis
**Algorithms:** Apriori, FP-Growth, Association Rule Mining
**Dataset:** [Groceries Dataset — Kaggle](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)

---

## Problem Statement

Identify which products are frequently bought together, so a retailer can act on those patterns
through shelf placement, bundling, or targeted promotions. Unlike a prediction task, the goal
here isn't a single output per customer — it's a set of interpretable rules ("customers who buy X
are more likely to also buy Y") that a product or marketing team can directly turn into decisions.

---

## Dataset

- **Source:** Real grocery store transaction log, downloaded from Kaggle
- **Size:** 38,765 individual item-purchase records, reconstructed into 14,963 customer baskets
- **Basket construction:** each basket = all items one customer (`Member_number`) bought on one
  `Date` — a long-format transaction log grouped into per-visit baskets
- **Catalog:** 167 distinct grocery items
- **Basket size:** ranges from 2 to 11 items, average 2.6 items per basket — every basket already
  contained at least 2 items, so no filtering was needed

**Note:** this project originally used a synthetic fintech product dataset (see Week 6) to
demonstrate association rule mining, since real financial product co-purchase data at this
granularity isn't publicly accessible. This week swaps in real grocery transaction data instead —
the same methodology, applied to genuine (if domain-mismatched) purchasing behavior — to see how
association mining performs on real, unconstrained data rather than curated patterns.

---

## Most Frequent Items

| Item | Support |
|---|---|
| Whole milk | 15.8% |
| Other vegetables | 12.2% |
| Rolls/buns | 11.0% |
| Soda | 9.7% |
| Yogurt | 8.6% |
| Root vegetables | 7.0% |
| Tropical fruit | 6.8% |
| Bottled water | 6.1% |
| Sausage | 6.0% |

High individual popularity doesn't imply meaningful association between items on its own — this
is exactly why the analysis moves to **lift**, not raw frequency, to identify genuine cross-sell
opportunities.

---

## Data Preparation

- **Basket reconstruction:** long-format transaction log grouped by `Member_number` + `Date` into
  one list of items per basket
- **Encoding:** one-hot encoded into a boolean DataFrame via `mlxtend`'s `TransactionEncoder`
  (required input format for both mining algorithms)
- **Support range:** individual item support here ranges roughly 0.02%–15.8%, far lower and more
  spread out than the synthetic dataset's curated 30–70% range. Two factors explain this:
  individual item support is diluted across 167 different products, so no single item can
  dominate the way it could when the synthetic dataset only had 6 curated products. On top of
  that, since the average basket size here is only 2.6 items, any multi-item combination's
  support is naturally even lower than the individual items that make it up — this follows
  directly from the anti-monotone property, where a combination's support can never exceed the
  support of its smallest subset.

---

## Methodology & Parameter Choices

```python
apriori(market_basket, min_support=0.002, use_colnames=True)
fpgrowth(market_basket, min_support=0.002, use_colnames=True)
```

Both Apriori and FP-Growth were run on the same encoded data and produced **identical frequent
itemsets** — confirming both algorithms found the same underlying structure via different search
strategies. **FP-Growth was used for the final analysis**, since it scans the database only twice
regardless of itemset depth, versus Apriori's one scan per itemset level, giving it a speed
advantage at this dataset's scale.

`min_support` was set to **0.2%**, substantially lower than a typical synthetic-data threshold
(0.05–0.08), because this dataset spans 167 different products — with that many options, no
single product is bought often enough to support a higher threshold without returning zero
itemsets.

Rules were ranked and filtered by **lift**, not confidence, since lift better reflects whether two
products are genuinely linked, rather than one product simply being popular on its own — a rule
can show high confidence purely because the consequent item is bought by nearly everyone,
regardless of the antecedent.

---

## Results

**330 frequent itemsets found** at `min_support = 0.002`. **18 directional rules found** with
`lift > 1.0`.

### Top 3 Rules

| Antecedent | Consequent | Lift |
|---|---|---|
| **Curd** | **Sausage** | **1.45** |
| **Brown bread** | **Canned beer** | **1.36** |
| **Frozen vegetables** | **Sausage** | **1.23** |

**Note on symmetry:** curd → sausage and sausage → curd show identical lift (1.45), since lift is
mathematically symmetric — it doesn't depend on direction. Confidence, however, does differ by
direction: curd → sausage has the higher confidence of the two, making it the more reliable
direction to act on.

---

## Cross-Sell Recommendations

1. **Recommendation:** when a customer buys curd, proactively place or suggest sausage nearby
   (lift = 1.45) — this association holds in both directions, though curd → sausage is the more
   reliable direction based on confidence.
2. **Recommendation:** when a customer buys brown bread, proactively suggest canned beer
   (lift = 1.36).
3. **Recommendation:** when a customer buys frozen vegetables, proactively suggest sausage
   (lift = 1.23).

---

## Key Learnings

- **Real association data produces far more modest lift values than curated synthetic data.** The
  strongest real rule found here (1.45) is well below the synthetic exercise's curated 2.5x,
  because lift is mathematically bounded by `1 / support(consequent)` — common items structurally
  cannot produce high lift even when genuinely associated.
- **Small basket size directly limits how strong associations can get.** With an average of only
  2.6 items per basket, there's limited room for multi-item patterns to emerge — most of the 330
  frequent itemsets found are pairs, not larger combinations.
- **`min_support` must be tuned to the dataset, not assumed.** A threshold calibrated for a
  6-product curated dataset (0.05–0.08) returned zero itemsets on a 167-product real dataset;
  0.002 was needed instead.
- **Lift is symmetric, confidence is not.** Two directional rules between the same pair of items
  will always share the same lift, but can have meaningfully different confidence — the higher-
  confidence direction is the more actionable one.
- **Apriori and FP-Growth are guaranteed to agree on frequent itemsets** for the same
  `min_support`, since they solve the same problem via different search strategies — verifying
  this agreement is a useful sanity check before trusting either algorithm's output.

---

## Limitations

- All transactions were weighted equally regardless of date — a production version would likely
  give more weight to recent purchases.
- All customers were weighted equally regardless of spend — a production version would likely
  prioritize patterns among higher-value customers.

## Next Steps

- Re-run this analysis on live transaction data quarterly, since purchase patterns can shift
  seasonally.
- Pilot the top 3 recommendations (e.g. shelf placement or a targeted promotion) and measure
  actual uplift in joint purchases against the lift predicted here.

---

## Notebook Structure

```
Introduction
└── Problem statement, dataset source

Dataset
├── Loading and inspection (38,765 rows, 167 items)
└── Basket reconstruction (group by Member + Date), basket size distribution

Frequent Items
└── Support distribution, real vs. synthetic comparison

Methodology
├── Apriori vs. FP-Growth verification
└── min_support and lift-based filtering rationale

Frequent Itemsets
└── Top 15 by support

Association Rules
├── Full rules table, sorted by lift
└── Symmetric lift / directional confidence note

Cross-Sell Recommendations
└── Top 3 actionable rules

Limitations & Next Steps
```

---

## How to Run

1. Download `Groceries_dataset.csv` from
   [Kaggle](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)
2. Upload to Google Colab or mount your Google Drive
3. Update the file path in **Cell 3** of the notebook to match your file location
4. Run all cells in order (`Runtime → Run all`)

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn` (pre-installed in Colab) plus
`mlxtend` — run `!pip install mlxtend` if not already available.

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*