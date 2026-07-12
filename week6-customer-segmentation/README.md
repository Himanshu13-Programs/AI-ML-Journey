# Fintech Customer Segmentation
### Week 6 — K-Means Clustering & PCA on Synthetic Transaction Data

**Domain:** Fintech / Customer Analytics  
**Algorithms:** K-Means Clustering, Principal Component Analysis (PCA)  
**Dataset:** Synthetic — generated in-notebook, no external download required

---

## Problem Statement

Segment fintech customers into behavioural personas based on spending patterns, so a fintech
company can route different customer types to different product offers instead of a
one-size-fits-all campaign.

Unlike Week 3's classification task, this is **unsupervised learning** — there is no target label
to predict, and the deliverable is not a prediction but a set of business-interpretable customer
groups. The core challenge is not fitting a model (K-Means is a few lines of code) but correctly
choosing the number of clusters, verifying the clusters are meaningful rather than arbitrary, and
translating raw cluster statistics into named personas with actionable business recommendations.

---

## Dataset

- **Source:** Synthetic, generated with `np.random.normal()` — 2000 customers, 6 spending
  features, 4 designed personas (500 customers each)
- **Personas:** High Spender, Conservative Saver, Young Investor, Bill Payer — each with
  persona-specific means/standard deviations across all 6 features
- **Ground truth:** Persona labels are retained separately and used **only for evaluation**,
  never fed into K-Means during fitting

**Note:** Real open banking transaction data at this granularity requires formal data-sharing
agreements and isn't publicly accessible at the individual-category level used here. Synthetic
data with realistic, persona-specific distributions demonstrates the full segmentation
methodology correctly, while also providing ground-truth labels — something real unlabeled data
would not allow, making it possible to verify the clustering is actually recovering meaningful
structure rather than an arbitrary split.

---

## Features Generated

| Feature | Description | Type |
|---|---|---|
| `avg_txn_amount` | Mean transaction amount | Numeric |
| `txn_frequency` | Transactions per month | Numeric |
| `retail_ratio` | Fraction of spend on retail | Ratio (0–1) |
| `investment_ratio` | Fraction of spend on SIPs/investments | Ratio (0–1) |
| `bill_ratio` | Fraction of spend on bills/EMIs | Ratio (0–1) |
| `entertainment_ratio` | Fraction of spend on food/entertainment | Ratio (0–1) |

**Data validity note:** Gaussian sampling can produce physically impossible values (negative
amounts, ratios outside [0,1], or the four ratio columns not summing to 1). These are clipped and
renormalized during preprocessing before any modelling occurs.

---

## Data Preparation

- **Outlier/validity clipping:** `avg_txn_amount` and `txn_frequency` clipped to sensible minimums;
  the 4 ratio columns clipped to [0,1] and renormalized to sum to 1 per row
- **Scaling:** `StandardScaler` applied to all 6 features — required since `avg_txn_amount` is in
  the thousands while the ratio columns are 0–1, and K-Means uses Euclidean distance, which would
  otherwise be dominated entirely by the largest-magnitude feature
- **Split:** None — unsupervised learning uses the full dataset; there is no target to leak

---

## Models & Hyperparameter Choices

### Choosing k
```
Elbow method (WCSS) + Silhouette score, swept across k = 2 to 8
```
Both methods were checked together rather than relying on either alone. The elbow curve bends
sharply at k=4, and the silhouette score also peaks at k=4 — agreement between the two gives more
confidence than either metric individually, and matches the 4 personas the data was designed
around.

### K-Means (final model)
```python
# Model A — fit directly on the 6 scaled features
KMeans(n_clusters=4, init='k-means++', random_state=42)

# Model B — fit on the top-4 PCA components (final model used for profiling)
PCA(n_components=4)  →  KMeans(n_clusters=4, init='k-means++', random_state=42)
```
`k-means++` initialization was used over random init, since it seeds centroids more intelligently
and reduces the risk of converging to a poor local minimum. Two versions of the final model were
compared — one on the raw scaled features, one on the top-4 PCA components — since PCA-reduced
clustering achieved a meaningfully higher silhouette score (see Results), likely because
compressing to the highest-variance components removes some noise/redundancy across the
correlated ratio features.

---

## Results

### Elbow & Silhouette Sweep (k = 2 to 7)

| k | Inertia (WCSS) | Silhouette Score |
|---|---|---|
| 2 | 6189.93 | 0.137 |
| 3 | 2222.31 | 0.138 |
| 4 | **1652.34** | **0.319*** |
| 5 | 1564.23 | 0.244 |
| 6 | 1414.01 | 0.199 |
| 7 | 1324.77 | 0.025 |

*Silhouette peaks at k=4 in this sweep, confirming the elbow method's choice.* \*Note: this
sweep's silhouette was computed on unscaled features — see the fix noted above; the correctly
scaled k=4 silhouette is 0.526 (Model A below).

### Final Model Comparison (k=4)

| | Model A — Raw Scaled Features | Model B — Top-4 PCA Components |
|---|---|---|
| Inertia (WCSS) | 1652.34 | **1383.58** |
| Silhouette Score | 0.526 | **0.560** |

**Model B (PCA-reduced) selected as the final model** for cluster profiling and business insights.

---

## Cluster Profiling & Persona Assignment

| Cluster | avg_txn_amount | txn_frequency | retail_ratio | investment_ratio | bill_ratio | entertainment_ratio | Persona |
|---|---|---|---|---|---|---|---|
| 0 | 1492 | 24.8 | 0.21 | **0.47** | 0.16 | 0.16 | Young Investor |
| 1 | 2234 | 9.9 | 0.11 | 0.05 | **0.72** | 0.11 | Bill Payer |
| 2 | 3500 | 45.2 | 0.45 | 0.05 | 0.15 | **0.35** | High Spender |
| 3 | 621 | 11.9 | 0.16 | 0.11 | **0.62** | 0.11 | Conservative Saver |

Persona names were assigned by comparing each cluster's feature means against the known
ground-truth persona parameters used to generate the data — not by eyeballing the raw table alone.

---

## Business Insights & Recommendation

| Persona | Key Signal | Recommended Product Action |
|---|---|---|
| High Spenders | Highest avg transaction (₹3500), highest frequency (45/mo), retail + entertainment heavy | Premium rewards credit card with cashback on retail/dining |
| Young Investors | investment_ratio = 0.47 — far above every other cluster | Cross-sell SIP top-ups, mutual fund advisory, or a robo-advisory product |
| Bill Payers | bill_ratio = 0.72, lowest frequency (9.9/mo), few large transactions | EMI restructuring, auto-pay bill consolidation, or a BNPL product for recurring payments |
| Conservative Savers | Lowest avg transaction (₹621), essentials-dominated spend | High-interest savings account or recurring deposit product |

**Why this segmentation is useful:** these four personas justify meaningfully different product
strategies rather than a single generic offer — a fintech acting on this segmentation could route
each persona to a different acquisition campaign (credit card vs. investment vs. lending vs.
savings) instead of broadcasting the same offer to all 2000 customers.

---

## Key Learnings

- **Inertia alone is not enough to choose k.** WCSS decreases monotonically as k increases — it
  needs to be paired with silhouette score (or another separation-aware metric) to avoid
  arbitrarily picking a large k that just overfits noise.
- **Cluster label numbers are arbitrary.** K-Means assigns integers (0, 1, 2, 3) with no inherent
  meaning — assigning persona names requires comparing feature means against real-world/
  ground-truth context, not just reading the table at face value. An earlier pass at this
  assignment was checked against actual numbers and two of the four persona labels turned out to
  be swapped — a clean reminder to verify with numbers, not eyeball a table.
- **PCA before clustering isn't automatically better.** It helped modestly here (0.526 → 0.560
  silhouette) because the top components retained ~95%+ of the original variance — very little
  signal was discarded. PCA can also *hurt* clustering when the directions of maximum variance
  don't align with the directions that actually separate the true groups, since PCA has no
  knowledge of class structure.
- **Evaluating unsupervised learning without ground truth is fundamentally harder.** This project
  could verify cluster quality against known personas; a real production segmentation task usually
  can't, and has to rely on silhouette score, business sanity-checks, and stakeholder validation
  instead.

---

## Notebook Structure

```
Introduction
└── Problem statement, why unsupervised, why synthetic

Data
├── Persona design (4 personas, 6 features, np.random.normal generation)
└── Preprocessing — clipping, renormalizing ratios, StandardScaler

Choosing k
└── Elbow method + Silhouette score sweep, k=2 to 8

K-Means Clustering
├── Model A — raw scaled features
└── Model B — top-4 PCA components (final model)

PCA Visualisation
├── 2D projection for visual cluster-separation check
└── Cumulative explained variance curve

Cluster Profiling
└── Per-cluster feature means, persona name assignment verified against ground truth

Business Insights
└── Persona table with recommended product action per segment
```

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*
