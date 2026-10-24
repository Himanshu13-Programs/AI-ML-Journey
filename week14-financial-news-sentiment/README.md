# Financial News Sentiment: TF-IDF + Logistic Regression vs FinBERT

### Is a Pretrained Financial Transformer Worth Its Cost, and Does It Work on Indian Headlines?

**Domain:** Finance / NLP (text classification) **Algorithms:** TF-IDF + Logistic Regression (scikit-learn), FinBERT zero-shot and FinBERT fine-tuned (HuggingFace Transformers, PyTorch) **Data:** Financial PhraseBank (4,846 labelled financial news sentences) plus 10 hand-labelled Indian business headlines

> **Disclaimer:** This project compares sentiment models on a small public dataset. It is not a trading system and not financial advice. The Indian headlines test has only 10 examples, so its scores show a direction, not a precise measurement.

* * *

## Problem Statement

Given a sentence from financial news, can a model tell whether it is **negative, neutral or positive**?

Three models are compared on the same test data:

1. **TF-IDF + Logistic Regression**: counts which words and word pairs appear in a sentence and learns a weight for each one. Fast, small, needs labelled data. This is the **baseline**.
2. **FinBERT (zero-shot)**: `ProsusAI/finbert`, a BERT model already trained on financial text, used through the HuggingFace `pipeline` with **no training** here.
3. **FinBERT (fine-tuned)**: the same model trained further on this project's training split (a bonus beyond the task).

A transformer that does not beat a TF-IDF baseline is not worth its cost, so the baseline is built first and everything is judged against it.

The project asks two questions: **does the transformer beat the baseline, and by how much?** and **does it still work on Indian headlines, which look different from the training data?**

* * *

## Dataset

- **Main data:** Financial PhraseBank (Malo et al., 2014), loaded from `data.csv`. 4,846 English sentences from financial news about listed companies, each labelled by annotators as negative, neutral or positive (the version where at least 50% of annotators agreed). Source: <add the link you downloaded it from>
- **License:** CC BY-NC-SA 3.0 (non-commercial, share-alike). Fine for a learning portfolio, not for a commercial product. Confirm on the page you downloaded it from
- **Indian headlines:** 10 recent Indian business headlines that I labelled by hand (4 positive, 3 negative, 3 neutral) before running any model. They are written in the notebook, so there is no separate file and no copyright issue
- **File format:** no header row, `latin-1` encoding. The columns are named `sentiment` and `text`

| Label        | Sentences | Share |
|--------------|-----------|-------|
| **neutral**  | 2,879     | 59.4% |
| **positive** | 1,363     | 28.1% |
| **negative** | 604       | 12.5% |

**Caveat:** the sentences are mostly about Finnish and European companies. Indian market language (crore, repo rate, Q2 results, IPO subscription) is rare or missing, which is why the Indian headlines test matters.

**Labels:** negative = 0, neutral = 1, positive = 2. FinBERT returns label *names*, so its output is mapped with the same dictionary to keep every model in one label space.

* * *

## Key EDA Findings

**The classes are imbalanced.** Neutral is about 59% of the data and negative only about 12%. A model that always says "neutral" would already be right about 59% of the time without learning anything, so plain accuracy is a weak yardstick. This project uses **macro F1** (the average of the three class F1 scores) and `class_weight="balanced"` in the baseline.

**Stratified split.** An 80/20 split with `stratify=y` and `random_state=42` gives 3,876 training and 970 test sentences with identical class shares (negative 12.5%, neutral 59.4%, positive 28.1% in both). The test set holds 121 negative, 576 neutral and 273 positive sentences.

**The baseline learned sensible words, with a few oddities.** From the Logistic Regression weights:

| Rank | Top positive words (weight)                                              | Top negative words (weight)                                          |
|------|--------------------------------------------------------------------------|----------------------------------------------------------------------|
| 1    | increase (2.88)                                                          | down (4.07)                                                          |
| 2    | increased (2.63)                                                         | decreased (3.65)                                                     |
| 3    | up (2.48)                                                                | fell (2.85)                                                          |
| 4    | rose (2.42)                                                              | lower (2.00)                                                         |
| 5    | up from (2.20)                                                           | staff (1.99)                                                         |
| 6    | grew (2.01)                                                              | down from (1.85)                                                     |
| 7    | signed (1.88)                                                            | result (1.82)                                                        |
| 8    | rose to (1.84)                                                           | off (1.69)                                                           |
| 9    | our (1.75)                                                               | lay (1.67)                                                           |
| 10   | leading (1.73)                                                           | dropped (1.63)                                                       |

Most words match intuition (numbers rising or falling, new deals). *staff*, *lay* and *off* are pieces of "lay off staff" (job cuts), which really is bad news, and *our* has no sentiment on its own, so it probably comes from company statements such as "our profit grew".

**Why `stop_words=None`.** scikit-learn's default English stop-word list contains "not" and "no" (negations flip sentiment), and also "up", "down", "off" and "our". Four of the strongest features above would have been deleted by stop-word removal.

* * *

## Preprocessing & Models

### The Baseline Pipeline

```
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=None)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
baseline.fit(x_train, y_train)
```

- **`ngram_range=(1, 2)`**: single words and word pairs ("net profit", "up from")
- **`min_df=2`**: drops words and pairs seen in only one sentence
- **`class_weight="balanced"`**: mistakes on the rare negative class cost more

### FinBERT (Zero-Shot)

```
bert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = bert(x_test.to_list(), batch_size=16, truncation=True)
```

FinBERT is BERT-base (about 110 million parameters) further trained on financial text and fine-tuned for financial sentiment on Financial PhraseBank. **Important:** that is the same dataset the test split comes from, so FinBERT has probably seen many of the test sentences and its test score is likely **optimistic**. The Indian headlines are the fairer test.

### FinBERT (Fine-Tuned)

| Setting          | Value                                                              |
|------------------|--------------------------------------------------------------------|
| Base model       | `ProsusAI/finbert` with a 3-class head (`num_labels=3`)            |
| Training data    | 3,876 training sentences                                           |
| Epochs           | 3 (Trainer default)                                                |
| Learning rate    | 5e-5 (Trainer default)                                             |
| Batch size       | 8 (Trainer default)                                                |
| Padding          | `max_length` (every sentence padded to 512 tokens)                 |
| Evaluation while training | None, so no validation curve and no early stopping        |
| Training time    | 1,455 steps in about 1,064 s (about 18 minutes), average loss 0.277 |

Two known weaknesses of this setup: FinBERT's own head uses a different label order from mine (positive / negative / neutral against my negative / neutral / positive), and I attached `id2label` only after training. Padding every sentence to 512 tokens also wastes compute.

* * *

## How the Models Are Evaluated

| Metric               | What it tells you                                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Precision**        | When the model says a class, how often it is right                                                                                  |
| **Recall**           | Of all real sentences in a class, how many the model finds                                                                          |
| **F1**               | The balance of precision and recall for one class                                                                                   |
| **Macro F1**         | The plain average of the three class F1 scores, so the small negative class counts as much as the big neutral class. Main metric   |
| **Confusion matrix** | Rows are real labels and columns are predictions. Shows *which* classes get mixed up                                                |
| **Inference time**   | Wall-clock time to predict all 970 test sentences. Depends on hardware, so only compare models run on the same machine              |

Every model is scored on **two test sets**: the 970-sentence hold-out and the 10 Indian headlines.

* * *

## Results

### Test Set (970 sentences)

| Model                 | Macro F1  | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
|-----------------------|-----------|----------|-------------|------------|-------------|
| TF-IDF + LR           | 0.709     | 0.75     | 0.65        | 0.82       | 0.65        |
| **FinBERT zero-shot** | **0.863** | 0.87     | 0.85        | 0.89       | 0.84        |
| FinBERT fine-tuned    | 0.831     | 0.85     | 0.80        | 0.88       | 0.81        |

### Indian Headlines (10 hand-labelled)

| Model                 | Macro F1  | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
|-----------------------|-----------|----------|-------------|------------|-------------|
| TF-IDF + LR           | 0.529     | 0.50     | 0.80        | 0.50       | 0.29        |
| FinBERT zero-shot     | 0.785     | 0.80     | 0.67        | 0.80       | 0.89        |
| FinBERT fine-tuned    | **0.794** | 0.80     | 0.86        | 0.67       | 0.86        |

### Comparison Table

| Model                         | Test Macro F1 | Indian Headlines Macro F1 | Inference Time (test set) | Needs Training?    |
|-------------------------------|---------------|---------------------------|---------------------------|--------------------|
| TF-IDF + LR                   | 0.709         | 0.529                     | ___ s                     | Yes (seconds)      |
| FinBERT (HF pipeline)         | 0.863         | 0.785                     | ___ s                     | No                 |
| FinBERT (fine-tuned)          | 0.831         | 0.794                     | ___ s                     | Yes (about 18 min) |

Hardware used for the times: <CPU or GPU>.

### Three Disagreements Between the Baseline and FinBERT

| # | Sentence   | True    | TF-IDF + LR | FinBERT | Who was right, and why the other failed |
|---|------------|---------|-------------|---------|-----------------------------------------|
| 1 | <sentence> | <label> | <label>     | <label> | <explanation>                           |
| 2 | <sentence> | <label> | <label>     | <label> | <explanation>                           |
| 3 | <sentence> | <label> | <label>     | <label> | <explanation>                           |

* * *

## Key Finding

**A pretrained financial transformer beat the TF-IDF baseline by 15.5 macro F1 points on the test set and 25.7 points on Indian headlines, with zero training.** The baseline dropped 18.0 points when moving to Indian text, zero-shot FinBERT only 7.8. Fine-tuning on this small dataset did not help in this run.

* * *

## Honest Interpretation

**FinBERT clearly beats the baseline, but part of its test score is probably leakage.** FinBERT was fine-tuned on Financial PhraseBank before this project started, so its 0.863 on the hold-out is likely optimistic. The Indian headlines are the fairer comparison, and the gap there (0.785 against 0.529) is even larger, though it rests on only 10 examples.

**Why the baseline struggles on Indian headlines.** A TF-IDF model can only use words it saw in training. "Crore", "Q2" and "repo rate" were never linked to a label, and positive phrasing such as "swings to profit" differs from the training sentences. It found only 1 of 4 positive headlines.

**Fine-tuning did not beat zero-shot.** On the test set it scored 0.831 against 0.863 (3.3 points lower). On Indian headlines, 0.794 against 0.785 is a tie: the same accuracy, a difference of less than 1 F1 point. Possible reasons for the test-set result (hypotheses, not tested):

1. FinBERT already saw this data, so there was little left to learn
2. The label order mismatch made the starting point worse than the original head
3. Default settings, no validation and no early stopping
4. One run with one seed, so a gap of 3 points may be partly run-to-run variation

**The Indian test is small.** The fine-tuned model scored 0.698 on these headlines in an earlier run and 0.794 in a later one. That is a swing of exactly one headline (accuracy 0.70 against 0.80), which shows how unstable a 10-example test is. Labels are also a judgement: "crude oil prices surge 85%" is negative from a macro view, and "IT companies add 16,000 employees, but demand remains weak" is mixed.

**Baseline or transformer for production?** 100K headlines a day is only about 1.2 headlines per second. At the speed measured above, one day of headlines takes about ___ minutes. For a system of that size I would choose ___ because ___.

* * *

## Notebook Structure

```
Introduction
└── Question, three models, metric, two test sets

Data
├── Financial PhraseBank (4,846 sentences), license note
├── Class balance
├── Label encoding
└── Stratified 80/20 split

Classical Baseline
├── TF-IDF + Logistic Regression pipeline (stop_words=None)
├── Classification report, confusion matrix, macro F1
├── Top 10 positive and negative words
└── Indian headlines test

Transformer (HuggingFace)
├── FinBERT zero-shot pipeline (batch_size=16, truncation=True)
├── Classification report, confusion matrix, macro F1
└── Indian headlines test

Fine-Tuning FinBERT (bonus)
├── Tokenizer and dataset preparation
├── Trainer (3 epochs)
└── Test set and Indian headlines evaluation

Comparison
├── Inference time for all three models
├── Results table
└── Three disagreements

Insights
└── Verdict table, conclusion, limits
```

* * *

## How to Run

1. Open `week14_5_news_sentiment.ipynb` in Jupyter or Google Colab
2. Place `data.csv` (Financial PhraseBank) next to the notebook, or upload it to Colab and update the path in the data-loading cell (it currently reads `/content/data.csv`)
3. Run all cells in order (`Kernel → Restart & Run All` in Jupyter, or `Runtime → Run all` in Colab)

**Notes:**

- The first run downloads `ProsusAI/finbert` from the HuggingFace Hub, so an internet connection is needed
- A GPU is strongly recommended for the fine-tuning section (about 18 minutes in my run). The baseline and zero-shot sections run fine on CPU
- Inference times depend on the hardware, so small differences from the numbers above are expected

**Requirements:**

```
numpy
pandas
matplotlib
seaborn
scikit-learn
transformers
datasets
torch
accelerate
```

Install with:

```
pip install numpy pandas matplotlib seaborn scikit-learn transformers datasets torch accelerate
```

* * *

## Folder Contents

| File                                | What it is                                       |
|-------------------------------------|--------------------------------------------------|
| `week14_5_news_sentiment.ipynb`     | This project: TF-IDF vs FinBERT sentiment study  |
| `<second notebook>`                 | <one line>                                       |
| `<third notebook>`                  | <one line>                                       |
| `README.md`                         | This file                                        |

* * *

## What I Would Do Next

- **Test on a fairer dataset:** FinBERT has seen Financial PhraseBank, so evaluate on labelled news it has not seen, and grow the Indian set from 10 headlines to a few hundred with more than one annotator
- **Fix the fine-tuning setup:** pass `id2label` / `label2id` before training, evaluate after every epoch with early stopping, try a smaller learning rate such as 2e-5, and repeat with several seeds before comparing
- **Speed up training:** use `DataCollatorWithPadding` instead of padding every sentence to 512 tokens
- **Fine-tune on Indian text:** if enough Indian headlines are labelled, test whether in-domain training closes the remaining gap
- **Try other pretrained models:** compare against other finance-tuned or general sentiment models, and a distilled (smaller, faster) model for the production case
- **Measure cost properly:** report inference time on CPU and GPU, plus memory use, for the 100K-headlines-a-day case

* * *

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*
