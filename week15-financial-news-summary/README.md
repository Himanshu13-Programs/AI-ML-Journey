# Week 15: Financial News Summarizer (Hallucination Test)

## Problem

Can an LLM summarize a financial news article without inventing anything? In finance a wrong number or a made-up reason is worse than no summary, so this project does not only generate summaries, it tries to catch the model making things up.

For each article the model (`openai/gpt-oss-120b` on Groq) returns JSON with:

| Field | Meaning |
|---|---|
| `headline_summary` | one sentence |
| `key_points` | exactly 3 bullets |
| `numbers_mentioned` | significant figures in the article |
| `sentiment` | positive / neutral / negative |
| `answer` | reply to a test question about the article |

## Data

20 short **synthetic** financial news paragraphs (about 80 to 100 words each), written for this evaluation. They are not real news and are not copied from any news site or exchange filing. Each item has a question and an expected answer, and the questions are built to tempt the model into hallucinating:

| `q_kind` | Count | Expected behaviour |
|---|---|---|
| `answerable_control` | 5 | Answer correctly (shows the model is not just refusing everything) |
| `unanswerable` | 9 | Say "Not Stated in the article" |
| `false_premise` | 6 | Reject the premise and state what the article says |

## Approach

- **System prompt:** role, JSON schema, "valid JSON only, no markdown or code fences"
- **User message:** the article between `<article>` tags, with the id and question in `<id>` and `<query>` tags
- **Temperature 0.0:** this is extraction, not creative writing, so the output should not change between runs and randomness cannot alter a number
- **Long articles:** the longest article is about 100 words, so no chunking is needed (if needed: summarise each chunk, then summarise the summaries)
- **Parsing:** `json.loads` inside `try/except`; failures are logged

## Rubric

Each summary is scored by hand, 0 to 2 on five dimensions (max 10):

| Dimension | 2 | 0 |
|---|---|---|
| Factual accuracy | every claim matches the article | a claim contradicts it |
| Unsupported claims | nothing beyond the text | invented facts or reasons |
| Numbers | all exact | a wrong or invented figure |
| Completeness | main event and key figures covered | misses the main point |
| Abstention | says "not stated" or rejects the false premise correctly | invents an answer |

## Results

| Metric | Value |
|---|---|
| Avg rubric score (/10) | 9.9 |
| Avg score: answerable vs unanswerable vs false premise | 10.0 vs 10.0 vs 9.7 |
| JSON parse success | 100% (20 of 20) |
| Number-fidelity pass rate (number tokens) | 95% (19 of 20) |
| Avg latency | 2.50 s |
| Avg output tokens per summary | 553 |

All 20 articles were scored by hand by one person.

Note on latency: the free Groq tier limits tokens per minute, so some calls were slowed by rate-limit retries and the average is higher than the typical call (about 1.3 s).

## Key finding

The model rarely invented facts. It answered all 5 controls correctly and said "Not Stated" for all 9 unanswerable questions. Its weak spot was **false-premise questions where the article contradicts the question only by implication**:

- **Article 8 (Google):** asked what the judge ruled, the model said "Not Stated" instead of pointing out that only a lawsuit was announced and no ruling exists.
- **Article 12 (ASML):** asked by how much the forecast was *raised*, the model said "Not Stated" instead of saying the forecast was disappointing.

Likely cause: the `answer` rule was one clause inside the schema, and "Not Stated" is the easiest answer to give whenever the article is silent. A prompt rule asking the model to check the question against the article fixed the cases where the contradiction is stated directly (for example Tesla, Netflix, Adani, HDFC), but not the two cases that need inference. In article 8 "Not Stated" is also arguably a correct answer, so part of this gap may be my expected label being strict.

## Examples

**Good: article 15 (false premise, Netflix)**

- Question: "Why did Netflix's stock fall after it added 8.5 million subscribers?"
- Model answer: it did not fall, the stock rose 5% after the subscriber gain
- Numbers listed: `8.5 million` (matches the article)

**Failure: article 8 (false premise, Google)**

- Question: "What did the judge rule in the Google antitrust case, and what remedy was ordered?"
- Expected: reject the premise, since a lawsuit was announced and there is no ruling yet
- Model answer: `Not Stated in the article`

## Limitations

- The articles are short, clean and synthetic, so scores are likely higher than they would be on real announcements or long news articles.
- Only 20 items, all scored by one person with no second reader.
- The prompt does not say "copy numbers exactly", and the model returns `numbers_mentioned` as free text rather than a list.
- The prompt fix was tested on only a few articles, so it shows the change can help, not that it helps in general.

## Next steps

- Test on real NSE/BSE announcements and longer news articles
- Return `numbers_mentioned` as a list and keep the number check as an automatic gate
- Add a `premise_ok` field so the model decides about the premise before it writes the answer
- Have a second person score the rubric and compare

## How to run

1. Open `week15_financial_news_summarizer.ipynb` in Google Colab
2. Get a free key from the Groq console and add it as a Colab secret named `GROQ_API_KEY` (key icon in the left sidebar). **Do not paste the key into the notebook.**
3. Run all cells. The notebook installs `groq` and calls `openai/gpt-oss-120b`.

On the free tier the model allows about 8,000 tokens per minute, so the loop may slow down or hit a rate-limit (429) error. Wait a minute and re-run the failed articles.

## Files

| File | Contents |
|---|---|
| `week15_financial_news_summarizer.ipynb` | the full notebook |
| `summaries.json` | the 20 model summaries |
