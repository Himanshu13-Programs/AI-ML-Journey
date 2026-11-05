# Clef Support Chatbot

### Week 16 — Retrieval-Augmented Generation with Qdrant on Real Company Policy Documents

**Domain:** Customer / Employee Support **Stack:** `MarkdownHeaderTextSplitter`, `all-MiniLM-L6-v2`, Qdrant (in-memory), Groq `openai/gpt-oss-120b` **Dataset:** Clef's public company-policy markdown files — [Clef policy repo](ADD-LINK-HERE)

> **Disclaimer:** This project is a methodology exercise in building and honestly evaluating a RAG pipeline — it is not a production support bot. It runs on a small subset of Clef's policy documents (11 chunks), evaluated on a 20-question hand-written set, so the metrics below are directional rather than conclusive. The goal is to demonstrate a correct workflow (structure-aware chunking, normalised embeddings, file-level retrieval metrics, hand-scored generation, abstention testing) and to be upfront about what the numbers can and cannot show.

---

## Problem Statement

Company policies (compensation, remote work, leave, healthcare, conduct) are spread across many documents, and employees usually want a specific fact — "how many options does a technical hire with under 5 years' experience get?" — not a whole document. A plain LLM will either guess or answer from general knowledge, which is dangerous for policy questions where the exact number or rule matters.

This project builds a **Clef Support AI** that retrieves the relevant policy chunks from a Qdrant vector store and answers *only* from them, citing the source file. It is evaluated on three questions:

1. **Retrieval:** does it find the right policy file?
2. **Generation:** are the answers faithful, relevant and correct?
3. **Abstention:** does it refuse questions the documents cannot answer, instead of inventing an answer?

**Scope note:** the original Week 16 task asked for a FAISS-vs-Qdrant comparison, metadata filtering, a Streamlit app and faithfulness scoring on RBI/SEBI documents. I deliberately used a real, smaller corpus (Clef) so the pipeline can grow into a complete end-to-end support bot later. Because the corpus is new there is no FAISS baseline, so the FAISS comparison, metadata filtering and Streamlit app are skipped. The same-eval-set discipline is kept: every metric is computed on one fixed eval set.

---

## Dataset

- **Source:** Clef's public company-policy documents (markdown), a subset of the files in their GitHub repo
- **Topics covered:** compensation and equity, remote work, healthcare and disability insurance, vacation and sick leave, holidays, community code of conduct, equal opportunity, plus a few files used only for abstention tests (complaint policy, drug and alcohol policy, employee privacy)
- **Size after chunking:** 11 chunks, each embedded as a 384-dimensional vector
- **Eval set:** 20 questions across 12 categories (factual retrieval, policy & process, benefits, holidays, equity mechanics, multi-document synthesis, and more). Each has a `ground_truth` answer, the expected `source_file`, and the exact `context_snippet` it should come from
- **Abstention set:** 5 questions that sound policy-related but ask for details the documents do not contain (a BAC limit, an office room number, encrypted flash drives, and so on)

---

## Architecture

```
 Clef policy .md files
          │
          ▼
 MarkdownHeaderTextSplitter  (#, ##, ### — headers kept in chunk)
          │   chunk text + source / file_name / title metadata
          ▼
 all-MiniLM-L6-v2  (384-dim, L2-normalised)
          │
          ▼
 Qdrant collection "clef_documents"  (cosine, in-memory)
          ▲
          │  top-k chunks (k=5)
 User query ──► embed query (same model, normalised) ──► retrieve()
                                                              │
                                                              ▼
                                  Prompt: <query> + <context> + system rules
                                                              │
                                                              ▼
                                       Groq  openai/gpt-oss-120b
                                                              │
                                                              ▼
                              Answer + source file, or fixed refusal phrase
```

---

## Design Choices

### Header-Based Chunking

```python
MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
    strip_headers=False
)
```

Policy documents are already organised by headings, so splitting on `#`, `##` and `###` keeps each chunk about one topic instead of cutting mid-paragraph at a fixed character count. `strip_headers=False` keeps the heading text inside the chunk, since a heading like "Less than 5 Years Experience" is useful context for the embedding model.

### Normalised Embeddings

`all-MiniLM-L6-v2` with `normalize_embeddings=True`, for both chunks and queries. With normalised vectors, cosine similarity equals a dot product, and a mismatch between how chunks and queries are normalised is one of the most common silent causes of bad retrieval.

### Qdrant Collection

```python
client = QdrantClient(":memory:")
client.create_collection(
    "clef_documents",
    vectors_config=VectorParams(size=emb.shape[1], distance=Distance.COSINE)
)
```

Each chunk is upserted as one point with its text and file metadata (`source`, `file_name`, `title`) stored as payload, so a retrieved chunk can always be traced back to its file. The vector size is read from `emb.shape[1]` so the collection always matches the embedding model. The collection is rebuilt each run, which is fine at this size but would need a persistent or cloud Qdrant instance for a real deployment.

### Generation

`generate(query, k=5)` retrieves the top 5 chunks, wraps the query in `<query>` tags and the chunks in `<context>` tags, and sends them to `openai/gpt-oss-120b` on Groq (`temperature=0.9`, `top_p=0.9`, `max_tokens=1000`). The system prompt makes the bot:

- introduce itself as **Clef Support AI** and greet back when greeted
- answer only from the provided context and cite the source and file name
- reply with the fixed phrase *"Please ask questions that are in line with company policy"* when the context does not contain the answer

The fixed refusal phrase makes abstention easy to detect during evaluation.

---

## Evaluation Method

**Retrieval (automatic, file-level):** a retrieval is a hit if any of the top-5 chunks comes from the expected source file.

- **Hit-Rate@5** — fraction of questions where the correct file appears in the top 5
- **Precision@5** — fraction of the top 5 chunks that come from the correct file
- **MRR@5** — mean of 1 / rank of the first correct chunk

**Generation (hand-scored, 0–2 each):**

- **Faithfulness** — every claim is supported by the retrieved context
- **Relevance** — the answer addresses the question asked
- **Correctness** — facts and numbers match the ground truth

**Abstention:** the 5 unanswerable questions are run through the full pipeline; success means the fixed refusal phrase and no invented answer.

---

## Results

| Metric | Value | Notes |
|---|---|---|
| Hit-Rate@5 | 0.95 | File-level, 20 questions |
| Precision@5 | 0.31 | Capped by few chunks per file (see below) |
| MRR@5 | 0.87 | Correct file usually ranked first |
| Avg faithfulness | 1.9 / 2 | Hand-scored |
| Answer relevance | 1.9 / 2 | Hand-scored |
| Correctness | 1.85 / 2 | Hand-scored |
| Correct abstention | 5 / 5 | Fixed refusal phrase on all unanswerable questions |
| Avg end-to-end latency | X s | Pending — see Honest Interpretation |

---

## Honest Interpretation

**Precision@5 looks low but is structurally capped.** With only 11 chunks spread across roughly ten files, most files have just one or two chunks. Even a perfect retriever cannot fill five slots with chunks from a single one-chunk file, so the ceiling for most questions is 0.2–0.4. A file with several chunks (one single-question check) reached 0.6. Hit-Rate and MRR are the more meaningful retrieval numbers here.

**The one retrieval miss may be an eval artifact.** The multi-document synthesis question stores its expected source as a single string (`"Vacation and Sick Leave.md, Working Remotely.md"`), so the check `source_file in files` can never match it. The 0.95 Hit-Rate may therefore understate retrieval quality; the fix is to store expected sources as a list.

**Retrieval metrics are file-level, not passage-level.** A hit means the right *file* was retrieved, not that the right *passage* was. The ground-truth `context_snippet` field exists in the eval set but is not yet used for scoring.

**Generation scores are hand-assigned on 20 questions.** They are a small-sample judgement, not a benchmark. No automatic number-fidelity check was run.

**The latency figure is not reported.** The original timing loop appended the elapsed time *before* calling `generate()`, so it measured only the `print` statements (about 120 microseconds) rather than retrieval plus LLM generation. The loop needs to be fixed and re-run before a latency number is quoted. It is left as `X s` in the table rather than reporting a wrong value.

**No conversation memory.** The system prompt mentions chat history, but `generate()` rebuilds the message list on every call, so follow-up questions are not actually supported yet.

**The metrics are generous because the corpus is tiny.** With 11 chunks, retrieving the right file is easier than it will be over the full Clef policy set. These numbers are a baseline to re-run, not a final verdict.

---

## Notebook Structure

```
Introduction
└── Goal, pipeline overview, scope (what was skipped and why)

Data
├── Load Clef policy markdown files
└── Header-based chunking (11 chunks)

Embeddings
└── all-MiniLM-L6-v2, normalised, 384-dim

Qdrant Setup & Indexing
└── In-memory collection, cosine distance, payload metadata

LLM Setup
└── Groq client, key from Colab Secrets

Retrieval & Generation
├── retrieve(query, k=5)
└── generate(query, k=5) with citation + refusal rules

Evaluation Set
└── 20 questions, 12 categories, ground truth + expected source

Retrieval Evaluation
└── Hit-Rate@5, Precision@5, MRR@5

Abstention Test
└── 5 unanswerable questions

Generation Evaluation
└── Hand-scored faithfulness, relevance, correctness

Results Summary & Insights
```

---

## Example Q&A

**Q:** What is the standard salary and equity offer for a technical candidate with less than 5 years of experience at Clef?

**A:** $100,000 per year and 41,963 stock options (about 0.9% of outstanding shares), with an option to reduce salary by $5,000 in exchange for an additional 4,663 options (about 0.1% more equity). *Source: Salary and Equity Compensation.md*

**Q (unanswerable):** What specific blood alcohol concentration (BAC) limit does Clef enforce for employees consuming beer or wine at office milestone celebrations?

**A:** *Please ask questions that are in line with company policy.* (The Drug and Alcohol Policy specifies no numerical BAC threshold, so the bot correctly refuses.)

---

## How to Run

1. Open `week16_rag_qdrant.ipynb` in Google Colab
2. Add the Clef policy `.md` files to `/content/documents/` (or update the path in the Data section)
3. Add your Groq API key to Colab Secrets as `GROQ_API_KEY` — **never paste it into the notebook or commit it**
4. Run all cells in order (`Runtime → Run all`)

**Requirements:**

```
langchain_text_splitters
langchain_core
qdrant-client
sentence_transformers
groq
```

Install with:

```
pip install langchain_text_splitters langchain_core qdrant-client sentence_transformers groq
```

---

## What I Would Do Next

- **Fix the latency measurement** — time the full `generate()` call (retrieval plus LLM), then report the real average
- **Fix the multi-source eval entry** — store `source_file` as a list so the multi-document question can be scored correctly
- **Score at the passage level** — use the `context_snippet` field to check that the right *passage*, not just the right file, was retrieved
- **Lower the generation temperature** — `0.9` is high for a factual bot; around `0.2` should reduce variation in cited numbers
- **Add an automatic number-fidelity check** — every number in an answer must appear in the retrieved text
- **Add conversation memory** — keep the message history across turns so follow-up questions work
- **Scale to all Clef policy files** — re-run the same eval set on the full corpus and add a persistent Qdrant instance
- **Add metadata filtering and hybrid search** — filter by document type and combine keyword with semantic search for exact-term queries
- **Compare against FAISS** — once the full corpus is indexed, run the original FAISS-vs-Qdrant comparison on the same eval set

---

*Part of the [AI-ML-Journey](https://github.com/Himanshu13-Programs/AI-ML-Journey) repository.*
