# Classic Search Engine

## Overview

This project implements a classic search engine / retrieval system tailored for **medical entity normalization**: given a short textual entity (e.g., "mri pelvis"), retrieve the best-matching coding-system target (e.g., SNOMED CT code and description). The system demonstrates:

- Preprocessing (tokenization, normalization, stopword removal, optional stemming/lemmatization)
- Indexing (inverted index, TF-IDF matrix, optionally BM25)
- Retrieval (exact-match, lexical ranking using TF-IDF / BM25, optional semantic re-ranking using embeddings)
- Evaluation (Top‑K accuracy / precision@K, MRR)

---

## Solution architecture (high level)

```
+-----------+     +-------------+     +-------------+     +--------------+
| User/Batch|---->| Preprocess   |---->| Indexer      |---->| Retrieval    |
| Queries   |     | (tokenize,   |     | (inverted     |     | (BM25/TF-IDF |
|           |     | normalize)   |     | index + TFIDF)|     |  + optional   |
|           |     |              |     |              |     |  embedding)  |
+-----------+     +-------------+     +-------------+     +--------------+
                                                      |
                                                      v
                                            +-----------------------+
                                            | Reranker / Formatter  |
                                            | (fuzzy matching, etc) |
                                            +-----------------------+
                                                      |
                                                      v
                                            +-----------------------+
                                            | Output: Top-K codes   |
                                            +-----------------------+
                                                      |
                                                      v
                                            +-----------------------+
                                            | Evaluation module     |
                                            | (precision@K, MRR)    |
                                            +-----------------------+
```

### Core components

- **Preprocessing**: tokenization, lowercasing, punctuation removal, optional lemmatization/stemming, stopword removal. Handles short clinical phrases and common abbreviations.
- **Indexer**: creates an inverted index and builds a TF-IDF matrix for all target descriptions (or a BM25 index using `rank_bm25`). Stores mappings between `target_code`, `description`, and preprocessed tokens.
- **Retriever**: given a query, runs the same preprocessing, then retrieves candidate targets via inverted index + scoring (BM25/TF-IDF cosine similarity). Optionally uses sentence embeddings (e.g., `sentence-transformers`) + faiss for semantic search.
- **Reranker**: optional stage that re-scores candidate list using fuzzy string matching (Levenshtein, token-set ratios), field-weighting (e.g., give more weight to exact code matches or phrase matches), or model-based re-ranking.
- **Evaluation**: computes Top-1/Top-5/Top-10 accuracy, Mean Reciprocal Rank (MRR), and optionally precision@K/AP/MAP when multiple gold targets exist.

---

## Dataset structure (from `Test_predictions_upgraded.xlsx`)

The uploaded Excel file contains 400 rows with these columns (exact names):

- `Input Entity Description` — the query / short phrase (e.g., `mri pelvis`)
- `Entity Type` — category such as `Procedure`
- `Output Coding System` — e.g., `SNOMEDCT_US`
- `Output Target Code` — numeric/alpha code in the target system (e.g., `430121004`)
- `Output Target Description` — canonical description for the target code
- `Output Target CUI` — UMLS CUI (where available)

This dataset appears to be the **predictions / results** mapping input entity strings to target codes (so it can be used both to build an index of candidates and to evaluate retrieval performance if the sheet contains ground-truth targets).

---

## Key features

- Lightweight, reproducible IR pipeline suitable for short clinical phrases.
- Lexical ranking via TF-IDF cosine similarity or BM25 scoring.
- Optional semantic re-ranking via sentence embeddings (improves recall for paraphrases and abbreviations).
- Fuzzy / token-based reranking to catch near-miss matches.
- Simple evaluation scripts to calculate Top-K accuracy and MRR.
- Ready-to-run Jupyter notebook (placeholder) + CLI-friendly Python modules.

---

## Recommended repository layout

```
classic-search-engine/
├── data/
│   ├── Test_predictions_upgraded.xlsx      # uploaded mapping / test file
│   └── raw/                                # optional: raw source files
├── notebooks/
│   └── Classic_search_engine.ipynb         # main notebook (copy your working notebook here)
├── src/
│   ├── preprocess.py                       # tokenization, cleaning utils
│   ├── indexer.py                          # building inverted index / TF-IDF / BM25
│   ├── search.py                           # query-time retrieval + reranking
│   ├── evaluate.py                         # evaluation metrics and reports
│   └── utils.py                            # helpers
├── requirements.txt
├── README.md                               # this file
└── LICENSE
```

---

## Setup instructions

1. **Python version**

   Use Python 3.8+ (3.9/3.10 recommended).

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**

Create a `requirements.txt` with the following (adapt as needed):

```
pandas>=1.3
numpy
scikit-learn
nltk
rank_bm25
sentence-transformers
faiss-cpu ; platform_system != 'Windows'
fuzzywuzzy[speedup]
python-Levenshtein
openpyxl
tqdm
```

Then run:

```bash
pip install -r requirements.txt
```

> **Notes:**
> - On Windows, `faiss-cpu` may require conda; if you don't need semantic search, omit it.
> - For basic lexical retrieval, `rank_bm25` (pure-Python) + `scikit-learn` suffice.

4. **NLTK data (if using NLTK tokenizer/stopwords)**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

You can run those lines in a Python shell or the notebook the first time.

---

## Usage examples

### 1) Run the notebook

Open `notebooks/Classic_search_engine.ipynb` in Jupyter and run the cells from top to bottom. The notebook should demonstrate:
- Loading `Test_predictions_upgraded.xlsx`
- Preprocessing targets and queries
- Building an index (TF-IDF and/or BM25)
- Running a few sample queries and printing top-K results
- Running the evaluation on the predictions file

### 2) CLI-style scripts

**Indexing** (example):

```bash
python src/indexer.py --input data/Test_predictions_upgraded.xlsx --index data/index.pkl
```

**Search** (example):

```bash
python src/search.py --index data/index.pkl --query "mri pelvis" --topk 5
```

**Evaluate** (example):

```bash
python src/evaluate.py --predictions data/Test_predictions_upgraded.xlsx --index data/index.pkl --topk 5
```

Expected `search.py` output (example):

```
Query: "mri pelvis"
Top-5 results:
1) 430121004 — Magnetic resonance imaging (MRI) of pelvis for radiotherapy planning — score: 0.73
2) 123456789 — MRI pelvis (generic) — score: 0.48
3) 987654321 — Pelvic MRI without contrast — score: 0.31
...
```

The evaluate script should print overall Top-1/Top-5 accuracy and MRR and optionally save a CSV with per-query results.

---

## Implementation notes / tips

- **Index choice**: For short clinical phrases, BM25 often outperforms raw TF-IDF for ranking. Use `rank_bm25` for a compact implementation. For scale, consider `Whoosh`, `Elasticsearch`, or `OpenSearch`.

- **Handling abbreviations**: medical abbreviations are common. Consider an abbreviation-expansion dictionary during preprocessing.

- **Semantic retrieval**: adding `sentence-transformers` and `faiss` can capture semantic matches (paraphrases), which helps recall for queries not lexically similar to descriptions.

- **Reranking**: after lexical/semantic retrieval, use fuzzy matching (token_set_ratio / partial_ratio) to promote near-exact matches.

- **Evaluation**: If the gold standard contains one canonical mapping per query, compute Top-K accuracy and MRR. If multiple gold targets exist, compute precision@k and MAP.

---

## Example code snippets

**Preprocessing (simplified)**

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

STOP = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in word_tokenize(text) if t not in STOP]
    return tokens
```

**BM25 retrieval (simplified)**

```python
from rank_bm25 import BM25Okapi

corpus = [preprocess(d) for d in descriptions]
bm25 = BM25Okapi(corpus)

def retrieve(query, topk=5):
    q = preprocess(query)
    scores = bm25.get_scores(q)
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [(idx, scores[idx]) for idx in top_idxs]
```

---

## Evaluation (recommended metrics)

- **Top-1 accuracy** (exact match of the top-ranked target)
- **Top-5 / Top-10 accuracy**
- **Mean Reciprocal Rank (MRR)**
- **Precision@K** (if multiple relevant targets)

---

## Troubleshooting

- If you get strange tokenization results: ensure `nltk` punkt models are downloaded and stopwords are available.
- If `faiss` installation fails on Windows: either install with conda or omit semantic search.
- If search returns poor results: experiment with stemming vs lemmatization, tune BM25 parameters (`k1`, `b`) or add a reranker.

---

## Next steps & extensions

- Add an abbreviation-expansion step specific to the clinical subdomain.
- Integrate a lightweight vector DB (FAISS/Annoy) for fast semantic search over embeddings.
- Add an API (Flask/FastAPI) to expose search as a service.
- Improve evaluation by adding proper ground-truth mapping and cross-validation splits.

---

## License & authors

```
MIT License
(c) <Your Name / Organization>
```

---

## Contact

If you want me to tailor this README to exactly match your code, upload the complete `Classic_search_engine.ipynb` (non-empty) and any supporting `src/` code. I examined the uploaded `Test_predictions_upgraded.xlsx` and used its columns to infer dataset structure and suggestions above.

---

*End of README*
