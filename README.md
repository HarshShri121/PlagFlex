# PlagFlex

An intelligent document similarity framework for plagiarism analysis and
interactive visualization, rebuilt to match the project report's design
(pre-processing → lexical + semantic feature extraction → score fusion →
tiered risk classification → visualization/report), with an added web
plagiarism check.

## What's in !!

- **Modular code**: `preprocessing.py`, `similarity.py`, `file_readers.py`,
  `web_check.py`, `app.py` (was a single `app.py`).
- **Proper pre-processing**: cleaning, sentence tokenization, stop-word
  removal, lemmatization (all toggleable).
- **Lexical + semantic score fusion**: TF-IDF/cosine similarity, combined
  with sentence-embedding similarity when the optional
  `sentence-transformers` package is installed. Falls back gracefully to
  lexical-only scoring if it isn't.
- **Tiered risk levels** (Low / Medium / High) with sidebar-adjustable
  thresholds, instead of a single raw similarity number.
- **Web plagiarism check, rebuilt**: uses Google's official Programmable
  Search Engine (Custom Search JSON) API when you provide an API key + a
  Search Engine ID, with an unofficial scrape fallback if you don't. Search
  results are matched against your sentences with the same fused
  lexical+semantic scoring, and results are rate-limited/capped to avoid
  hammering the search backend.
- **Focused dashboard**: an overall-score gauge, a risk-distribution chart,
  inline highlighted text, and a sentence-level results table - instead of
  eight overlapping chart types.
- **Downloadable CSV report**.

## Features

- Paste text or upload `.txt` / `.docx` / `.pdf`.
- Compare a document against an uploaded reference document, the web, or
  both.
- Compare multiple documents against each other (pairwise similarity +
  heatmap for 3+ files).
- Adjustable pre-processing and risk-threshold settings in the sidebar.

## Tech Stack

**Client:** Streamlit
**Server:** Python
**Libraries:** pandas, NLTK, BeautifulSoup, requests, scikit-learn
(TF-IDF + cosine similarity), optional `sentence-transformers` (semantic
similarity), docx2txt, PyPDF2, Plotly

## Run locally

```bash
pip install -r requirements.txt
# optional, for semantic similarity:
pip install sentence-transformers
streamlit run app.py
```

## Setting up the web check

1. Create a Programmable Search Engine at
   https://programmablesearchengine.google.com/ and set it to search the
   entire web.
2. Enable the "Custom Search API" and generate an API key at
   https://console.cloud.google.com/.
3. Paste the API key and Search Engine ID (`cx`) into the app's sidebar
   under **Web check settings**.

Without an API key/cx, the app falls back to scraping Google search
results directly. That's unofficial, can be rate-limited or blocked at
any time, and is only meant for occasional local testing - not something
to depend on for real coursework submissions.

## Authors

- Harshwardhan S. Shrivastav
- Jay A. Paturkar
- Kunal P. Jambutkar
- Parth J. Satija

Based on the original prototype by [@KarthikS](https://www.github.com/Karthik-02).
