"""
Pre-processing module.

Implements the pipeline described in the project report:
cleaning -> sentence tokenization -> word tokenization ->
stop-word removal -> lemmatization.
"""
import re
import string

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_data():
    """Download required NLTK data files only if they are missing."""
    for path, pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()
try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    _STOPWORDS = set()


def clean_text(text: str) -> str:
    """Remove noise (extra whitespace, control characters) while
    preserving sentence-ending punctuation, which sentence tokenization
    needs."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences."""
    text = clean_text(text)
    if not text:
        return []
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def normalize(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Word-level normalization used before similarity scoring:
    lowercase, strip punctuation, optional stop-word removal, optional
    lemmatization. Returns a re-joined string (what the vectorizers expect)."""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation and re.search(r"[a-z0-9]", t)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
