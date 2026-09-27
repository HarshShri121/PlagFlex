"""
Similarity analysis module.

Combines lexical similarity (TF-IDF + cosine) with semantic similarity
(sentence embeddings, when the optional `sentence-transformers` package
is installed) into a single weighted "fused" score, and maps that score
to an interpretable risk tier - matching the score-fusion / decision
model described in the project report.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import normalize

_semantic_model = None
_semantic_unavailable = False


def semantic_model_available() -> bool:
    """Lazily load the sentence-embedding model. Returns False (once,
    cheaply) if the optional dependency isn't installed, so the rest of
    the app can fall back to lexical-only scoring."""
    global _semantic_model, _semantic_unavailable
    if _semantic_unavailable:
        return False
    if _semantic_model is not None:
        return True
    try:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        return True
    except Exception:
        _semantic_unavailable = True
        return False


def lexical_similarity(text_a: str, text_b: str, remove_stopwords: bool = True,
                        lemmatize: bool = True) -> float:
    """TF-IDF cosine similarity between two texts, after normalization."""
    a = normalize(text_a, remove_stopwords, lemmatize)
    b = normalize(text_b, remove_stopwords, lemmatize)
    if not a.strip() or not b.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform([a, b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def semantic_similarity(text_a: str, text_b: str) -> float | None:
    """Sentence-embedding cosine similarity. Returns None when the
    optional model isn't available."""
    if not semantic_model_available():
        return None
    embeddings = _semantic_model.encode([text_a, text_b])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def fused_score(text_a: str, text_b: str, lexical_weight: float = 0.4,
                 semantic_weight: float = 0.6, remove_stopwords: bool = True,
                 lemmatize: bool = True) -> dict:
    """Weighted fusion of lexical + semantic similarity. If the semantic
    model is unavailable, the score falls back to lexical-only (weight 1.0)
    so the app keeps working without the optional dependency."""
    lex = lexical_similarity(text_a, text_b, remove_stopwords, lemmatize)
    sem = semantic_similarity(text_a, text_b)
    if sem is None:
        fused = lex
    else:
        total = lexical_weight + semantic_weight
        fused = (lexical_weight * lex + semantic_weight * sem) / total
    return {"fused": fused, "lexical": lex, "semantic": sem}


def risk_level(score: float, low_threshold: float = 0.30, high_threshold: float = 0.60) -> str:
    """Map a 0-1 similarity score to a tiered risk label."""
    if score >= high_threshold:
        return "High"
    if score >= low_threshold:
        return "Medium"
    return "Low"


RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
