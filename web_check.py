"""
Web plagiarism check module.

Looks a sentence up on the web and reports the best-matching source and
how similar that source's text is. Backends, in order:

0. Brave Search API when a key is supplied (works from cloud hosts).
1. Google Programmable Search Engine (Custom Search JSON API) when an
   API key + cx are supplied - the supported way to query Google.
2. Unofficial fallbacks when no credentials are given: Google HTML
   scrape, then DuckDuckGo HTML. These can be blocked or change layout
   at any time, so failures are reported back to the UI rather than
   silently treated as "no plagiarism found".
"""
from __future__ import annotations

import re
import time
from urllib.parse import parse_qs, quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

from similarity import fused_score, phrase_containment

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 10


def _search_via_cse(query: str, api_key: str, cx: str, num: int = 3) -> list[dict]:
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": min(num, 10)},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        try:
            detail = resp.json()["error"]["message"]
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(f"Google API error {resp.status_code}: {detail}")
    items = resp.json().get("items", [])
    return [
        {"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in items
    ]


def _search_via_brave(query: str, api_key: str, num: int = 3) -> list[dict]:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": min(num, 20)},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Brave API error {resp.status_code}: {resp.text[:200]}")
    items = resp.json().get("web", {}).get("results", [])
    return [
        {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("description", "")}
        for i in items
    ]


def _search_via_google_scrape(query: str, num: int = 3) -> list[dict]:
    url = f"https://www.google.com/search?q={quote_plus(query)}&num={num}&hl=en"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for block in soup.select("div.g")[:num]:
        link = block.find("a")
        title_el = block.find("h3")
        snippet_el = block.find("div", class_=re.compile("VwiC3b|IsZvec|MUxGbd"))
        if link and link.get("href") and title_el:
            results.append({
                "title": title_el.get_text(),
                "url": link["href"],
                "snippet": snippet_el.get_text() if snippet_el else "",
            })
    return results


def _search_via_ddg(query: str, num: int = 3) -> list[dict]:
    resp = requests.post(
        "https://html.duckduckgo.com/html/", data={"q": query},
        headers=_HEADERS, timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for block in soup.select("div.result"):
        link = block.select_one("a.result__a")
        snippet_el = block.select_one(".result__snippet")
        if not link:
            continue
        href = link.get("href", "")
        redirect = parse_qs(urlparse(href).query).get("uddg")
        if redirect:
            href = redirect[0]
        results.append({
            "title": link.get_text(),
            "url": href,
            "snippet": snippet_el.get_text() if snippet_el else "",
        })
        if len(results) >= num:
            break
    return results


def search(query: str, api_key: str | None, cx: str | None, num: int = 3,
           brave_key: str | None = None):
    """Returns (results, backend_name, error_message)."""
    words = query.split()
    query = " ".join(words[:30])  # search engines ignore very long queries
    if brave_key:
        try:
            return _search_via_brave(query, brave_key, num), "Brave API", None
        except Exception as exc:
            return [], "Brave API", str(exc)
    if api_key and cx:
        try:
            return _search_via_cse(query, api_key, cx, num), "Google API", None
        except Exception as exc:
            return [], "Google API", str(exc)

    last_error = None
    for name, fn in (("Google (scrape)", _search_via_google_scrape),
                     ("DuckDuckGo (scrape)", _search_via_ddg)):
        try:
            results = fn(query, num)
            if results:
                return results, name, None
        except Exception as exc:
            last_error = f"{name}: {exc}"
    return [], "scrape fallbacks", last_error or "Search engines returned no parsable results (likely blocked)."


def fetch_page_text(url: str, max_chars: int = 4000) -> str | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text[:max_chars] if text.strip() else None
    except Exception:
        return None


def check_sentence_online(sentence: str, api_key: str | None = None, cx: str | None = None,
                           compare_mode: str = "snippet", num_results: int = 3,
                           delay: float = 1.0, brave_key: str | None = None) -> dict:
    """Result always has 'status': 'ok' | 'no_results' | 'error', plus 'backend'."""
    results, backend, error = search(sentence, api_key, cx, num_results, brave_key=brave_key)
    time.sleep(delay)
    if error and not results:
        return {"status": "error", "matched": False, "backend": backend, "error": error}
    if not results:
        return {"status": "no_results", "matched": False, "backend": backend}

    best = None
    for res in results:
        compare_text = res["snippet"]
        if compare_mode == "page":
            page_text = fetch_page_text(res["url"])
            if page_text:
                compare_text = page_text
        if not compare_text.strip():
            continue
        scores = fused_score(sentence, compare_text)
        # Fused score catches paraphrase; phrase containment catches copied
        # wording even when the snippet is much shorter than the sentence.
        copied = phrase_containment(sentence, compare_text)
        final = max(scores["fused"], copied)
        if best is None or final > best["score"]:
            best = {
                "url": res["url"], "title": res["title"], "snippet": res["snippet"],
                "score": final, "lexical": scores["lexical"],
                "semantic": scores["semantic"], "phrase_overlap": copied,
            }
    if best is None:
        return {"status": "no_results", "matched": False, "backend": backend}
    return {"status": "ok", "matched": True, "backend": backend, **best}
