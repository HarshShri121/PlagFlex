"""
Web plagiarism check module.

For a given sentence, looks the sentence up on the web and reports the
best-matching source and how similar that source's text is to the
sentence. Two search backends are supported:

1. Google Programmable Search Engine (Custom Search JSON API) - the
   supported, ToS-compliant way to query Google. Needs an API key + a
   Search Engine ID (cx), both free to create at
   https://programmablesearchengine.google.com/.
2. A lightweight HTML-scrape fallback (used only when no API key/cx is
   supplied), kept for parity with the original prototype. Google can
   rate-limit or block this at any time, so it is best-effort only and
   should not be relied on for anything beyond casual/local checks.
"""
from __future__ import annotations

import re
import time
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from similarity import fused_score

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT = 10


def _search_via_cse(query: str, api_key: str, cx: str, num: int = 3) -> list[dict]:
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": num},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in items
    ]


def _search_via_scrape(query: str, num: int = 3) -> list[dict]:
    url = f"https://www.google.com/search?q={quote_plus(query)}&num={num}"
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


def fetch_page_text(url: str, max_chars: int = 4000) -> str | None:
    """Best-effort fetch of a source page's visible paragraph text."""
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
                           delay: float = 1.0) -> dict:
    """Search the web for `sentence` and return the best-matching source.

    compare_mode: "snippet" compares against the search-result snippet
    (fast, no extra requests). "page" additionally fetches the full
    source page for a more accurate - but much slower and less
    reliable - comparison.
    """
    try:
        if api_key and cx:
            results = _search_via_cse(sentence, api_key, cx, num_results)
        else:
            results = _search_via_scrape(sentence, num_results)
    except Exception as exc:
        return {"matched": False, "error": str(exc)}
    finally:
        time.sleep(delay)

    if not results:
        return {"matched": False}

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
        if best is None or scores["fused"] > best["score"]:
            best = {
                "url": res["url"],
                "title": res["title"],
                "snippet": res["snippet"],
                "score": scores["fused"],
                "lexical": scores["lexical"],
                "semantic": scores["semantic"],
            }
    if best is None:
        return {"matched": False}
    return {"matched": True, **best}
