"""
PlagFlex - An Intelligent Document Similarity Framework for Plagiarism
Analysis and Interactive Visualization.

Rebuilt around the project report's design: input -> pre-processing ->
feature extraction (lexical + semantic) -> similarity analysis / score
fusion -> tiered risk classification -> visualization -> report.
"""
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from file_readers import get_text_from_file
from preprocessing import get_sentences
from similarity import fused_score, lexical_similarity, risk_level, semantic_model_available, RISK_COLORS
from web_check import check_sentence_online

st.set_page_config(page_title="PlagFlex", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------
# Sidebar: configuration shared by both tabs
# --------------------------------------------------------------------------
st.sidebar.title("PlagFlex")
st.sidebar.caption("Document similarity & plagiarism analysis")

with st.sidebar.expander("Pre-processing", expanded=False):
    remove_stopwords = st.checkbox("Remove stop-words", value=True)
    lemmatize = st.checkbox("Lemmatize", value=True)

with st.sidebar.expander("Risk thresholds", expanded=False):
    low_threshold = st.slider("Low → Medium boundary", 0.0, 1.0, 0.30, 0.05)
    high_threshold = st.slider("Medium → High boundary", 0.0, 1.0, 0.60, 0.05)
    if high_threshold < low_threshold:
        st.warning("High threshold should be ≥ low threshold.")

semantic_on = semantic_model_available()
with st.sidebar.expander("Web check settings", expanded=False):
    st.caption(
        "Uses Google's Programmable Search Engine API when credentials are "
        "supplied. Without them, it falls back to scraping Google search "
        "results directly, which Google may rate-limit or block - fine for "
        "occasional local use, not for production."
    )
    try:
        _secret_brave = st.secrets.get("BRAVE_API_KEY", "")
    except Exception:
        _secret_brave = ""
    try:
        _secret_tavily = st.secrets.get("TAVILY_API_KEY", "")
    except Exception:
        _secret_tavily = ""
    tavily_key = st.text_input(
        "Tavily API key (recommended, free tier)", type="password", value=_secret_tavily,
        help="1,000 free searches/month, no credit card. Sign up at https://tavily.com",
    )
    brave_key = st.text_input(
        "Brave Search API key (optional; card required)", type="password", value=_secret_brave,
        help="Free/cheap official API that works from cloud hosts. Get one at https://api-dashboard.search.brave.com/",
    )
    gcse_api_key = st.text_input("Google API key (optional, legacy)", type="password")
    gcse_cx = st.text_input("Search Engine ID / cx (optional)")
    compare_mode = st.radio("Compare against", ["Search snippet (fast)", "Full source page (slower)"])
    max_sentences_web = st.number_input(
        "Max sentences to check online", min_value=1, max_value=100, value=20,
        help="Caps how many sentences get sent to the search engine, to keep runtime and rate-limit risk reasonable.",
    )

if st.sidebar.button("Test web search"):
    _t = check_sentence_online(
        "The quick brown fox jumps over the lazy dog near the river bank",
        api_key=gcse_api_key or None, cx=gcse_cx or None, delay=0, brave_key=brave_key or None, tavily_key=tavily_key or None,
    )
    if _t.get("status") == "error":
        st.sidebar.error(f"{_t['backend']} failed: {_t['error']}")
    elif _t.get("status") == "no_results":
        st.sidebar.warning(f"{_t['backend']} returned no results.")
    else:
        st.sidebar.success(f"Working via {_t['backend']}. Top hit: {_t['url']}")

st.sidebar.info(
    f"Semantic (embedding) similarity: {'enabled' if semantic_on else 'unavailable - install `sentence-transformers` to enable'}"
)

tab_analyze, tab_compare, tab_about = st.tabs(
    ["📄 Analyze a document", "🔀 Compare documents", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# Tab 1: analyze a single document, optionally against the web
# --------------------------------------------------------------------------
with tab_analyze:
    st.subheader("Analyze a document")
    col_input, col_ref = st.columns(2)

    with col_input:
        source = st.radio("Input", ["Paste text", "Upload file"], horizontal=True, key="src1")
        if source == "Paste text":
            text = st.text_area("Text to check", height=220)
        else:
            uploaded = st.file_uploader("Upload a .txt, .docx or .pdf file", type=["txt", "docx", "pdf"])
            text = get_text_from_file(uploaded) if uploaded else ""

    with col_ref:
        st.markdown("**Optional: compare against a reference document**")
        ref_uploaded = st.file_uploader(
            "Upload a reference file (leave empty to only check the web)",
            type=["txt", "docx", "pdf"], key="ref_upload",
        )
        reference_text = get_text_from_file(ref_uploaded) if ref_uploaded else ""
        check_web = st.checkbox("Also check each sentence against the web", value=False)

    if st.button("Run analysis", type="primary"):
        if not text.strip():
            st.warning("Please provide some text to analyze.")
            st.stop()

        sentences = get_sentences(text)
        if not sentences:
            st.warning("Couldn't find any sentences in the input.")
            st.stop()

        if not reference_text.strip() and not check_web:
            st.warning(
                "Nothing to compare against: upload a reference document and/or tick "
                "'Also check each sentence against the web', otherwise every score is 0."
            )
            st.stop()

        rows = []
        web_statuses = []
        progress = st.progress(0.0, text="Analyzing...")
        checked_online = 0
        for i, sentence in enumerate(sentences):
            row = {"Sentence": sentence}

            if reference_text.strip():
                scores = fused_score(sentence, reference_text, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
                row["Reference score"] = scores["fused"]

            if check_web and checked_online < max_sentences_web and len(sentence.split()) >= 6:
                result = check_sentence_online(
                    sentence,
                    api_key=gcse_api_key or None,
                    cx=gcse_cx or None,
                    brave_key=brave_key or None, tavily_key=tavily_key or None,
                    compare_mode="page" if compare_mode.startswith("Full") else "snippet",
                )
                checked_online += 1
                web_statuses.append(result)
                row["Web status"] = result.get("status", "ok" if result.get("matched") else "no_results")
                if result.get("matched"):
                    row["Web score"] = result["score"]
                    row["Source URL"] = result["url"]
                else:
                    row["Web score"] = 0.0
                    row["Source URL"] = result.get("error", "")

            best_score = max(
                [v for k, v in row.items() if k.endswith("score")] or [0.0]
            )
            row["Overall score"] = best_score
            row["Risk"] = risk_level(best_score, low_threshold, high_threshold)
            rows.append(row)
            progress.progress((i + 1) / len(sentences), text=f"Analyzing sentence {i + 1}/{len(sentences)}")
        progress.empty()

        if check_web:
            errors = [r for r in web_statuses if r.get("status") == "error"]
            empty = [r for r in web_statuses if r.get("status") == "no_results"]
            if web_statuses and len(errors) == len(web_statuses):
                st.error(
                    f"Web search failed for every sentence ({web_statuses[0].get('backend', 'unknown')}): "
                    f"{errors[0].get('error', 'unknown error')}. The 0% scores below mean 'not checked', not 'original'. "
                    "Add a Tavily (free) or Brave API key in the sidebar, or use the 'Test web search' button."
                )
            elif errors or empty:
                st.info(f"Web check: {len(web_statuses) - len(errors) - len(empty)} sentences matched a source, "
                        f"{len(empty)} returned no results, {len(errors)} failed.")

        df = pd.DataFrame(rows)
        overall = df["Overall score"].mean() if len(df) else 0.0
        overall_risk = risk_level(overall, low_threshold, high_threshold)

        # ---- Dashboard --------------------------------------------------
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall similarity", f"{overall * 100:.1f}%")
        m2.metric("Overall risk", overall_risk)
        m3.metric("Sentences flagged (Medium/High)", int((df["Risk"] != "Low").sum()))

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall * 100,
            title={"text": "Overall similarity score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS[overall_risk]},
                "steps": [
                    {"range": [0, low_threshold * 100], "color": "#eafaf1"},
                    {"range": [low_threshold * 100, high_threshold * 100], "color": "#fef5e7"},
                    {"range": [high_threshold * 100, 100], "color": "#fdedec"},
                ],
            },
        ))
        st.plotly_chart(gauge, use_container_width=True)

        risk_counts = df["Risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        bar = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index, color_discrete_map=RISK_COLORS,
            labels={"x": "Risk level", "y": "Sentence count"}, title="Risk distribution across sentences",
        )
        st.plotly_chart(bar, use_container_width=True)

        st.markdown("### Highlighted document")
        highlighted = []
        for row in rows:
            color = RISK_COLORS[row["Risk"]]
            if row["Risk"] == "Low":
                highlighted.append(row["Sentence"])
            else:
                highlighted.append(
                    f'<span style="background-color:{color}33;border-bottom:2px solid {color};" '
                    f'title="{row["Risk"]} risk - {row["Overall score"]*100:.0f}% similar">{row["Sentence"]}</span>'
                )
        st.markdown(" ".join(highlighted), unsafe_allow_html=True)

        st.markdown("### Sentence-level detail")
        display_df = df.copy()
        for col in display_df.columns:
            if col.endswith("score") or col == "Overall score":
                display_df[col] = (display_df[col] * 100).round(1)
        st.dataframe(display_df, use_container_width=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download report (CSV)", csv, "plagflex_report.csv", "text/csv")

# --------------------------------------------------------------------------
# Tab 2: compare multiple documents against each other
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader("Compare multiple documents")
    files = st.file_uploader(
        "Upload two or more files (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"],
        accept_multiple_files=True, key="multi_upload",
    )

    if st.button("Compare", type="primary"):
        if not files or len(files) < 2:
            st.warning("Upload at least two files to compare.")
            st.stop()

        names = [f.name for f in files]
        texts = [get_text_from_file(f) for f in files]

        pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                scores = fused_score(texts[i], texts[j], remove_stopwords=remove_stopwords, lemmatize=lemmatize)
                pairs.append({
                    "File 1": names[i], "File 2": names[j],
                    "Similarity": scores["fused"],
                    "Risk": risk_level(scores["fused"], low_threshold, high_threshold),
                })

        df = pd.DataFrame(pairs).sort_values("Similarity", ascending=False).reset_index(drop=True)
        display_df = df.copy()
        display_df["Similarity"] = (display_df["Similarity"] * 100).round(1)
        st.dataframe(display_df, use_container_width=True)

        bar = px.bar(
            df, x=[f"{r['File 1']} vs {r['File 2']}" for _, r in df.iterrows()], y="Similarity",
            color="Risk", color_discrete_map=RISK_COLORS, title="Pairwise similarity",
            labels={"x": "File pair", "Similarity": "Similarity score"},
        )
        st.plotly_chart(bar, use_container_width=True)

        if len(names) > 2:
            import numpy as np
            matrix = pd.DataFrame(1.0, index=names, columns=names)
            for _, r in df.iterrows():
                matrix.loc[r["File 1"], r["File 2"]] = r["Similarity"]
                matrix.loc[r["File 2"], r["File 1"]] = r["Similarity"]
            heat = px.imshow(matrix, text_auto=".2f", color_continuous_scale="Reds", title="Similarity heatmap")
            st.plotly_chart(heat, use_container_width=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download report (CSV)", csv, "plagflex_comparison.csv", "text/csv")

# --------------------------------------------------------------------------
# Tab 3: about / methodology
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("How PlagFlex scores similarity")
    st.markdown(
        """
Following the report's design (input → pre-processing → feature extraction →
similarity analysis → visualization → report):

1. **Pre-processing** - text is cleaned, sentence-tokenized, and (optionally)
   stop-words are removed and terms are lemmatized.
2. **Feature extraction** - each sentence/document is represented lexically
   (TF-IDF) and, when the optional `sentence-transformers` package is
   installed, semantically (dense sentence embeddings).
3. **Score fusion** - lexical and semantic cosine-similarity scores are
   combined into one weighted score (falls back to lexical-only when the
   embedding model isn't installed).
4. **Risk classification** - the fused score is mapped to Low / Medium /
   High risk using the thresholds set in the sidebar.
5. **Web check** - each sentence can be searched on the web (via Google's
   Programmable Search Engine API, or a best-effort scrape fallback) and
   compared against the best-matching result.
6. **Visualization & report** - results are shown as a gauge, a risk-
   distribution chart, inline highlighting, and a downloadable CSV report.

**Setting up the web check API (recommended):**
1. Create a Programmable Search Engine at
   https://programmablesearchengine.google.com/ (set it to search the
   entire web).
2. Get an API key from https://console.cloud.google.com/ with the
   "Custom Search API" enabled.
3. Paste the API key and the Search Engine ID (cx) into the sidebar.

Without those credentials the app falls back to scraping Google search
results directly, which is unofficial and can be blocked at any time -
fine for casual, local testing only.
        """
    )
