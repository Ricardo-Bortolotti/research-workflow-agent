"""Streamlit UI for the Book Research Agent."""

import json
import os

import httpx
import streamlit as st


def _default_api_url() -> str:
    """Resolve API base URL from env (Docker/local) or Streamlit Cloud secrets."""
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = st.secrets.get("API_URL")
        if secret_url:
            return str(secret_url).rstrip("/")
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return "http://localhost:8000"


API_URL = _default_api_url()
DEFAULT_TOP_K = 5

st.set_page_config(
    page_title="Book Research Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Book Research Agent")
st.caption("Upload a PDF, run the agent workflow, and explore structured study materials.")

st.markdown(
    "Choose a PDF to analyze. The document is sent to the API, indexed for semantic search, "
    "and used as the sole source of context for every agent in the workflow."
)
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

st.markdown(
    "Ask a focused question about the document. The retriever selects the most relevant "
    "passages for that question, and all agents (summary, concepts, quiz, flashcards, "
    "and mind map) answer using only that retrieved context."
)
question = st.text_area(
    "Research question",
    value="What are the main ideas in this document?",
    height=100,
)

with st.expander("Retrieval settings"):
    top_k = st.slider(
        "Retrieved chunks (top_k)",
        min_value=1,
        max_value=20,
        value=DEFAULT_TOP_K,
        help="Number of document chunks passed to each agent. Higher values add context "
        "but increase processing time and API cost.",
    )

analyze_clicked = st.button("Run analysis", type="primary", disabled=uploaded_file is None)


def _api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=600.0)


if analyze_clicked and uploaded_file is not None:
    with st.spinner("Uploading PDF and running agents. This may take several minutes..."):
        try:
            with _api_client() as client:
                health = client.get("/health")
                health.raise_for_status()

                upload_response = client.post(
                    "/upload",
                    files={
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    },
                )
                upload_response.raise_for_status()
                upload_data = upload_response.json()
                document_id = upload_data["document_id"]

                analyze_response = client.post(
                    "/analyze",
                    json={
                        "document_id": document_id,
                        "question": question.strip(),
                        "top_k": top_k,
                    },
                )
                analyze_response.raise_for_status()
                analyze_data = analyze_response.json()

                if analyze_data["status"] != "completed":
                    st.error(f"Analysis failed: {analyze_data.get('message', 'Unknown error')}")
                    st.stop()

                results_response = client.get(f"/results/{analyze_data['analysis_id']}")
                results_response.raise_for_status()
                results = results_response.json()

            st.session_state["results"] = results
            st.success(f"Analysis completed: {results['analysis_id']}")
        except httpx.HTTPError as exc:
            st.error(f"API request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


results = st.session_state.get("results")

if results:
    st.markdown("---")
    st.subheader("Analysis results")
    st.write(f"**Question:** {results.get('question', '')}")
    st.write(f"**Analysis ID:** `{results.get('analysis_id', '')}`")

    summary = results.get("summary")
    if summary:
        st.markdown("### Executive summary")
        st.write(summary.get("executive_summary", ""))
        insights = summary.get("key_insights", [])
        if insights:
            st.markdown("**Key insights**")
            for insight in insights:
                st.markdown(f"- {insight}")

    concepts = results.get("concepts", [])
    if concepts:
        st.markdown("### Concepts")
        for item in concepts:
            with st.expander(item.get("concept", "Concept")):
                st.write(f"**Definition:** {item.get('definition', '')}")
                st.write(f"**Relevance:** {item.get('relevance', '')}")

    quiz = results.get("quiz", [])
    if quiz:
        st.markdown("### Quiz")
        for index, item in enumerate(quiz, start=1):
            difficulty = item.get("difficulty", "n/a").upper()
            st.markdown(f"**{index}. [{difficulty}] {item.get('question', '')}**")
            with st.expander("Show answer"):
                st.write(item.get("answer", ""))

    flashcards = results.get("flashcards", [])
    if flashcards:
        st.markdown("### Flashcards")
        cols = st.columns(2)
        for index, card in enumerate(flashcards):
            with cols[index % 2]:
                st.info(f"**Front:** {card.get('front', '')}")
                st.success(f"**Back:** {card.get('back', '')}")

    mindmap = results.get("mindmap")
    if mindmap:
        st.markdown("### Mind map")
        st.code(mindmap.get("text", ""), language=None)
        with st.expander("Mind map JSON"):
            st.json(mindmap)

    with st.expander("Raw JSON response"):
        st.code(json.dumps(results, indent=2), language="json")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 1rem 0 2rem;">
        Developed by Ricardo Bortolotti<br>
        <a href="https://www.linkedin.com/in/ricardo-bortolotti" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/Ricardo-Bortolotti" target="_blank" rel="noopener noreferrer">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
