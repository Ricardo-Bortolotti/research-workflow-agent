"""Streamlit UI for the Book Research Agent."""

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Book Research Agent", page_icon="📚", layout="wide")
st.title("Book Research Agent")
st.caption("Upload a PDF, run the agent workflow, and explore structured study materials.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.markdown("**Local run (Docker):**")
    st.code("docker compose up --build", language="bash")


def _api_client() -> httpx.Client:
    return httpx.Client(base_url=api_url, timeout=600.0)


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_area(
    "Research question",
    value="What are the main ideas in this document?",
    height=100,
)

analyze_clicked = st.button("Run analysis", type="primary", disabled=uploaded_file is None)

if analyze_clicked and uploaded_file is not None:
    with st.spinner("Uploading PDF and running agents. This may take several minutes..."):
        try:
            with _api_client() as client:
                health = client.get("/health")
                health.raise_for_status()

                upload_response = client.post(
                    "/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
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
