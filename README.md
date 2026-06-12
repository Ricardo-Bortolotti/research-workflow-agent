# Book Research Agent

AI-powered system that transforms books, articles, and PDFs into structured knowledge using RAG and specialized agents.

## Stack

- **RAG:** LangChain, ChromaDB, sentence-transformers
- **Agents:** LangGraph, Hugging Face Inference API
- **Backend:** FastAPI
- **UI:** Streamlit

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env with your Hugging Face API token
```

## Manual tests

```bash
uv run python scripts/test_loader.py sample.pdf
uv run pytest -v
```

## Notebooks

- `notebooks/01_test_rag_pipeline.ipynb` — Index PDF and run similarity search
- `notebooks/02_test_llm.ipynb` — Test Hugging Face LLM
- `notebooks/03_test_summary_agent.ipynb` — Full RAG + Summary Agent
- `notebooks/04_test_concept_agent.ipynb` — Full RAG + Concept Agent (structured JSON)
- `notebooks/05_test_quiz_agent.ipynb` — Full RAG + Quiz Agent
- `notebooks/06_test_flashcard_agent.ipynb` — Full RAG + Flashcard Agent
- `notebooks/07_test_mindmap_agent.ipynb` — Full RAG + Mind Map Agent

## Agent workflow (LangGraph)

```python
from graph.workflow import run_workflow
from rag.embeddings import ChunkEmbedder
from rag.retriever import DocumentRetriever
from rag.vectorstore import ChromaVectorStore

store = ChromaVectorStore()
store.load_collection()
retriever = DocumentRetriever(vector_store=store, embedder=ChunkEmbedder())
context = retriever.invoke("Your question")

result = run_workflow("Your question", context)
print(result["summary"])
print(result["mindmap"]["text"])
```
