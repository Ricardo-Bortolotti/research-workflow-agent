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

## Run the API

```bash
uv run uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a PDF |
| `POST` | `/analyze` | Index document + run agent workflow |
| `GET` | `/results/{analysis_id}` | Fetch structured results |
| `GET` | `/health` | Health check |

## Run Streamlit UI

```bash
# Terminal 1 — API
uv run uvicorn app.main:app --reload

# Terminal 2 — UI
uv run streamlit run ui/streamlit_app.py
```

Set `API_URL` in `.env` if the API is not on `http://localhost:8000`.

## Manual tests

```bash
uv run python scripts/test_loader.py sample.pdf
uv run pytest -v
```

## Notebooks

- `01_test_rag_pipeline.ipynb` — Index PDF and similarity search
- `02_test_llm.ipynb` — Hugging Face LLM
- `03_test_summary_agent.ipynb` — Summary Agent
- `04_test_concept_agent.ipynb` — Concept Agent
- `05_test_quiz_agent.ipynb` — Quiz Agent
- `06_test_flashcard_agent.ipynb` — Flashcard Agent
- `07_test_mindmap_agent.ipynb` — Mind Map Agent
- `08_test_workflow.ipynb` — Full LangGraph workflow

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
print(result["mindmap"]["text"])
```
