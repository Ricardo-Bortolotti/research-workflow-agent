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

## Docker (local)

**Prerequisites:** Docker Desktop, `.env` with `HUGGINGFACE_API_TOKEN`.

```bash
cp .env.example .env
# Edit .env with your Hugging Face token

docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

Stop: `docker compose down` · Reset data volumes: `docker compose down -v`

### Docker design decisions

| Decision | Why |
|----------|-----|
| **Single Dockerfile** | Same dependencies for API and UI; Railway reuses it for FastAPI only |
| **`uv sync --frozen`** | Reproducible installs from `uv.lock` |
| **Pre-download BGE model** | Avoids long delay on first `/analyze` in a fresh container |
| **Named volumes** (`app_data`, `model_cache`) | Persist PDFs, ChromaDB, and HF caches across restarts |
| **`API_URL=http://api:8000` in compose** | Streamlit container talks to API over the Docker network |
| **`PORT` in CMD** | Railway injects `$PORT`; defaults to `8000` locally |
| **`.dockerignore`** | Keeps secrets, `.venv`, and runtime data out of the image |

## Future deploy

### FastAPI on Railway

1. Connect the GitHub repo to Railway.
2. Railway detects the `Dockerfile` (default start command runs uvicorn on `$PORT`).
3. Set environment variables:
   - `HUGGINGFACE_API_TOKEN`
   - `HF_MODEL_ID` (optional)
4. Add a **Volume** mounted at `/app/data` if you want uploads/results to persist.
5. Health check path: `/health`

### Streamlit on Streamlit Community Cloud

Streamlit Cloud runs from GitHub (not this `docker-compose`). Steps:

1. Deploy repo at [share.streamlit.io](https://share.streamlit.io).
2. Main file: `ui/streamlit_app.py`
3. In **Secrets**, set:
   ```toml
   API_URL = "https://your-railway-api.up.railway.app"
   ```
4. The UI only calls the API — no HF token needed in Streamlit secrets unless you change architecture.

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
