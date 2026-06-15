# Book Research Agent — single image for API (Railway) and Streamlit (local compose)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

# System deps occasionally required by wheels (e.g. chromadb / crypto)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast, reproducible dependency sync)
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# Dependency layer (cached when lockfile unchanged)
COPY pyproject.toml uv.lock README.md ./
COPY app/ app/
COPY agents/ agents/
COPY api/ api/
COPY graph/ graph/
COPY rag/ rag/
COPY ui/ ui/

RUN uv sync --frozen --no-dev

# Optional: skip in CI for faster Docker build validation
ARG PRELOAD_EMBEDDINGS=true
RUN if [ "$PRELOAD_EMBEDDINGS" = "true" ]; then \
        uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"; \
    fi

EXPOSE 8000 8501

# Default: API server (Railway overrides PORT; compose overrides command for Streamlit)
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
