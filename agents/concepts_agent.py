"""Concept agent: structured extraction of key concepts from retrieved context."""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.documents import Document

from app.llm import HuggingFaceLLM

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert knowledge engineer. Extract concepts only from the provided context. "
    "Return valid JSON only, with no markdown fences or extra commentary."
)

CONCEPT_PROMPT_TEMPLATE = """Extract the most important concepts from the context below.

Research focus:
{question}

Context:
{context}

Return a JSON object with this exact schema:
{{
  "concepts": [
    {{
      "concept": "<short concept name>",
      "definition": "<clear definition grounded in the context>",
      "relevance": "<why this concept matters for the research focus>"
    }}
  ]
}}

Rules:
- Extract between 3 and 8 concepts.
- Use only information supported by the context.
- Keep definitions concise but precise.
- Output JSON only.
"""


@dataclass(frozen=True)
class ConceptItem:
    """A single extracted concept with definition and relevance."""

    concept: str
    definition: str
    relevance: str


@dataclass(frozen=True)
class ConceptResult:
    """Structured JSON-serializable output from the concept agent."""

    concepts: list[ConceptItem]

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        """Convert to a plain dictionary."""
        return {"concepts": [asdict(item) for item in self.concepts]}

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ConceptAgentError(Exception):
    """Raised when concept extraction or JSON parsing fails."""


class ConceptAgent:
    """Extract structured concepts, definitions, and relevance from retrieved chunks."""

    def __init__(self, llm: HuggingFaceLLM | None = None) -> None:
        self.llm = llm or HuggingFaceLLM()

    def run(self, question: str, context_documents: Sequence[Document]) -> ConceptResult:
        """Extract concepts grounded in retrieved context.

        Args:
            question: Research question or topic guiding concept extraction.
            context_documents: Chunks retrieved by the RAG retriever.

        Returns:
            ConceptResult containing a list of ConceptItem objects.
        """
        if not question.strip():
            raise ValueError("Question text cannot be empty")
        if not context_documents:
            raise ValueError("At least one context document is required")

        context = _format_context(context_documents)
        prompt = CONCEPT_PROMPT_TEMPLATE.format(question=question.strip(), context=context)

        logger.info(
            "Running concept agent with %d context chunk(s) for question: %r",
            len(context_documents),
            question,
        )

        raw_response = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.1,
        )
        return _parse_concept_response(raw_response)


def _format_context(documents: Sequence[Document]) -> str:
    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "n/a")
        sections.append(
            f"[Chunk {index} | source: {source} | page: {page}]\n{document.page_content.strip()}"
        )
    return "\n\n".join(sections)


def _parse_concept_response(text: str) -> ConceptResult:
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse concept agent JSON response")
        raise ConceptAgentError(f"Invalid JSON from concept agent: {exc}") from exc

    concepts_raw = payload.get("concepts")
    if not isinstance(concepts_raw, list):
        raise ConceptAgentError("JSON response must contain a 'concepts' array")

    concepts: list[ConceptItem] = []
    for index, item in enumerate(concepts_raw):
        if not isinstance(item, dict):
            raise ConceptAgentError(f"Concept at index {index} must be an object")
        concepts.append(_parse_concept_item(item, index))

    if not concepts:
        raise ConceptAgentError("Concept agent returned an empty concepts list")

    return ConceptResult(concepts=concepts)


def _parse_concept_item(item: dict[str, Any], index: int) -> ConceptItem:
    missing = [
        field
        for field in ("concept", "definition", "relevance")
        if not str(item.get(field, "")).strip()
    ]
    if missing:
        raise ConceptAgentError(
            f"Concept at index {index} is missing required fields: {', '.join(missing)}"
        )

    return ConceptItem(
        concept=str(item["concept"]).strip(),
        definition=str(item["definition"]).strip(),
        relevance=str(item["relevance"]).strip(),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ConceptAgentError("Concept agent JSON root must be an object")
    return payload
