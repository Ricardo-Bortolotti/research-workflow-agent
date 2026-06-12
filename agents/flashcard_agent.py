"""Flashcard agent: study flashcards generated from retrieved context."""

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
    "You are an expert study coach. Create flashcards only from the provided context. "
    "Return valid JSON only, with no markdown fences or extra commentary."
)

FLASHCARD_PROMPT_TEMPLATE = """Create study flashcards from the key concepts in the context below.

Topic focus:
{question}

Context:
{context}

Return a JSON object with this exact schema:
{{
  "flashcards": [
    {{
      "front": "<term, question, or prompt on the front of the card>",
      "back": "<concise answer or explanation on the back>"
    }}
  ]
}}

Rules:
- Generate between 5 and 10 flashcards.
- Use only information supported by the context.
- Front side should be short and memorable.
- Back side should be clear and self-contained.
- Cover the most important concepts from the context.
- Output JSON only.
"""


@dataclass(frozen=True)
class FlashcardItem:
    """A single flashcard with front and back sides."""

    front: str
    back: str


@dataclass(frozen=True)
class FlashcardResult:
    """Structured JSON-serializable output from the flashcard agent."""

    flashcards: list[FlashcardItem]

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        """Convert to a plain dictionary."""
        return {"flashcards": [asdict(item) for item in self.flashcards]}

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class FlashcardAgentError(Exception):
    """Raised when flashcard generation or JSON parsing fails."""


class FlashcardAgent:
    """Generate front/back flashcard pairs from retrieved document chunks."""

    def __init__(self, llm: HuggingFaceLLM | None = None) -> None:
        self.llm = llm or HuggingFaceLLM()

    def run(self, question: str, context_documents: Sequence[Document]) -> FlashcardResult:
        """Generate flashcards grounded in retrieved context.

        Args:
            question: Topic or research question guiding flashcard creation.
            context_documents: Chunks retrieved by the RAG retriever.

        Returns:
            FlashcardResult containing a list of FlashcardItem objects.
        """
        if not question.strip():
            raise ValueError("Question text cannot be empty")
        if not context_documents:
            raise ValueError("At least one context document is required")

        context = _format_context(context_documents)
        prompt = FLASHCARD_PROMPT_TEMPLATE.format(question=question.strip(), context=context)

        logger.info(
            "Running flashcard agent with %d context chunk(s) for question: %r",
            len(context_documents),
            question,
        )

        raw_response = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.2,
        )
        return _parse_flashcard_response(raw_response)


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


def _parse_flashcard_response(text: str) -> FlashcardResult:
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse flashcard agent JSON response")
        raise FlashcardAgentError(f"Invalid JSON from flashcard agent: {exc}") from exc

    flashcards_raw = payload.get("flashcards")
    if not isinstance(flashcards_raw, list):
        raise FlashcardAgentError("JSON response must contain a 'flashcards' array")

    flashcards: list[FlashcardItem] = []
    for index, item in enumerate(flashcards_raw):
        if not isinstance(item, dict):
            raise FlashcardAgentError(f"Flashcard at index {index} must be an object")
        flashcards.append(_parse_flashcard_item(item, index))

    if not flashcards:
        raise FlashcardAgentError("Flashcard agent returned an empty flashcards list")

    return FlashcardResult(flashcards=flashcards)


def _parse_flashcard_item(item: dict[str, Any], index: int) -> FlashcardItem:
    missing = [field for field in ("front", "back") if not str(item.get(field, "")).strip()]
    if missing:
        raise FlashcardAgentError(
            f"Flashcard at index {index} is missing required fields: {', '.join(missing)}"
        )

    return FlashcardItem(
        front=str(item["front"]).strip(),
        back=str(item["back"]).strip(),
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
        raise FlashcardAgentError("Flashcard agent JSON root must be an object")
    return payload
