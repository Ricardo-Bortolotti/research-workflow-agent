"""Quiz agent: structured quiz generation from retrieved context."""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from langchain_core.documents import Document

from app.llm import HuggingFaceLLM

logger = logging.getLogger(__name__)

DifficultyLevel = Literal["easy", "medium", "hard"]
VALID_DIFFICULTY_LEVELS = frozenset({"easy", "medium", "hard"})

SYSTEM_PROMPT = (
    "You are an expert educator. Create quiz questions only from the provided context. "
    "Return valid JSON only, with no markdown fences or extra commentary."
)

QUIZ_PROMPT_TEMPLATE = """Create review quiz questions from the context below.

Topic focus:
{question}

Context:
{context}

Return a JSON object with this exact schema:
{{
  "questions": [
    {{
      "question": "<quiz question grounded in the context>",
      "answer": "<concise correct answer>",
      "difficulty": "<easy | medium | hard>"
    }}
  ]
}}

Rules:
- Generate between 3 and 6 questions.
- Use only information supported by the context.
- Mix difficulty levels when possible.
- Answers must be specific and verifiable from the context.
- difficulty must be exactly one of: easy, medium, hard.
- Output JSON only.
"""


@dataclass(frozen=True)
class QuizItem:
    """A single quiz question with answer and difficulty level."""

    question: str
    answer: str
    difficulty: DifficultyLevel


@dataclass(frozen=True)
class QuizResult:
    """Structured JSON-serializable output from the quiz agent."""

    questions: list[QuizItem]

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        """Convert to a plain dictionary."""
        return {"questions": [asdict(item) for item in self.questions]}

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class QuizAgentError(Exception):
    """Raised when quiz generation or JSON parsing fails."""


class QuizAgent:
    """Generate structured quiz questions from retrieved document chunks."""

    def __init__(self, llm: HuggingFaceLLM | None = None) -> None:
        self.llm = llm or HuggingFaceLLM()

    def run(self, question: str, context_documents: Sequence[Document]) -> QuizResult:
        """Generate quiz questions grounded in retrieved context.

        Args:
            question: Topic or research question guiding quiz generation.
            context_documents: Chunks retrieved by the RAG retriever.

        Returns:
            QuizResult containing a list of QuizItem objects.
        """
        if not question.strip():
            raise ValueError("Question text cannot be empty")
        if not context_documents:
            raise ValueError("At least one context document is required")

        context = _format_context(context_documents)
        prompt = QUIZ_PROMPT_TEMPLATE.format(question=question.strip(), context=context)

        logger.info(
            "Running quiz agent with %d context chunk(s) for question: %r",
            len(context_documents),
            question,
        )

        raw_response = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.3,
        )
        return _parse_quiz_response(raw_response)


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


def _parse_quiz_response(text: str) -> QuizResult:
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse quiz agent JSON response")
        raise QuizAgentError(f"Invalid JSON from quiz agent: {exc}") from exc

    questions_raw = payload.get("questions")
    if not isinstance(questions_raw, list):
        raise QuizAgentError("JSON response must contain a 'questions' array")

    questions: list[QuizItem] = []
    for index, item in enumerate(questions_raw):
        if not isinstance(item, dict):
            raise QuizAgentError(f"Question at index {index} must be an object")
        questions.append(_parse_quiz_item(item, index))

    if not questions:
        raise QuizAgentError("Quiz agent returned an empty questions list")

    return QuizResult(questions=questions)


def _parse_quiz_item(item: dict[str, Any], index: int) -> QuizItem:
    missing = [
        field
        for field in ("question", "answer", "difficulty")
        if not str(item.get(field, "")).strip()
    ]
    if missing:
        raise QuizAgentError(
            f"Question at index {index} is missing required fields: {', '.join(missing)}"
        )

    difficulty = str(item["difficulty"]).strip().lower()
    if difficulty not in VALID_DIFFICULTY_LEVELS:
        raise QuizAgentError(
            f"Question at index {index} has invalid difficulty '{item['difficulty']}'. "
            f"Expected one of: {', '.join(sorted(VALID_DIFFICULTY_LEVELS))}"
        )

    return QuizItem(
        question=str(item["question"]).strip(),
        answer=str(item["answer"]).strip(),
        difficulty=difficulty,  # type: ignore[arg-type]
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
        raise QuizAgentError("Quiz agent JSON root must be an object")
    return payload
