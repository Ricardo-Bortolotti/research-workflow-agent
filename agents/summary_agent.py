"""Summary agent: executive summary and key insights from retrieved context."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from app.llm import HuggingFaceLLM

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert research analyst. Use only the provided context to answer. "
    "If the context is insufficient, state what is missing instead of inventing facts."
)

SUMMARY_PROMPT_TEMPLATE = """Analyze the context below and answer the research question.

Research question:
{question}

Context:
{context}

Return your answer using exactly this structure:

EXECUTIVE SUMMARY:
<Write a concise executive summary in 2-3 short paragraphs.>

KEY INSIGHTS:
- <insight 1>
- <insight 2>
- <insight 3>
"""


@dataclass(frozen=True)
class SummaryResult:
    """Structured output from the summary agent."""

    executive_summary: str
    key_insights: list[str]


class SummaryAgent:
    """Generate an executive summary and key insights from retrieved chunks."""

    def __init__(self, llm: HuggingFaceLLM | None = None) -> None:
        self.llm = llm or HuggingFaceLLM()

    def run(self, question: str, context_documents: Sequence[Document]) -> SummaryResult:
        """Produce a summary grounded in retrieved context.

        Args:
            question: Research question guiding the summary.
            context_documents: Chunks retrieved by the RAG retriever.

        Returns:
            SummaryResult with executive_summary and key_insights.
        """
        if not question.strip():
            raise ValueError("Question text cannot be empty")
        if not context_documents:
            raise ValueError("At least one context document is required")

        context = _format_context(context_documents)
        prompt = SUMMARY_PROMPT_TEMPLATE.format(question=question.strip(), context=context)

        logger.info(
            "Running summary agent with %d context chunk(s) for question: %r",
            len(context_documents),
            question,
        )

        raw_response = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1200,
            temperature=0.2,
        )
        return _parse_summary_response(raw_response)


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


def _parse_summary_response(text: str) -> SummaryResult:
    summary_match = re.search(
        r"EXECUTIVE SUMMARY:\s*(.*?)\s*KEY INSIGHTS:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    insights_match = re.search(r"KEY INSIGHTS:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)

    executive_summary = summary_match.group(1).strip() if summary_match else text.strip()
    insights_block = insights_match.group(1).strip() if insights_match else ""

    key_insights = [
        insight.lstrip("-• ").strip()
        for insight in insights_block.splitlines()
        if insight.strip().lstrip("-• ")
    ]

    if not key_insights and insights_block:
        key_insights = [insights_block]

    return SummaryResult(
        executive_summary=executive_summary,
        key_insights=key_insights,
    )
