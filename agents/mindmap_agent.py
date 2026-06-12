"""Mind map agent: hierarchical concept organization from retrieved context."""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.llm import HuggingFaceLLM

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert knowledge organizer. Build mind maps only from the provided context. "
    "Return valid JSON only, with no markdown fences or extra commentary."
)

MINDMAP_PROMPT_TEMPLATE = """Transform the key concepts from the context below into a hierarchical mind map.

Topic focus:
{question}

Context:
{context}

Return a JSON object with this exact schema:
{{
  "title": "<central topic of the mind map>",
  "nodes": [
    {{
      "label": "<main branch label>",
      "children": [
        {{
          "label": "<sub-concept>",
          "children": []
        }}
      ]
    }}
  ]
}}

Rules:
- Organize concepts into 3 to 6 main branches under the title.
- Nest sub-concepts where it adds clarity.
- Use only information supported by the context.
- Keep labels short and descriptive.
- children must always be an array (use [] for leaf nodes).
- Output JSON only.
"""


@dataclass
class MindMapNode:
    """A node in the mind map tree."""

    label: str
    children: list["MindMapNode"] = field(default_factory=list)


@dataclass(frozen=True)
class MindMapResult:
    """Structured output from the mind map agent."""

    title: str
    nodes: list[MindMapNode]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "title": self.title,
            "nodes": [_node_to_dict(node) for node in self.nodes],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Render the mind map as an indented textual tree."""
        lines = [self.title]
        for index, node in enumerate(self.nodes):
            is_last = index == len(self.nodes) - 1
            lines.extend(_render_node(node, prefix="", is_last=is_last))
        return "\n".join(lines)


class MindMapAgentError(Exception):
    """Raised when mind map generation or JSON parsing fails."""


class MindMapAgent:
    """Organize retrieved concepts into a hierarchical mind map structure."""

    def __init__(self, llm: HuggingFaceLLM | None = None) -> None:
        self.llm = llm or HuggingFaceLLM()

    def run(self, question: str, context_documents: Sequence[Document]) -> MindMapResult:
        """Build a mind map grounded in retrieved context.

        Args:
            question: Topic or research question guiding the mind map.
            context_documents: Chunks retrieved by the RAG retriever.

        Returns:
            MindMapResult with a title and hierarchical nodes.
        """
        if not question.strip():
            raise ValueError("Question text cannot be empty")
        if not context_documents:
            raise ValueError("At least one context document is required")

        context = _format_context(context_documents)
        prompt = MINDMAP_PROMPT_TEMPLATE.format(question=question.strip(), context=context)

        logger.info(
            "Running mind map agent with %d context chunk(s) for question: %r",
            len(context_documents),
            question,
        )

        raw_response = self.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=1500,
            temperature=0.2,
        )
        return _parse_mindmap_response(raw_response)


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


def _parse_mindmap_response(text: str) -> MindMapResult:
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse mind map agent JSON response")
        raise MindMapAgentError(f"Invalid JSON from mind map agent: {exc}") from exc

    title = str(payload.get("title", "")).strip()
    if not title:
        raise MindMapAgentError("JSON response must contain a non-empty 'title'")

    nodes_raw = payload.get("nodes")
    if not isinstance(nodes_raw, list):
        raise MindMapAgentError("JSON response must contain a 'nodes' array")

    nodes = [_parse_mindmap_node(item, index) for index, item in enumerate(nodes_raw)]
    if not nodes:
        raise MindMapAgentError("Mind map agent returned an empty nodes list")

    return MindMapResult(title=title, nodes=nodes)


def _parse_mindmap_node(item: dict[str, Any], index: int) -> MindMapNode:
    if not isinstance(item, dict):
        raise MindMapAgentError(f"Node at index {index} must be an object")

    label = str(item.get("label", "")).strip()
    if not label:
        raise MindMapAgentError(f"Node at index {index} is missing a non-empty 'label'")

    children_raw = item.get("children", [])
    if children_raw is None:
        children_raw = []
    if not isinstance(children_raw, list):
        raise MindMapAgentError(f"Node at index {index} must have a 'children' array")

    children = [_parse_mindmap_node(child, child_index) for child_index, child in enumerate(children_raw)]
    return MindMapNode(label=label, children=children)


def _node_to_dict(node: MindMapNode) -> dict[str, Any]:
    return {
        "label": node.label,
        "children": [_node_to_dict(child) for child in node.children],
    }


def _render_node(node: MindMapNode, prefix: str, is_last: bool) -> list[str]:
    branch = "└── " if is_last else "├── "
    lines = [f"{prefix}{branch}{node.label}"]

    child_prefix = prefix + ("    " if is_last else "│   ")
    for index, child in enumerate(node.children):
        lines.extend(_render_node(child, child_prefix, is_last=index == len(node.children) - 1))

    return lines


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
        raise MindMapAgentError("Mind map agent JSON root must be an object")
    return payload
