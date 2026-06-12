"""Tests for the mind map agent."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agents.mindmap_agent import MindMapAgent, MindMapAgentError, MindMapNode, MindMapResult


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "title": "Transformer Architecture",
            "nodes": [
                {
                    "label": "Self-Attention",
                    "children": [
                        {"label": "Scaled Dot-Product", "children": []},
                        {"label": "Multi-Head Attention", "children": []},
                    ],
                },
                {"label": "Encoder-Decoder", "children": []},
            ],
        }
    )
    return llm


@pytest.fixture
def agent(mock_llm: MagicMock) -> MindMapAgent:
    return MindMapAgent(llm=mock_llm)


def test_run_returns_structured_mindmap(agent: MindMapAgent) -> None:
    context = [Document(page_content="The Transformer uses self-attention.", metadata={"page": 1})]
    result = agent.run("Map Transformer concepts", context)

    assert isinstance(result, MindMapResult)
    assert result.title == "Transformer Architecture"
    assert len(result.nodes) == 2
    assert result.nodes[0].label == "Self-Attention"
    assert len(result.nodes[0].children) == 2


def test_to_text_renders_tree(agent: MindMapAgent) -> None:
    context = [Document(page_content="Attention.", metadata={})]
    result = agent.run("Mind map", context)

    text = result.to_text()
    assert "Transformer Architecture" in text
    assert "Self-Attention" in text
    assert "├──" in text or "└──" in text


def test_to_json_serializes_nodes(agent: MindMapAgent) -> None:
    context = [Document(page_content="Attention.", metadata={})]
    result = agent.run("Mind map", context)

    payload = json.loads(result.to_json())
    assert payload["title"] == "Transformer Architecture"
    assert payload["nodes"][0]["children"][0]["label"] == "Scaled Dot-Product"


def test_run_requires_question(agent: MindMapAgent) -> None:
    context = [Document(page_content="text", metadata={})]
    with pytest.raises(ValueError, match="Question text cannot be empty"):
        agent.run("  ", context)


def test_run_requires_context(agent: MindMapAgent) -> None:
    with pytest.raises(ValueError, match="At least one context document"):
        agent.run("valid question", [])


def test_invalid_json_raises(agent: MindMapAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = "not json"
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(MindMapAgentError, match="Invalid JSON"):
        agent.run("question", context)


def test_missing_title_raises(agent: MindMapAgent, mock_llm: MagicMock) -> None:
    mock_llm.generate.return_value = json.dumps({"nodes": [{"label": "A", "children": []}]})
    context = [Document(page_content="text", metadata={})]

    with pytest.raises(MindMapAgentError, match="title"):
        agent.run("question", context)


def test_mindmap_node_dataclass() -> None:
    node = MindMapNode(label="Root", children=[MindMapNode(label="Child")])
    assert node.children[0].label == "Child"
