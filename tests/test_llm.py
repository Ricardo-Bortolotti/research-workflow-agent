"""Tests for Hugging Face LLM client."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.llm import DEFAULT_MODEL_ID, HuggingFaceLLM, LLMError


@pytest.fixture
def mock_chat_response() -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="Generated answer"))]
    return response


def test_default_model_id() -> None:
    assert DEFAULT_MODEL_ID == "meta-llama/Llama-3.1-8B-Instruct"


def test_missing_api_token_raises() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMError, match="Missing Hugging Face API token"):
            HuggingFaceLLM(api_token=None)


def test_generate_returns_model_text(mock_chat_response: MagicMock) -> None:
    with patch("app.llm.InferenceClient") as mock_client_cls:
        mock_client_cls.return_value.chat_completion.return_value = mock_chat_response
        llm = HuggingFaceLLM(
            model_id=DEFAULT_MODEL_ID,
            api_token="hf_test_token",
        )

        result = llm.generate("Summarize this document.")

    assert result == "Generated answer"
    mock_client_cls.return_value.chat_completion.assert_called_once()
    call_kwargs = mock_client_cls.return_value.chat_completion.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL_ID
    assert call_kwargs["messages"][-1]["content"] == "Summarize this document."


def test_generate_with_system_prompt(mock_chat_response: MagicMock) -> None:
    with patch("app.llm.InferenceClient") as mock_client_cls:
        mock_client_cls.return_value.chat_completion.return_value = mock_chat_response
        llm = HuggingFaceLLM(api_token="hf_test_token")

        llm.generate("Question?", system_prompt="You are helpful.")

    messages = mock_client_cls.return_value.chat_completion.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "Question?"}


def test_empty_prompt_raises(mock_chat_response: MagicMock) -> None:
    with patch("app.llm.InferenceClient"):
        llm = HuggingFaceLLM(api_token="hf_test_token")
        with pytest.raises(LLMError, match="Prompt text cannot be empty"):
            llm.generate("   ")


def test_provider_suffix_is_applied(mock_chat_response: MagicMock) -> None:
    with patch("app.llm.InferenceClient") as mock_client_cls:
        mock_client_cls.return_value.chat_completion.return_value = mock_chat_response
        llm = HuggingFaceLLM(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            api_token="hf_test",
            provider="groq",
        )
        assert llm.model_id == "meta-llama/Llama-3.1-8B-Instruct:groq"


def test_hf_token_alias_is_supported(mock_chat_response: MagicMock) -> None:
    with patch.dict(os.environ, {"HF_TOKEN": "hf_from_alias"}, clear=True):
        with patch("app.llm.InferenceClient") as mock_client_cls:
            mock_client_cls.return_value.chat_completion.return_value = mock_chat_response
            llm = HuggingFaceLLM()
            llm.generate("hello")

    mock_client_cls.assert_called_once_with(token="hf_from_alias")
