"""Hugging Face Inference API client for text generation."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, HfHubHTTPError

load_dotenv()

logger = logging.getLogger(__name__)

# Widely available on HF Inference Providers router (chat completion).
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TOKEN_ENV_VARS = ("HUGGINGFACE_API_TOKEN", "HF_TOKEN")
MODEL_ENV_VAR = "HF_MODEL_ID"
PROVIDER_ENV_VAR = "HF_INFERENCE_PROVIDER"


class LLMError(Exception):
    """Raised when LLM inference fails."""


class HuggingFaceLLM:
    """Generate text using the Hugging Face Inference API (Inference Providers router).

    The API token is read from ``.env`` using ``HUGGINGFACE_API_TOKEN`` or ``HF_TOKEN``.

    Model routing:
    - ``HF_MODEL_ID`` — model id (default: Llama 3.1 8B Instruct)
    - ``HF_INFERENCE_PROVIDER`` — optional provider suffix, e.g. ``groq`` → ``model:groq``

    Enable providers at: https://huggingface.co/settings/inference-providers
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_token: str | None = None,
        provider: str | None = None,
    ) -> None:
        base_model = model_id or os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL_ID)
        self.provider = provider or os.getenv(PROVIDER_ENV_VAR)
        self.model_id = _build_router_model_id(base_model, self.provider)
        self.api_token = api_token or _resolve_api_token()
        self._client = InferenceClient(token=self.api_token)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Generate a completion for a user prompt.

        Args:
            prompt: User message or task instruction.
            system_prompt: Optional system instruction for the model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Generated text from the model.
        """
        if not prompt.strip():
            raise LLMError("Prompt text cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("Calling Hugging Face Inference API with model: %s", self.model_id)

        try:
            response = self._client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except BadRequestError as exc:
            logger.exception("Hugging Face API bad request")
            raise LLMError(_format_bad_request_error(exc, self.model_id)) from exc
        except HfHubHTTPError as exc:
            logger.exception("Hugging Face API request failed")
            raise LLMError(f"Hugging Face API request failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected LLM inference error")
            raise LLMError(f"LLM inference failed: {exc}") from exc

        content = _extract_message_content(response)
        if not content.strip():
            raise LLMError("Model returned an empty response")

        logger.info("Generated %d characters from model %s", len(content), self.model_id)
        return content


def _build_router_model_id(model_id: str, provider: str | None) -> str:
    """Build model id for the HF router (optionally with provider suffix)."""
    if ":" in model_id:
        return model_id
    if provider:
        return f"{model_id}:{provider.strip()}"
    return model_id


def _format_bad_request_error(exc: BadRequestError, model_id: str) -> str:
    message = str(exc)
    if "model_not_supported" in message or "not supported by any provider" in message:
        return (
            f"Model '{model_id}' is not available on your enabled Inference Providers. "
            "Enable providers at https://huggingface.co/settings/inference-providers "
            "or set HF_MODEL_ID to a supported model (e.g. meta-llama/Llama-3.1-8B-Instruct). "
            f"Original error: {exc}"
        )
    return f"Hugging Face API bad request: {exc}"


def _resolve_api_token() -> str:
    for env_var in TOKEN_ENV_VARS:
        token = os.getenv(env_var)
        if token:
            return token

    joined_vars = " or ".join(TOKEN_ENV_VARS)
    raise LLMError(
        f"Missing Hugging Face API token. Set {joined_vars} in your .env file."
    )


def _extract_message_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError("Unexpected response format from Hugging Face API") from exc
