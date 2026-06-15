"""Shared Hugging Face API token resolution."""

import os

TOKEN_ENV_VARS = ("HUGGINGFACE_API_TOKEN", "HF_TOKEN")


def resolve_hf_api_token() -> str:
    """Return the Hugging Face API token from environment variables."""
    for env_var in TOKEN_ENV_VARS:
        token = os.getenv(env_var)
        if token:
            return token

    joined_vars = " or ".join(TOKEN_ENV_VARS)
    raise ValueError(f"Missing Hugging Face API token. Set {joined_vars} in your .env file.")
