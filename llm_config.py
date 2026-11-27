"""Shared native chat-model setup; retrieval and agent logic stay in the notebooks."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import SecretStr

load_dotenv(Path(__file__).resolve().parent / ".env")

PROVIDERS = {
    "groq": ("groq", "openai/gpt-oss-20b"),
    "mistral": ("mistralai", "ministral-14b-latest"),
    "openrouter": ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "google": ("google_genai", "gemma-4-26b-a4b-it"),
}

# Local provider: Ollama runs the model on this computer, so there is no API key.
OLLAMA = ("ollama", "qwen3.5:4b")
OLLAMA_BASE_URL = "http://localhost:11434"


def create_chat_model(
    *, temperature: float = 0, max_tokens: int = 1024
) -> BaseChatModel:
    """Select only the requested provider and its own key, without API calls."""
    name = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if name == "ollama":
        return _create_ollama_model(temperature=temperature, max_tokens=max_tokens)
    if name not in PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER must be one of: {', '.join([*PROVIDERS, 'ollama'])}"
        )
    key = os.getenv(f"{name.upper()}_API_KEY", "").strip()
    if not key:
        raise ValueError(f"Set {name.upper()}_API_KEY in .env for LLM_PROVIDER={name}")
    adapter, default = PROVIDERS[name]
    model = os.getenv(f"{name.upper()}_MODEL", default)
    options: dict[str, Any] = {}
    if name == "openrouter":
        fallback = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/free").strip()
        options = {
            "model_kwargs": {
                "retries": None,
                "models": [model, fallback] if fallback else None,
            },
            "openrouter_provider": {"require_parameters": True},
        }
    elif name == "google":
        options = {"vertexai": False}
    return init_chat_model(
        model=model,
        model_provider=adapter,
        api_key=SecretStr(key),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30000 if name == "openrouter" else 30,
        max_retries=0,
        **options,
    )


def _create_ollama_model(*, temperature: float, max_tokens: int) -> BaseChatModel:
    """Select the local Ollama server: no key, and no data leaves this computer."""
    adapter, default = OLLAMA
    return init_chat_model(
        model=os.getenv("OLLAMA_MODEL", default),
        model_provider=adapter,
        base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
        temperature=temperature,
        # Ollama calls the output budget num_predict.
        num_predict=max_tokens,
        # Room for the instruction, the retrieved chunks and the question.
        num_ctx=8192,
        # Skip the hidden "thinking" text so the budget goes to the answer.
        reasoning=False,
        # A local model on shared hardware is slower than a cloud API.
        client_kwargs={"timeout": 300},
    )


def validate_response(message: BaseMessage) -> BaseMessage:
    """Reject incomplete or empty final answers before parsing away metadata."""
    metadata = message.response_metadata
    # Cloud adapters report finish_reason; the Ollama adapter calls it done_reason.
    reason = metadata.get("finish_reason") or metadata.get("done_reason")
    if str(reason).lower() != "stop":
        raise ValueError(f"Incomplete provider response: {reason}")
    if not message.text.strip():
        raise ValueError("The provider returned no answer text")
    return message


def response_text(message: BaseMessage) -> str:
    """Return only visible answer text, never reasoning or other content blocks."""
    return validate_response(message).text
