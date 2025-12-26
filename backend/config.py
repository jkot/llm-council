"""Configuration for the LLM Council."""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

# Provider types
Provider = Literal["openai", "anthropic", "google", "xai", "openrouter"]


class ModelConfig(TypedDict):
    model: str
    provider: Provider


# API Keys - OpenRouter (for backward compatibility)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# API Keys - Direct provider access via LiteLLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini
XAI_API_KEY = os.getenv("XAI_API_KEY")  # For Grok

# Council members - list of model configurations
# Each model specifies its provider for routing
COUNCIL_MODELS: list[ModelConfig] = [
    {"model": "gpt-5-mini", "provider": "openai"},
    {"model": "gemini-3-flash-preview", "provider": "google"},
    {"model": "claude-haiku-4-5", "provider": "anthropic"},
    {"model": "grok-4-1-fast-reasoning", "provider": "xai"},
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL: ModelConfig = {"model": "gpt-5.2-chat-latest", "provider": "openai"}

# Data directory for conversation storage
DATA_DIR = "data/conversations"


def get_model_display_name(model_config: ModelConfig) -> str:
    """Get a display name for a model config."""
    return f"{model_config['provider']}/{model_config['model']}"
