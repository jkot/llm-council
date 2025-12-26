"""LLM client for making requests via OpenRouter or direct APIs (LiteLLM)."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import (
    ModelConfig,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    XAI_API_KEY,
    get_model_display_name,
)


async def query_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying OpenRouter model {model}: {e}")
        return None


async def query_litellm(
    model: str,
    provider: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model directly via LiteLLM.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        provider: Provider name (e.g., "openai", "anthropic", "google", "xai")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    import litellm

    # Determine model string and API key based on provider
    if provider == "openai":
        api_key = OPENAI_API_KEY
        litellm_model = f"openai/{model}"
    elif provider == "anthropic":
        api_key = ANTHROPIC_API_KEY
        litellm_model = f"anthropic/{model}"
    elif provider == "google":
        api_key = GOOGLE_API_KEY
        litellm_model = f"gemini/{model}"
    elif provider == "xai":
        api_key = XAI_API_KEY
        litellm_model = f"xai/{model}"
    else:
        print(f"Unknown provider: {provider}")
        return None

    try:
        response = await litellm.acompletion(
            model=litellm_model,
            messages=messages,
            timeout=timeout,
            api_key=api_key,
        )

        message = response.choices[0].message

        return {
            'content': message.content,
            'reasoning_details': getattr(message, 'reasoning_details', None)
        }

    except Exception as e:
        print(f"Error querying LiteLLM model {provider}/{model}: {e}")
        return None


async def query_model(
    model_config: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model using the appropriate provider.

    Args:
        model_config: Model configuration with 'model' and 'provider' keys
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    provider = model_config['provider']
    model = model_config['model']

    if provider == "openrouter":
        return await query_openrouter(model, messages, timeout)
    else:
        return await query_litellm(model, provider, messages, timeout)


async def query_models_parallel(
    model_configs: List[ModelConfig],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        model_configs: List of model configurations
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model display name to response dict (or None if failed)
    """
    # Create tasks for all models
    tasks = [query_model(config, messages) for config in model_configs]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map model display names to their responses
    return {
        get_model_display_name(config): response
        for config, response in zip(model_configs, responses)
    }
