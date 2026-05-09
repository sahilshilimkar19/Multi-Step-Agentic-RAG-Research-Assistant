"""Provider-agnostic LLM factory with transient-error retry.

`get_llm(model)` returns a `_RetryingChat` that wraps either ChatAnthropic
or ChatOpenAI. `.invoke` / `.ainvoke` are wrapped with tenacity exponential
backoff (3 attempts, 1s -> 8s) on transient provider errors:
RateLimitError, APITimeoutError, APIConnectionError.

Authentication failures, 4xx validation errors, and other non-transient
errors propagate immediately.
"""
from __future__ import annotations

from typing import Any

from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    APITimeoutError as OpenAIAPITimeoutError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from agentic_rag.config import get_settings

_TRANSIENT: tuple[type[BaseException], ...] = (
    AnthropicRateLimitError,
    AnthropicAPITimeoutError,
    AnthropicAPIConnectionError,
    OpenAIRateLimitError,
    OpenAIAPITimeoutError,
    OpenAIAPIConnectionError,
)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT)


_RETRY_KW = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


class _RetryingRunnable:
    """Wraps any Runnable; retries .invoke / .ainvoke on transient errors."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @retry(**_RETRY_KW)
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.invoke(*args, **kwargs)

    @retry(**_RETRY_KW)
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.ainvoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RetryingChat(_RetryingRunnable):
    """Chat-specific wrapper that re-wraps the result of with_structured_output."""

    def with_structured_output(self, *args: Any, **kwargs: Any) -> _RetryingRunnable:
        return _RetryingRunnable(self._inner.with_structured_output(*args, **kwargs))


def get_llm(model: str, temperature: float = 0.0) -> _RetryingChat:
    """Return a retry-wrapped ChatModel selected by model-name prefix."""
    settings = get_settings()
    inner: Any
    if model.startswith("claude"):
        inner = ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key or None,
            temperature=temperature,
        )
    else:
        inner = ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key or None,
            temperature=temperature,
        )
    return _RetryingChat(inner)
