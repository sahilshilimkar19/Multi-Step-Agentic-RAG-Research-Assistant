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
from langchain_core.callbacks import BaseCallbackHandler
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


class UsageCollector(BaseCallbackHandler):
    """Accumulates input/output token usage from LLM responses.

    Pass instances via `config={"callbacks": [collector]}` on .invoke()
    to capture usage even from .with_structured_output() runnables (where
    the AIMessage is consumed before the caller sees it).
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: Any, **_: Any) -> None:  # noqa: ANN401
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg else None
                if usage:
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)

    def as_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


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
