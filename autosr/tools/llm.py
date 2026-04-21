"""
LLM tool layer for AutoSR.

Adapted from TrialMind's llm.py + llm_utils/openai.py + llm_utils/openai_async.py.
Key change: OpenAI → OpenRouter with model-based API key routing.
API surface remains identical to TrialMind so the rest of the codebase stays familiar.
"""

import asyncio
import json
import re
import httpx
import tenacity
from openai import OpenAI, AsyncOpenAI
from typing import List, Optional
import logging
from functools import lru_cache

from configs.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_MODEL = settings.model_name
_WARNED_KEY2_MISSING = False


def _model_matches_pattern(model_name: str, pattern: str) -> bool:
    model = (model_name or "").strip().lower()
    pat = (pattern or "").strip().lower()
    if not model or not pat:
        return False
    if pat.endswith("*"):
        return model.startswith(pat[:-1])
    return model == pat


def _should_use_key2(model_name: str) -> bool:
    return any(
        _model_matches_pattern(model_name, pattern)
        for pattern in settings.openrouter_api_key2_model_patterns
    )


def get_openrouter_key_alias_for_model(model: Optional[str] = None) -> str:
    model_name = model or _MODEL
    if _should_use_key2(model_name):
        if settings.openrouter_api_key2:
            return "key2"
        return "key1"
    return "key1"


def get_openrouter_api_key_for_model(model: Optional[str] = None) -> str:
    alias = get_openrouter_key_alias_for_model(model=model)
    if alias == "key2":
        return settings.openrouter_api_key2
    return settings.openrouter_api_key1


def _get_proxy_for_alias(alias: str) -> Optional[str]:
    """Return the proxy URL for a given key alias, or None for direct connection."""
    if alias == "key2":
        return settings.proxy_url_key2 or None
    return settings.proxy_url_key1 or None


@lru_cache(maxsize=4)
def _get_sync_client(api_key: str, proxy: Optional[str] = None) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        http_client=httpx.Client(
            proxy=proxy,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        ),
    )


@lru_cache(maxsize=4)
def _get_async_client(api_key: str, proxy: Optional[str] = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        http_client=httpx.AsyncClient(
            proxy=proxy,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        ),
    )


def _resolve_clients(model: Optional[str] = None) -> tuple[OpenAI, AsyncOpenAI]:
    global _WARNED_KEY2_MISSING

    model_name = model or _MODEL
    alias = get_openrouter_key_alias_for_model(model_name)
    api_key = get_openrouter_api_key_for_model(model_name)

    if not api_key:
        raise ValueError(
            f"OpenRouter API key for {alias} is empty. "
            "Please configure OPENROUTER_API_KEY1/OPENROUTER_API_KEY2."
        )

    if _should_use_key2(model_name) and alias == "key1" and not _WARNED_KEY2_MISSING:
        logger.warning(
            "Model '%s' matched OPENROUTER_API_KEY2_MODEL_PATTERNS, "
            "but OPENROUTER_API_KEY2 is empty. Falling back to KEY1.",
            model_name,
        )
        _WARNED_KEY2_MISSING = True

    proxy = _get_proxy_for_alias(alias)
    logger.debug("Resolved client: alias=%s  proxy=%s", alias, proxy or "(direct)")
    return _get_sync_client(api_key, proxy), _get_async_client(api_key, proxy)


# ---------------------------------------------------------------------------
# Low-level API calls (sync + async) with retry
# ---------------------------------------------------------------------------

@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=20),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)
def _call_sync(messages: list, temperature: float = 0.0, model: str = None, **kwargs):
    model_name = model or _MODEL
    sync_client, _ = _resolve_clients(model_name)
    return sync_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=20),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)
async def _call_async(messages: list, temperature: float = 0.0, model: str = None, **kwargs):
    model_name = model or _MODEL
    _, async_client = _resolve_clients(model_name)
    return await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_to_messages(prompt_template: str, inputs: dict) -> list:
    content = prompt_template.format(**inputs)
    return [{"role": "user", "content": content}]


def _clean_content(text: str) -> str:
    """Strip <think>…</think> blocks that Qwen3 sometimes emits."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


async def _gather_text(messages_list: list, temperature: float = 0.0, max_concurrency: int = 50, model: str = None) -> list:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call_with_sem(msgs):
        async with semaphore:
            return await _call_async(msgs, temperature=temperature, model=model)

    return await asyncio.gather(*[_call_with_sem(msgs) for msgs in messages_list])


async def _gather_tools(messages_list: list, tools: list, temperature: float = 0.0, max_concurrency: int = 50, model: str = None) -> list:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call_with_sem(msgs):
        async with semaphore:
            return await _call_async(msgs, tools=tools, temperature=temperature, model=model)

    return await asyncio.gather(*[_call_with_sem(msgs) for msgs in messages_list])


def _run_async(coro):
    """Run an async coroutine from sync context (handles already-running loop)."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        asyncio.get_running_loop()
        # We're inside a running loop (e.g. Jupyter / uvicorn)
        with ThreadPoolExecutor(1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Public API  (mirrors TrialMind's call_llm / batch_call_llm / batch_function_call_llm)
# ---------------------------------------------------------------------------

def call_llm(
    prompt_template: str,
    inputs: dict,
    temperature: float = 0.0,
    model: str = None,
) -> str:
    """Single synchronous LLM call, returns raw text."""
    messages = _prompt_to_messages(prompt_template, inputs)
    response = _call_sync(messages, temperature=temperature, model=model)
    return _clean_content(response.choices[0].message.content)


def batch_call_llm(
    prompt_template: str,
    batch_inputs: list,
    temperature: float = 0.0,
    max_concurrency: int = 50,
    model: str = None,
) -> List[str]:
    """
    Parallel async LLM calls on a batch of inputs, returns list of text responses.
    max_concurrency limits how many requests are in-flight at once (Semaphore-based).
    """
    all_messages = [_prompt_to_messages(prompt_template, inp) for inp in batch_inputs]
    results = _run_async(_gather_text(all_messages, temperature=temperature, max_concurrency=max_concurrency, model=model))
    return [_clean_content(r.choices[0].message.content) for r in results]


def batch_function_call_llm(
    prompt_template: str,
    batch_inputs: list,
    tool: dict,
    temperature: float = 0.0,
    max_concurrency: int = 50,
    model: str = None,
) -> List[dict]:
    """
    Parallel async LLM calls with function calling.
    max_concurrency limits how many requests are in-flight at once (Semaphore-based).
    Returns list of parsed dicts (tool_calls arguments). Falls back to {} on parse failure.
    """
    tools = [tool]
    all_messages = [_prompt_to_messages(prompt_template, inp) for inp in batch_inputs]
    results = _run_async(_gather_tools(all_messages, tools=tools, temperature=temperature, max_concurrency=max_concurrency, model=model))

    parsed = []
    for r in results:
        try:
            tool_calls = r.choices[0].message.tool_calls
            if tool_calls:
                parsed.append(json.loads(tool_calls[0].function.arguments))
            else:
                # Fallback: try to parse JSON from content
                content = _clean_content(r.choices[0].message.content)
                parsed.append(json.loads(content))
        except Exception:
            parsed.append({})
    return parsed
