"""
LLM tool layer for AutoSR.

Adapted from TrialMind's llm.py + llm_utils/openai.py + llm_utils/openai_async.py.
Key change: OpenAI → OpenRouter (api_key + base_url), single model qwen/qwen3.6-plus.
API surface is identical to TrialMind so the rest of the codebase stays familiar.
"""

import asyncio
import json
import re
import httpx
import tenacity
from openai import OpenAI, AsyncOpenAI
from typing import List, Optional
import logging

from configs.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_client = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    http_client=httpx.Client(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    ),
)

_async_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    ),
)

_MODEL = settings.model_name


# ---------------------------------------------------------------------------
# Low-level API calls (sync + async) with retry
# ---------------------------------------------------------------------------

@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=20),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)
def _call_sync(messages: list, temperature: float = 0.0, **kwargs):
    return _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=20),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)
async def _call_async(messages: list, temperature: float = 0.0, **kwargs):
    return await _async_client.chat.completions.create(
        model=_MODEL,
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


async def _gather_text(messages_list: list, temperature: float = 0.0) -> list:
    tasks = [_call_async(msgs, temperature=temperature) for msgs in messages_list]
    return await asyncio.gather(*tasks)


async def _gather_tools(messages_list: list, tools: list, temperature: float = 0.0) -> list:
    tasks = [
        _call_async(msgs, tools=tools, temperature=temperature)
        for msgs in messages_list
    ]
    return await asyncio.gather(*tasks)


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
) -> str:
    """Single synchronous LLM call, returns raw text."""
    messages = _prompt_to_messages(prompt_template, inputs)
    response = _call_sync(messages, temperature=temperature)
    return _clean_content(response.choices[0].message.content)


def batch_call_llm(
    prompt_template: str,
    batch_inputs: list,
    temperature: float = 0.0,
    batch_size: Optional[int] = None,
) -> List[str]:
    """Parallel async LLM calls on a batch of inputs, returns list of text responses."""
    all_messages = [_prompt_to_messages(prompt_template, inp) for inp in batch_inputs]

    if batch_size:
        results = []
        for i in range(0, len(all_messages), batch_size):
            chunk = all_messages[i : i + batch_size]
            raw = _run_async(_gather_text(chunk, temperature=temperature))
            results.extend(raw)
    else:
        results = _run_async(_gather_text(all_messages, temperature=temperature))

    return [_clean_content(r.choices[0].message.content) for r in results]


def batch_function_call_llm(
    prompt_template: str,
    batch_inputs: list,
    tool: dict,
    temperature: float = 0.0,
    batch_size: Optional[int] = None,
) -> List[dict]:
    """
    Parallel async LLM calls with function calling.
    Returns list of parsed dicts (tool_calls arguments).
    Falls back to {} on parse failure.
    """
    tools = [tool]
    all_messages = [_prompt_to_messages(prompt_template, inp) for inp in batch_inputs]

    if batch_size:
        results = []
        for i in range(0, len(all_messages), batch_size):
            chunk = all_messages[i : i + batch_size]
            raw = _run_async(_gather_tools(chunk, tools=tools, temperature=temperature))
            results.extend(raw)
    else:
        results = _run_async(_gather_tools(all_messages, tools=tools, temperature=temperature))

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
