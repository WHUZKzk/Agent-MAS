"""
Agent system — BaseAgent, ExecutorAgent, ReviewerAdjudicatorAgent.

Spec: docs/03_CORE_ENGINE_SPEC.md §3

All agents use the OpenAI SDK (with custom base_url for OpenRouter).
Anthropic and Google providers are also supported via separate branches.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Literal, Optional

if TYPE_CHECKING:
    from .context_manager import MountedContext
    from .model_registry import ModelConfig

logger = logging.getLogger("autosr.agents")


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    def __init__(self, model_id: str, model_config: "ModelConfig") -> None:
        self.model_id = model_id
        self.model_config = model_config

    @abstractmethod
    def call(self, context: "MountedContext") -> str:
        """Send mounted context to LLM. Returns raw string output."""

    def _call_llm(
        self,
        messages: List[dict],
        model_config: "ModelConfig",
        temperature: float,
        response_format: str = "json",
    ) -> str:
        """
        Route to the correct provider SDK.

        - "openrouter" / "openai": openai SDK with custom base_url.
        - "anthropic": anthropic SDK Messages API.
        - "google": google-generativeai SDK.

        Handles HTTP 429 rate-limit with exponential backoff (3 retries).
        Logs: model_id, input_tokens, output_tokens, latency.
        """
        provider = model_config.provider

        if provider in ("openai", "openrouter"):
            return self._call_openai_compat(messages, model_config, temperature,
                                            response_format)
        elif provider == "anthropic":
            return self._call_anthropic(messages, model_config, temperature)
        elif provider == "google":
            return self._call_google(messages, model_config, temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # OpenAI / OpenRouter
    # ------------------------------------------------------------------

    def _call_openai_compat(
        self,
        messages: List[dict],
        model_config: "ModelConfig",
        temperature: float,
        response_format: str,
    ) -> str:
        import os
        from openai import OpenAI, RateLimitError

        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=model_config.api_base,
        )

        kwargs: dict = {
            "model": model_config.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            t0 = time.monotonic()
            try:
                resp = client.chat.completions.create(**kwargs)
                latency = time.monotonic() - t0
                usage = resp.usage
                logger.info(
                    "[LLM] model=%s in=%d out=%d latency=%.2fs",
                    model_config.model_id,
                    usage.prompt_tokens if usage else -1,
                    usage.completion_tokens if usage else -1,
                    latency,
                )
                content = resp.choices[0].message.content
                return content or ""
            except RateLimitError:
                if attempt == max_attempts:
                    raise
                wait = 2 ** attempt
                logger.warning("[LLM] Rate limited. Retrying in %ds (attempt %d)…",
                               wait, attempt)
                time.sleep(wait)

        return ""  # unreachable

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _call_anthropic(
        self,
        messages: List[dict],
        model_config: "ModelConfig",
        temperature: float,
    ) -> str:
        import os
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            base_url=model_config.api_base,
        )
        # Anthropic uses separate system message
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        t0 = time.monotonic()
        resp = client.messages.create(
            model=model_config.model_id,
            max_tokens=4096,
            system=system,
            messages=filtered,
            temperature=temperature,
        )
        latency = time.monotonic() - t0
        logger.info("[LLM] model=%s in=%d out=%d latency=%.2fs",
                    model_config.model_id,
                    resp.usage.input_tokens, resp.usage.output_tokens, latency)
        return resp.content[0].text if resp.content else ""

    # ------------------------------------------------------------------
    # Google
    # ------------------------------------------------------------------

    def _call_google(
        self,
        messages: List[dict],
        model_config: "ModelConfig",
        temperature: float,
    ) -> str:
        import os
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        model = genai.GenerativeModel(model_config.model_id)

        # Convert OpenAI-style messages to Gemini format
        parts = "\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        t0 = time.monotonic()
        resp = model.generate_content(
            parts,
            generation_config={"temperature": temperature},
        )
        latency = time.monotonic() - t0
        logger.info("[LLM] model=%s latency=%.2fs", model_config.model_id, latency)
        return resp.text


# ---------------------------------------------------------------------------
# ExecutorAgent
# ---------------------------------------------------------------------------

class ExecutorAgent(BaseAgent):
    """
    For deterministic extraction tasks. Temperature = 0.0.
    Used by: Search (PICO generation, Pearl Growing), Extraction (all Soft Nodes).
    """
    temperature: float = 0.0

    def call(self, context: "MountedContext") -> str:
        return self._call_llm(
            messages=[
                {"role": "system", "content": context.system_message},
                {"role": "user",   "content": context.user_message},
            ],
            model_config=self.model_config,
            temperature=0.0,
            response_format=context.response_format,
        )


# ---------------------------------------------------------------------------
# ReviewerAdjudicatorAgent
# ---------------------------------------------------------------------------

class ReviewerAdjudicatorAgent(BaseAgent):
    """
    For screening review and adjudication.

    - role="reviewer": standard screening, temperature=0.0.
      Heterogeneity comes from different base models (not temperature).
    - role="adjudicator": blinded CoT adjudication, temperature=0.0.
    """

    def __init__(
        self,
        model_id: str,
        model_config: "ModelConfig",
        role: Literal["reviewer", "adjudicator"],
    ) -> None:
        super().__init__(model_id, model_config)
        self.role = role

    def call(self, context: "MountedContext") -> str:
        return self._call_llm(
            messages=[
                {"role": "system", "content": context.system_message},
                {"role": "user",   "content": context.user_message},
            ],
            model_config=self.model_config,
            temperature=0.0,
            response_format=context.response_format,
        )
