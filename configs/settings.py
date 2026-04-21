import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Try .env first, then fall back to .env.example
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(_root / ".env.example", override=False)


class Settings:
    openrouter_api_key1: str = os.environ.get("OPENROUTER_API_KEY1", "")
    openrouter_api_key2: str = os.environ.get("OPENROUTER_API_KEY2", "")
    # Backward compatibility: if only OPENROUTER_API_KEY is provided,
    # use it as KEY1.
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_api_key1:
        openrouter_api_key1 = openrouter_api_key

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = os.environ.get("MODEL_NAME", "qwen/qwen3.6-plus")
    # Supports fallback chain via comma-separated models, e.g.:
    # "anthropic/claude-sonnet-4.6,qwen/qwen3.6-plus"
    extraction_model_name: str = os.environ.get(
        "EXTRACTION_MODEL_NAME",
        "anthropic/claude-sonnet-4.6,qwen/qwen3.6-plus",
    )
    # Comma-separated model patterns that should use OPENROUTER_API_KEY2.
    # Pattern supports exact match and prefix wildcard `*`, e.g.:
    # OPENROUTER_API_KEY2_MODEL_PATTERNS=anthropic/claude-sonnet-*,openai/gpt-4o
    openrouter_api_key2_model_patterns_raw: str = os.environ.get(
        "OPENROUTER_API_KEY2_MODEL_PATTERNS",
        "anthropic/claude-sonnet-*",
    )

    @property
    def openrouter_api_key2_model_patterns(self) -> List[str]:
        return [
            item.strip()
            for item in self.openrouter_api_key2_model_patterns_raw.split(",")
            if item.strip()
        ]

    # Model used for Stage 3 uncertain-paper review (stronger model).
    review_model_name: str = os.environ.get(
        "REVIEW_MODEL_NAME",
        "anthropic/claude-sonnet-4.6",
    )

    pubmed_api_key: str = os.environ.get("PUBMED_API_KEY", "")
    proxy_url: str = os.environ.get("PROXY_URL", "")

    # Per-key proxy: KEY1 defaults to no proxy (server native network),
    # KEY2 defaults to PROXY_URL (reverse tunnel to local environment).
    proxy_url_key1: str = os.environ.get("PROXY_URL_KEY1", "")
    proxy_url_key2: str = os.environ.get(
        "PROXY_URL_KEY2", os.environ.get("PROXY_URL", "")
    )


settings = Settings()
