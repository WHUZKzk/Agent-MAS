import os
from pathlib import Path
from dotenv import load_dotenv

# Try .env first, then fall back to .env.example
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(_root / ".env.example", override=False)


class Settings:
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "qwen/qwen3.6-plus"
    extraction_model_name: str = "openai/gpt-5.4"
    pubmed_api_key: str = os.environ.get("PUBMED_API_KEY", "")


settings = Settings()
