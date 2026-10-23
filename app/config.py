"""Settings loaded from environment variables (.env in development).

The API key is never hardcoded: it is read from GEMINI_API_KEY at startup.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    # Context management: once history exceeds MAX_MESSAGES, older messages
    # are folded into a running summary and only KEEP_RECENT are sent verbatim.
    max_messages: int = _int("MAX_MESSAGES", 12)
    keep_recent: int = _int("KEEP_RECENT", 6)
    # "summarize" (default) or "truncate"
    context_strategy: str = os.getenv("CONTEXT_STRATEGY", "summarize").lower()
    request_timeout_s: int = _int("REQUEST_TIMEOUT_S", 30)


settings = Settings()
