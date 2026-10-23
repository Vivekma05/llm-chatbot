"""Thin wrapper around the Google Gemini SDK.

All provider-specific errors are converted into LLMError with a message that
is safe to show to the user, so the rest of the app never sees raw SDK errors.
"""
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .config import settings


class LLMError(Exception):
    """Raised when the LLM call fails. `user_message` is shown in the UI."""

    def __init__(self, user_message: str, status_code: int = 502):
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


class LLMClient(Protocol):
    def generate(self, system_prompt: str, messages: list[dict]) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str = settings.api_key, model: str = settings.model):
        if not api_key:
            raise LLMError(
                "Server is not configured: GEMINI_API_KEY is missing. "
                "Copy .env.example to .env and add your key.",
                status_code=500,
            )
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=settings.request_timeout_s * 1000,
                # Retry brief overloads ("503 high demand") and timeouts with
                # exponential backoff before giving up. 429 is not retried:
                # free-tier limits are per minute, so the user is told to wait.
                retry_options=types.HttpRetryOptions(
                    attempts=3, max_delay=8, http_status_codes=[500, 502, 503, 504]
                ),
            ),
        )

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}, ...]"""
        contents = [
            types.Content(
                # Gemini calls the assistant role "model"
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                ),
            )
        except genai_errors.ClientError as e:  # 4xx
            if e.code == 429:
                raise LLMError("The AI service is rate-limited right now. Please wait a minute and try again.", 429)
            if e.code in (401, 403) or "API key" in str(e):
                raise LLMError("The AI service rejected the API key. Check GEMINI_API_KEY in .env.", 500)
            if e.code == 404:  # model retired or misspelled
                raise LLMError(f"The model '{self.model}' is not available. Set GEMINI_MODEL in .env.", 500)
            raise LLMError("The AI service could not process that request.", 502)
        except genai_errors.ServerError:  # 5xx
            raise LLMError("The AI service is having problems. Please try again shortly.", 503)
        except Exception as e:  # network errors, timeouts
            name = type(e).__name__.lower()
            if "timeout" in name or "timeout" in str(e).lower():
                raise LLMError("The AI service took too long to respond. Please try again.", 504)
            raise LLMError("Could not reach the AI service. Check your internet connection.", 503)

        text = (response.text or "").strip() if response else ""
        if not text:
            # e.g. blocked by safety filters or an empty candidate
            raise LLMError("The AI returned an empty response. Try rephrasing your message.", 502)
        return text
