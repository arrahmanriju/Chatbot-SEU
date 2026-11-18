"""Configuration and connection reuse for the Groq provider."""
import logging
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY to enable university answers.")
    return Groq(api_key=api_key, timeout=20.0, max_retries=1)


def ask_llm(prompt: str, model: str | None = None) -> str:
    return complete([{"role": "user", "content": prompt}], model)


def ask_llm_json(prompt: str) -> str:
    instructions, _, payload = prompt.partition("\n")
    return complete(
        [{"role": "system", "content": instructions}, {"role": "user", "content": payload}],
        response_format={"type": "json_object"},
    )


def complete(messages, model=None, **options) -> str:
    client = _get_client()
    started = perf_counter()
    response = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b",
        messages=messages,
        temperature=0.1,
        max_completion_tokens=900,
        **options,
    )
    logger.info("groq_generation elapsed_ms=%.1f", (perf_counter()-started)*1000)
    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise RuntimeError("Groq returned an empty answer.")
    return content.strip()
