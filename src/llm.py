"""The ONLY place in the project that talks to an LLM.

Everything else calls call_llm(). Later this file will point at the LLM gateway
and add Gemini / Ollama fallbacks - nothing else will need to change.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import APIConnectionError, APITimeoutError, Groq, InternalServerError, RateLimitError

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 2  # so at most 3 tries in total
RETRY_WAIT_SECONDS = 3
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "llm_calls.jsonl"

# Errors worth trying again (network problems, rate limit, server hiccups)
RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

_client = None  # created on first use


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from the environment
    return _client


def _log(entry: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def call_llm(prompt: str, system: str = "", json_mode: bool = False, step: str = "") -> str:
    """Send a prompt to the LLM and return its text answer.

    json_mode=True asks the model to reply with valid JSON (your prompt must also
    mention the word "JSON"). `step` is just a label for the log, e.g. "classify".
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.time()
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(**kwargs)
            break
        except RETRYABLE as e:
            if attempt == MAX_RETRIES:
                _log({"step": step, "model": MODEL, "seconds": round(time.time() - start, 2),
                      "status": "failed", "error": type(e).__name__, "retries": attempt})
                raise
            time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))

    usage = response.usage
    _log({
        "step": step,
        "model": MODEL,
        "seconds": round(time.time() - start, 2),
        "status": "ok",
        "retries": attempt,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    })
    return response.choices[0].message.content
