"""
Lightweight Anthropic (and optional OpenAI) wrapper for the Emergence project.
Adapted from /home/hj2742/Paper/utils/api_requests.py.
"""

import os
import time
import logging
from typing import Optional

import anthropic


def call_claude(
    prompt: str,
    model: str = "claude-3-7-sonnet-20250219",
    system: str = "You are a careful literary and data analyst.",
    max_tokens: int = 512,
    temperature: float = 0.3,
    max_retries: int = 8,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Send a single user prompt to a Claude model and return the text response.

    Parameters
    ----------
    prompt      : User message string.
    model       : Anthropic model ID.
    system      : System prompt.
    max_tokens  : Maximum tokens in the completion.
    temperature : Sampling temperature (lower = more deterministic).
    max_retries : Number of retry attempts on transient errors.
    logger      : Optional logger; falls back to print.

    Returns
    -------
    str : The assistant's response text, or "" on failure.
    """
    log = logger.info if logger else print

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    config = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(max_retries):
        try:
            response = client.messages.create(**config)
            text = response.content[0].text
            return text
        except anthropic.NotFoundError as e:
            log(f"[api] Model not found: '{model}'. Pass a valid --model <id> for your API key.")
            raise  # model name is wrong — no point retrying
        except anthropic.RateLimitError:
            wait = 10 * (attempt + 1)
            log(f"[api] Rate limit hit, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except anthropic.APIConnectionError:
            wait = 5 * (attempt + 1)
            log(f"[api] Connection error, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except Exception as e:
            log(f"[api] Unexpected error: {e} (attempt {attempt+1}/{max_retries})")
            time.sleep(5)

    log("[api] All retries exhausted, returning empty string.")
    return ""
