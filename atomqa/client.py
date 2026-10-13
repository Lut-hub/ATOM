"""OpenAI-compatible chat client configured through environment variables."""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client():
    from openai import OpenAI
    return OpenAI(timeout=120.0, max_retries=2)


def call_gpt(messages, model="o3-mini"):
    response = get_client().chat.completions.create(
        model=os.environ.get("ATOM_MODEL", model), messages=messages
    )
    return [choice.message.content or "" for choice in response.choices]
