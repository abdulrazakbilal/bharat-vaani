from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq


DEFAULT_MODEL = "llama-3.1-8b-instant"


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object from model output.
    Keeps the API robust even if the model wraps JSON in prose.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found in response.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        # Not in JSON string
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("No complete JSON object found in response.")


class GroqJSONClient:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Set it in your environment or .env file."
            )
        self._client = Groq(api_key=api_key)

    def json_call(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        prompt = f"{system}\n\n{user}".strip()
        response = self._client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = (response.choices[0].message.content or "").strip()
        raw_json = _extract_first_json_object(result)
        return json.loads(raw_json)

    def raw_call(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        prompt = f"{system}\n\n{user}".strip()
        response = self._client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

