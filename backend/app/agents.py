from __future__ import annotations

import json
import re
from typing import Any

from .groq_client import GroqJSONClient
from .models import TargetLanguage, UserProfile


FALLBACK_TRANSFORM = {
    "adapted_text": "Groq is rate limited. Please wait 30 seconds and try again.",
    "detected_tone": "unknown",
    "analogies_used": [],
    "suggested_format": "short_summary",
    "persona_score": 0.0,
}


def _coerce_output(data: dict[str, Any]) -> dict[str, Any]:
    analogies = data.get("analogies_used", [])
    if not isinstance(analogies, list):
        analogies = []

    try:
        persona_score = float(data.get("persona_score", 0.0) or 0.0)
    except Exception:
        persona_score = 0.0

    return {
        "adapted_text": str(data.get("adapted_text", "") or ""),
        "detected_tone": str(data.get("detected_tone", "unknown") or "unknown"),
        "analogies_used": analogies,
        "suggested_format": str(
            data.get("suggested_format", "short_summary") or "short_summary"
        ),
        "persona_score": persona_score,
    }


def _parse_partial_response(text: str) -> dict[str, Any]:
    partial: dict[str, Any] = {}
    if not text:
        return partial

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _coerce_output(parsed)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return _coerce_output(parsed)
        except Exception:
            pass

    m = re.search(r'"detected_tone"\s*:\s*"([^"]+)"', text)
    if m:
        partial["detected_tone"] = m.group(1)
    m = re.search(r'"suggested_format"\s*:\s*"([^"]+)"', text)
    if m:
        partial["suggested_format"] = m.group(1)
    m = re.search(r'"persona_score"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        try:
            partial["persona_score"] = float(m.group(1))
        except Exception:
            pass
    m = re.search(r'"adapted_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        partial["adapted_text"] = bytes(m.group(1), "utf-8").decode("unicode_escape")

    return _coerce_output(partial)


def combined_transform_agent(
    client: GroqJSONClient,
    *,
    article_text: str,
    target_language: TargetLanguage,
    user_profile: UserProfile,
) -> dict[str, Any]:
    system = (
        "You are a combined 4-agent pipeline for Bharat Vaani.\n"
        "In ONE pass, do all tasks:\n"
        "1) Emotion & Tone detection\n"
        "2) Cultural context analogies for Indian audiences\n"
        "3) Persona adaptation (language, region, literacy, format preference)\n"
        "4) Format decision\n\n"
        'Allowed suggested_format: ["audio","short_summary","long_article","visual"]\n'
        "Return ONLY JSON with keys:\n"
        '- "adapted_text": string\n'
        '- "detected_tone": string\n'
        '- "analogies_used": list of {"original","replacement","rationale"}\n'
        '- "suggested_format": one of the allowed formats\n'
        '- "persona_score": number 0..1\n\n'
        "Formatting rules:\n"
        "- short_summary: Write EXACTLY 3-4 sentences. No more. This is a short summary format.\n"
        "- long_article: Write a FULL article of minimum 6-8 sentences across 2-3 paragraphs. Do NOT summarize. Expand every point with cultural context and explanation.\n"
        "- visual: adapted_text must be a JSON string with keys format/headline/key_stats/sentiment/one_line. Max 4 key_stats. Extract only real numbers from article.\n"
        "- audio: adapted_text should be narration-friendly, concise spoken script.\n"
        "Preserve factual numbers and named entities. Do not invent facts."
    )
    user = (
        "Input article (English):\n"
        f"{article_text}\n\n"
        "User profile:\n"
        f"- target_language: {target_language.value}\n"
        f"- region: {user_profile.region}\n"
        f"- literacy_level: {user_profile.literacy_level}\n"
        f"- preferred_format: {user_profile.preferred_format}"
    )

    try:
        raw = client.raw_call(system=system, user=user, max_tokens=1500, temperature=0.2)
        if not raw:
            return FALLBACK_TRANSFORM.copy()
        parsed = _parse_partial_response(raw)
        if not parsed.get("adapted_text"):
            merged = FALLBACK_TRANSFORM.copy()
            merged.update(parsed)
            return merged
        return parsed
    except Exception as exc:
        msg = str(exc).lower()
        if "rate" in msg or "429" in msg or "limit" in msg:
            return FALLBACK_TRANSFORM.copy()
        parsed = _parse_partial_response(str(exc))
        if parsed.get("adapted_text"):
            return parsed
        return FALLBACK_TRANSFORM.copy()
