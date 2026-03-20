from __future__ import annotations

from typing import Any

from .groq_client import GroqJSONClient
from .models import TargetLanguage, UserProfile


def emotion_tone_agent(client: GroqJSONClient, *, article_text: str) -> dict[str, Any]:
    system = (
        "You are the Emotion & Tone Agent for a news-to-localization pipeline.\n"
        "Return ONLY a single JSON object with keys:\n"
        '- "detected_tone": short label (e.g., neutral, anxious, optimistic, critical)\n'
        '- "sentiment": one of ["negative","neutral","positive"]\n'
        '- "confidence": number 0..1\n'
        "No extra text."
    )
    user = f"Analyze the tone and sentiment of this English news article:\n\n{article_text}"
    return client.json_call(system=system, user=user)


def cultural_context_agent(
    client: GroqJSONClient, *, article_text: str, region: str
) -> dict[str, Any]:
    system = (
        "You are the Cultural Context Agent.\n"
        "Identify finance/economics concepts that could confuse a typical Indian audience and propose India-local analogies.\n"
        "Return ONLY JSON with keys:\n"
        '- "analogies_used": list of objects each with {"original","replacement","rationale"}\n'
        '- "notes": short string guidance for the next agent\n'
        "No extra text."
    )
    user = (
        f"Region focus: {region}\n\n"
        "Article:\n"
        f"{article_text}\n\n"
        "Find financial concepts needing Indian analogies."
    )
    return client.json_call(system=system, user=user)


def persona_match_agent(
    client: GroqJSONClient,
    *,
    article_text: str,
    target_language: TargetLanguage,
    user_profile: UserProfile,
    detected_tone: str,
    analogies_used: list[dict[str, Any]],
    cultural_notes: str,
) -> dict[str, Any]:
    system = (
        "You are the Persona Match Agent.\n"
        "Task: adapt the article for the user's profile, apply suggested analogies, and translate into the target language.\n"
        "Be factual. Preserve numbers and named entities. Avoid adding new claims.\n"
        "Return ONLY JSON with keys:\n"
        '- "adapted_text": the culturally adapted translation\n'
        '- "persona_score": number 0..1 where 1 means perfect match to user profile\n'
        '- "style_notes": short string describing choices made\n'
        "No extra text."
    )
    user = (
        "User profile:\n"
        f"- region: {user_profile.region}\n"
        f"- literacy_level: {user_profile.literacy_level}\n"
        f"- preferred_format: {user_profile.preferred_format}\n"
        f"Target language: {target_language.value}\n"
        f"Detected tone: {detected_tone}\n\n"
        f"Cultural notes: {cultural_notes}\n\n"
        f"Analogies to apply (when appropriate): {analogies_used}\n\n"
        "Article (English):\n"
        f"{article_text}"
    )
    return client.json_call(system=system, user=user)


def format_decision_agent(
    client: GroqJSONClient,
    *,
    user_profile: UserProfile,
    detected_tone: str,
    adapted_text: str,
) -> dict[str, Any]:
    system = (
        "You are the Format Decision Agent.\n"
        "Choose ONE suggested output format based on user preference, literacy level, and tone.\n"
        'Allowed formats: ["audio","short_summary","long_article","visual"]\n'
        "Return ONLY JSON with keys:\n"
        '- "suggested_format": one of the allowed formats\n'
        '- "reason": one short sentence\n'
        "No extra text."
    )
    user = (
        "User profile:\n"
        f"- literacy_level: {user_profile.literacy_level}\n"
        f"- preferred_format: {user_profile.preferred_format}\n"
        f"Detected tone: {detected_tone}\n\n"
        "Adapted text (target language):\n"
        f"{adapted_text}\n\n"
        "Decide the best output format."
    )
    return client.json_call(system=system, user=user)
