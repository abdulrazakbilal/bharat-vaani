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
    decision_system = (
        "You are the Format Decision Agent.\n"
        "Choose ONE suggested output format based on user preference, literacy level, and tone.\n"
        'Allowed formats: ["audio","short_summary","long_article","visual"]\n'
        "Return ONLY JSON with keys:\n"
        '- "suggested_format": one of the allowed formats\n'
        '- "reason": one short sentence\n'
        "No extra text."
    )
    decision_user = (
        "User profile:\n"
        f"- literacy_level: {user_profile.literacy_level}\n"
        f"- preferred_format: {user_profile.preferred_format}\n"
        f"Detected tone: {detected_tone}\n\n"
        "Adapted text (target language) for context:\n"
        f"{adapted_text}"
    )

    decision_out = client.json_call(
        system=decision_system,
        user=decision_user,
        max_tokens=200,
        temperature=0.1,
    )
    suggested_format = str(decision_out.get("suggested_format", "") or "")
    reason = str(decision_out.get("reason", "") or "")

    max_tokens_map = {
        "short_summary": 300,
        "long_article": 1000,
        "visual": 600,
        "audio": 400,
    }
    max_tokens = max_tokens_map.get(suggested_format, 600)

    generation_system = (
        "You are the Format Writer Agent.\n"
        "Rewrite the provided adapted_text into the required output format.\n"
        "Return ONLY JSON with keys:\n"
        '- "adapted_text": string\n'
        "No extra text."
    )

    if suggested_format == "short_summary":
        generation_user = (
            "Required format: short_summary\n"
            "Write EXACTLY 3-4 sentences.\n"
            "No more. This is a short summary format.\n\n"
            "Use the target language already present in the provided adapted_text.\n"
            "Preserve all real numbers and named entities from the provided adapted_text.\n\n"
            f"Input adapted_text:\n{adapted_text}"
        )
    elif suggested_format == "long_article":
        generation_user = (
            "Required format: long_article\n"
            "Write a FULL article of minimum 6-8 sentences across 2-3 paragraphs. "
            "Do NOT summarize. Expand every point with cultural context and explanation.\n\n"
            "Use the target language already present in the provided adapted_text.\n"
            "Preserve all real numbers and named entities from the provided adapted_text.\n\n"
            f"Input adapted_text:\n{adapted_text}"
        )
    elif suggested_format == "visual":
        generation_user = (
            "Required format: visual\n"
            "Return adapted_text as a JSON string (escaped) containing ONLY this object:\n"
            '{\n'
            '  "format": "visual",\n'
            '  "headline": "one punchy sentence summary in the target language",\n'
            '  "key_stats": [\n'
            '    { "label": "Net Profit", "value": "21,930 Cr", "context": "8% above estimates" }\n'
            "  ],\n"
            '  "sentiment": "positive/negative/neutral",\n'
            '  "one_line": "one sentence the common person needs to know, in target language"\n'
            "}\n\n"
            "Rules:\n"
            "- Extract real numbers from the provided adapted_text (profit figures, percentages, rupee amounts, years). Maximum 4 key_stats.\n"
            "- Use only numbers that appear in the provided adapted_text. Do not invent figures.\n"
            "- label must be short and human-friendly.\n"
            '- value must keep original units when present (Cr, %, Rs, years).\n'
            "- headline, one_line must be in the same language as adapted_text.\n"
            "- sentiment must be positive/negative/neutral.\n\n"
            "IMPORTANT: Because the overall response is JSON, escape any double quotes inside the adapted_text JSON string.\n\n"
            f"Input adapted_text:\n{adapted_text}"
        )
    elif suggested_format == "audio":
        generation_user = (
            "Required format: audio\n"
            "Write a narration script in the target language, suitable for text-to-speech.\n"
            "Keep it clear and natural to read aloud (roughly 4-6 sentences).\n"
            "Preserve all real numbers and named entities.\n\n"
            f"Input adapted_text:\n{adapted_text}"
        )
    else:
        generation_user = (
            "Required format: fallback\n"
            "Return adapted_text exactly equal to the provided input.\n\n"
            f"Input adapted_text:\n{adapted_text}"
        )

    generation_out = client.json_call(
        system=generation_system,
        user=generation_user,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    final_adapted_text = str(generation_out.get("adapted_text", "") or "")

    return {
        "suggested_format": suggested_format,
        "reason": reason,
        "adapted_text": final_adapted_text,
    }
