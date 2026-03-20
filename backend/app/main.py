from __future__ import annotations

import os
import base64
import sqlite3
from io import BytesIO
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import edge_tts

from .agents import (
    cultural_context_agent,
    emotion_tone_agent,
    format_decision_agent,
    persona_match_agent,
)
from .groq_client import GroqJSONClient
from .models import (
    AudioRequest,
    AudioResponse,
    FeedbackRequest,
    PersonaUpsertRequest,
    TransformRequest,
    TransformResponse,
)


def _db_path() -> str:
    # backend root = parent of app/
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(backend_root, "feedback.db")


def _init_db() -> None:
    conn = sqlite3.connect(_db_path())
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS personas (
                user_id TEXT PRIMARY KEY,
                language TEXT,
                region TEXT,
                literacy_level TEXT,
                preferred_format TEXT,
                transform_count INTEGER DEFAULT 0,
                last_updated DATETIME
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                reaction TEXT,
                timestamp DATETIME
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


load_dotenv()

app = FastAPI(title="Bharat Vaani Backend", version="0.1.0")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
allowed_origins = (
    ["*"]
    if allowed_origins_env == "*"
    else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def _startup() -> None:
    _init_db()


@app.post("/audio", response_model=AudioResponse)
async def audio(req: AudioRequest) -> AudioResponse:
    voice_map: dict[str, str] = {
        "hindi": "hi-IN-SwaraNeural",
        "telugu": "te-IN-ShrutiNeural",
        "tamil": "ta-IN-PallaviNeural",
        "bengali": "bn-IN-TanishaaNeural",
    }

    voice = voice_map.get(req.language.value)
    if not voice:
        raise HTTPException(status_code=400, detail="Voice not found for language.")

    buf = BytesIO()
    try:
        communicate = edge_tts.Communicate(text=req.text, voice=voice)
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.write(chunk["data"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")

    audio_bytes = buf.getvalue()
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="No audio data generated.")

    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    return AudioResponse(audio_base64=audio_base64, format="mp3")


@app.post("/transform", response_model=TransformResponse)
def transform(req: TransformRequest) -> TransformResponse:
    try:
        client = GroqJSONClient()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        tone_out = emotion_tone_agent(client, article_text=req.article_text)
        detected_tone = str(tone_out.get("detected_tone", "unknown"))

        cultural_out = cultural_context_agent(
            client, article_text=req.article_text, region=req.user_profile.region
        )
        analogies_used = cultural_out.get("analogies_used", []) or []
        cultural_notes = str(cultural_out.get("notes", "") or "")

        persona_out = persona_match_agent(
            client,
            article_text=req.article_text,
            target_language=req.target_language,
            user_profile=req.user_profile,
            detected_tone=detected_tone,
            analogies_used=analogies_used,
            cultural_notes=cultural_notes,
        )
        adapted_text = str(persona_out.get("adapted_text", "") or "")
        persona_score = float(persona_out.get("persona_score", 0.0) or 0.0)

        format_out = format_decision_agent(
            client,
            user_profile=req.user_profile,
            detected_tone=detected_tone,
            adapted_text=adapted_text,
        )
        suggested_format = str(format_out.get("suggested_format", "") or "")

        if not adapted_text:
            raise ValueError("Persona agent returned empty adapted_text.")
        if not suggested_format:
            raise ValueError("Format agent returned empty suggested_format.")

        return TransformResponse(
            adapted_text=adapted_text,
            detected_tone=detected_tone,
            analogies_used=analogies_used,
            suggested_format=suggested_format,
            persona_score=persona_score,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {e}")


@app.post("/persona")
def upsert_persona(req: PersonaUpsertRequest) -> dict[str, Any]:
    """
    Upsert user profile into the `personas` table.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(_db_path())
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO personas (
                user_id, language, region, literacy_level, preferred_format, transform_count, last_updated
            ) VALUES (?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                language=excluded.language,
                region=excluded.region,
                literacy_level=excluded.literacy_level,
                preferred_format=excluded.preferred_format,
                last_updated=excluded.last_updated
            """,
            (
                req.user_id,
                req.language.value,
                req.region,
                req.literacy_level,
                req.preferred_format,
                now,
            ),
        )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@app.post("/feedback")
def post_feedback(req: FeedbackRequest) -> dict[str, Any]:
    """
    Store feedback and increment persona transform_count.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(_db_path())
    try:
        cursor = conn.cursor()

        # Ensure persona row exists (with blank fields if user never called /persona).
        cursor.execute(
            """
            INSERT INTO personas (
                user_id, language, region, literacy_level, preferred_format, transform_count, last_updated
            ) VALUES (?, '', '', '', '', 0, ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (req.user_id, now),
        )

        cursor.execute(
            """
            INSERT INTO feedback (user_id, reaction, timestamp)
            VALUES (?, ?, ?)
            """,
            (req.user_id, req.reaction.value, now),
        )
        feedback_id = cursor.lastrowid

        cursor.execute(
            """
            UPDATE personas
            SET transform_count = transform_count + 1,
                last_updated = ?
            WHERE user_id = ?
            """,
            (now, req.user_id),
        )

        cursor.execute(
            "SELECT transform_count FROM personas WHERE user_id = ?",
            (req.user_id,),
        )
        row = cursor.fetchone()
        transform_count = int(row[0]) if row else 0

        conn.commit()
        return {
            "status": "ok",
            "feedback_id": feedback_id,
            "transform_count": transform_count,
        }
    finally:
        conn.close()

