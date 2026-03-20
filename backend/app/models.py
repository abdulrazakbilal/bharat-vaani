from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TargetLanguage(str, Enum):
    hindi = "hindi"
    telugu = "telugu"
    tamil = "tamil"
    bengali = "bengali"


LiteracyLevel = Literal["basic", "intermediate", "expert"]


class UserProfile(BaseModel):
    region: str = Field(..., min_length=1, max_length=80)
    literacy_level: LiteracyLevel
    preferred_format: str = Field(..., min_length=1, max_length=40)


class TransformRequest(BaseModel):
    article_text: str = Field(..., min_length=1, max_length=200_000)
    target_language: TargetLanguage
    user_profile: UserProfile


class TransformResponse(BaseModel):
    adapted_text: str
    detected_tone: str
    analogies_used: list[dict[str, Any]] = Field(default_factory=list)
    suggested_format: str
    persona_score: float


class AudioLanguage(str, Enum):
    hindi = "hindi"
    telugu = "telugu"
    tamil = "tamil"
    bengali = "bengali"


class AudioRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    language: AudioLanguage


class AudioResponse(BaseModel):
    audio_base64: str
    format: str = "mp3"


class Reaction(str, Enum):
    liked = "liked"
    disliked = "disliked"
    replayed_audio = "replayed_audio"


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    reaction: Reaction


class PersonaUpsertRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    language: TargetLanguage
    region: str = Field(..., min_length=1, max_length=80)
    literacy_level: LiteracyLevel
    preferred_format: str = Field(..., min_length=1, max_length=40)
