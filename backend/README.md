# Bharat Vaani (Backend)

FastAPI backend for the **Bharat Vaani** AI transformation pipeline.

## Setup

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:

- Copy `.env.example` to `.env`
- Set `GROQ_API_KEY`

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET /health`
- `POST /audio`
- `POST /persona`
- `POST /feedback`
- `POST /transform`

Example payload:

```json
{
  "article_text": "Your English news article...",
  "target_language": "hindi",
  "user_profile": {
    "region": "Maharashtra",
    "literacy_level": "basic",
    "preferred_format": "audio"
  }
}
```

Audio endpoint payload:

```json
{
  "text": "नमस्ते",
  "language": "hindi"
}
```

Persona endpoint payload:

```json
{
  "user_id": "user123",
  "language": "hindi",
  "region": "Maharashtra",
  "literacy_level": "basic",
  "preferred_format": "audio"
}
```

Feedback endpoint payload:

```json
{
  "user_id": "user123",
  "reaction": "liked"
}
```

