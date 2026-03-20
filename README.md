# Bharat Vaani

The news that thinks like you

## Project Name

**Bharat Vaani** — *The news that thinks like you*

## Problem Statement

700M+ Indians consume news in regional languages but get either no coverage or poor literal translations with no cultural context.

## Solution

A **4-agent AI pipeline** that transforms ET English articles into culturally adapted regional language content with desi analogies.

## Features

- Cultural adaptation in Hindi, Telugu, Tamil, Bengali
- Desi Analogy Engine replacing Western concepts with local references
- Audio TTS output in regional languages
- Side by side comparison mode
- Persona learning from user feedback
- Format intelligence: Audio, Short Summary, Long Article, Visual

## Tech Stack

FastAPI, Next.js, Groq (llama-3.1-8b-instant), edge-tts, SQLite, Tailwind CSS

## How to Run

### Backend Setup (FastAPI)

From the repo root:

1. Install backend deps

   ```bash
   cd backend
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment variables

   ```bash
   copy .env.example .env
   ```

   Set `GROQ_API_KEY` in `.env` (and `ALLOWED_ORIGINS` if needed).

3. Start the server

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup (Next.js)

From the repo root:

1. Install frontend deps

   ```bash
   cd frontend
   npm install
   ```

2. Start the dev server

   ```bash
   npm run dev
   ```

3. Open `http://localhost:3000`

### Notes

- The frontend calls the backend at `http://localhost:8000` (as wired in the demo UI).

## Architecture (4 Agents)

The backend’s `/transform` pipeline runs four agents in sequence:

1. **Emotion & Tone Agent**: detects the article’s tone/sentiment to guide how the adaptation should read.
2. **Cultural Context Agent**: identifies confusing concepts and generates India-local analogies (Desi Analogy Engine).
3. **Persona Match Agent**: adapts and translates the article into the selected regional language, tuned to the user’s profile (region, literacy level, preferred format) while preserving facts.
4. **Format Decision Agent**: chooses the best output format from `audio`, `short_summary`, `long_article`, or `visual`.

