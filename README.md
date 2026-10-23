# Mentor — Multi-turn LLM Chatbot (Gemini)

A chatbot backed by the Google Gemini API that holds a real multi-turn
conversation, with context-window management, a defined persona, and clean
error handling. Built for task **AI-1: LLM API Integration**.

## Features

| Requirement | How it's done |
|---|---|
| Backend endpoint | FastAPI `POST /api/chat` → Gemini → reply (`app/main.py`) |
| Multi-turn memory | History stored per `conversation_id`, sent with every request (`app/conversation.py`) |
| Context management | **Running summary** (default): once history passes `MAX_MESSAGES`, older messages are summarised by the LLM and prepended to the system prompt; only the last `KEEP_RECENT` are sent verbatim. **Truncation** is also available (`CONTEXT_STRATEGY=truncate`). If summarising fails it falls back to truncation. |
| System prompt | "Mentor", a study & career mentor for engineering students (`app/prompts.py`) |
| Error handling | Brief overloads (503) and timeouts are retried automatically (3 attempts, exponential backoff). If it still fails, rate limits, bad key, unavailable model, server errors, timeouts, network failure and empty responses become a clear message in the UI with a Retry button. A failed turn isn't saved, so retrying is safe. |
| Secrets | API key read from `GEMINI_API_KEY` via `.env`; `.env` is in `.gitignore` |

## Project structure

```
app/
  config.py        settings from environment variables
  prompts.py       system prompt + summariser prompt
  llm.py           Gemini wrapper, converts SDK errors to user-friendly LLMError
  conversation.py  per-conversation history + truncate/summarise strategy
  main.py          FastAPI endpoints, serves the web UI
static/index.html  chat UI (vanilla JS)
cli.py             terminal interface
tests/             unit + API tests using a fake LLM (no key needed)
```

## Setup

```bash
git clone <your-repo-url>
cd llm-chatbot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # Windows: copy .env.example .env
# edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)
```

## Run

**Web UI**

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The status line under the input shows the turn
count, how many messages are in context, and when older messages have been
summarised.

**Terminal**

```bash
python cli.py        # /debug shows context size + summary, /reset, /quit
```

**Tests**

```bash
pytest -q
```

## How context management works

```
turns 1-6   : [system prompt] + all messages
turn 7+     : history > MAX_MESSAGES (12)
              -> oldest messages summarised into a running summary
              -> [system prompt + summary] + last KEEP_RECENT (6) messages
```

The summary is cumulative: each time it compacts, the previous summary and the
newly-dropped messages are merged, so facts like the user's name survive long
conversations while the request size stays bounded.

## Example test conversation

1. "Hi, I'm Vivek, a 4th-year CSE student."
2. "I have placement tests in 6 weeks and I'm weak at dynamic programming."
3. "Give me a weekly plan for DP."
4. "Make week 2 lighter, I have exams then."
5. "What's my name and what was my weak topic?" → should answer Vivek / DP

Set `MAX_MESSAGES=4` and `KEEP_RECENT=2` in `.env` to see summarisation kick in
within a few turns.

## API

`POST /api/chat`
```json
{ "message": "Hi", "conversation_id": null }
```
→
```json
{ "conversation_id": "…", "reply": "…", "turns": 1, "messages_in_context": 2, "has_summary": false }
```
Errors return `{ "error": "<readable message>" }` with an appropriate status code.

`POST /api/reset` `{ "conversation_id": "…" }` clears a conversation.

Note: conversations are kept in memory, so they reset when the server restarts.
