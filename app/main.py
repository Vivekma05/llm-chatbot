"""FastAPI backend: POST /api/chat sends a message, POST /api/reset clears it."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .conversation import ConversationManager
from .llm import GeminiClient, LLMError

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Mentor Chatbot")
_manager: ConversationManager | None = None


def get_manager() -> ConversationManager:
    """Create the LLM client lazily so the server can start (and show a clear
    error) even when the API key is missing."""
    global _manager
    if _manager is None:
        _manager = ConversationManager(llm=GeminiClient())
    return _manager


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    turns: int
    messages_in_context: int
    has_summary: bool


class ResetRequest(BaseModel):
    conversation_id: str


@app.exception_handler(LLMError)
async def llm_error_handler(_, exc: LLMError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.user_message})


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})
    conv, reply = get_manager().chat(req.conversation_id, text)
    return ChatResponse(
        conversation_id=conv.id,
        reply=reply,
        turns=conv.total_turns,
        messages_in_context=len(conv.messages),
        has_summary=bool(conv.summary),
    )


@app.post("/api/reset")
def reset(req: ResetRequest):
    if _manager:
        _manager.reset(req.conversation_id)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True, "model": settings.model, "strategy": settings.context_strategy}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
