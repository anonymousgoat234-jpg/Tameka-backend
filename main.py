"""
Tameka Backend — orchestration API
Routes text queries to Groq (Llama, cheap/fast) or Claude (complex reasoning),
stores conversation memory in Supabase, and exposes a /ping route to keep
the Render free instance warm.

This service does NOT handle audio. It expects transcribed text in,
and returns text out. Wake-word detection, speech-to-text (Whisper),
and text-to-speech (Piper) run on your local client device (Pi/laptop/phone),
not here.
"""

import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from supabase import create_client, Client
from anthropic import Anthropic

load_dotenv()

# --- Config ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Model choices — adjust as needed
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
CLAUDE_MODEL_CHEAP = os.environ.get("CLAUDE_MODEL_CHEAP", "claude-haiku-4-5-20251001")
CLAUDE_MODEL_STRONG = os.environ.get("CLAUDE_MODEL_STRONG", "claude-sonnet-4-6")

# How many recent turns to include as context
MEMORY_TURNS = int(os.environ.get("MEMORY_TURNS", "6"))

app = FastAPI(title="Tameka Backend")

# --- Clients (created lazily so missing keys don't crash /ping) ---
_supabase: Client | None = None
_anthropic: Anthropic | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_anthropic() -> Anthropic:
    global _anthropic
    if _anthropic is None:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="Anthropic API key not configured")
        _anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic


# --- Request/response models ---
class QueryRequest(BaseModel):
    text: str
    user_id: str = "default"  # lets you support multiple people/devices later
    force_model: str | None = None  # "groq" | "claude_cheap" | "claude_strong" | None (auto)


class QueryResponse(BaseModel):
    reply: str
    model_used: str
    latency_ms: int


# --- Simple routing logic ---
# You will likely tune this over time — start simple, adjust based on what
# actually needs Claude vs. what Groq/Llama handles fine.
COMPLEX_KEYWORDS = [
    "write", "draft", "code", "debug", "analyze", "explain in detail",
    "plan", "compare", "summarize", "review", "essay", "strategy",
    "why", "reason", "design",
]


def choose_model(text: str) -> str:
    lowered = text.lower()
    word_count = len(text.split())

    # Long queries or ones with complex-task keywords go to Claude
    if word_count > 25 or any(kw in lowered for kw in COMPLEX_KEYWORDS):
        return "claude_strong"

    # Medium queries needing some reasoning but not heavy lifting
    if word_count > 10:
        return "claude_cheap"

    # Short/simple queries -> cheap fast local-ish model
    return "groq"


# --- Memory helpers (Supabase) ---
def fetch_recent_memory(user_id: str, limit: int = MEMORY_TURNS) -> list[dict]:
    sb = get_supabase()
    result = (
        sb.table("conversations")
        .select("role, content")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    return list(reversed(rows))  # oldest first for correct conversation order


def store_turn(user_id: str, role: str, content: str) -> None:
    sb = get_supabase()
    sb.table("conversations").insert(
        {
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()


# --- Model callers ---
async def call_groq(prompt: str, history: list[dict]) -> str:
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 500,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def call_claude(prompt: str, history: list[dict], model: str) -> str:
    client = get_anthropic()
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": prompt})

    response = client.messages.create(
        model=model,
        max_tokens=800,
        messages=messages,
    )
    # Concatenate any text blocks in the response
    return "".join(block.text for block in response.content if block.type == "text")


# --- Routes ---
@app.get("/ping")
def ping():
    """Lightweight health check — used by the keep-alive cron job."""
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    start = time.time()

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    model_choice = req.force_model or choose_model(req.text)

    history = fetch_recent_memory(req.user_id)

    try:
        if model_choice == "groq":
            reply = await call_groq(req.text, history)
        elif model_choice == "claude_cheap":
            reply = call_claude(req.text, history, CLAUDE_MODEL_CHEAP)
        elif model_choice == "claude_strong":
            reply = call_claude(req.text, history, CLAUDE_MODEL_STRONG)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model choice: {model_choice}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e}")

    # Store both sides of the turn for future context
    store_turn(req.user_id, "user", req.text)
    store_turn(req.user_id, "assistant", reply)

    latency_ms = int((time.time() - start) * 1000)

    return QueryResponse(reply=reply, model_used=model_choice, latency_ms=latency_ms)


@app.get("/")
def root():
    return {"service": "Tameka Backend", "status": "running"}
