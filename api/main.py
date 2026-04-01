"""
FastAPI Application
────────────────────
Serves a full chat UI at GET /  and exposes the memory API.

Endpoints:
  GET  /              — Chat UI (HTML frontend)
  POST /chat          — Main chat endpoint
  GET  /status/{id}   — Memory diagnostics
  GET  /profile/{id}  — User profile
  POST /profile/{id}  — Update user profile
  DELETE /session/{id}— Reset short-term memory
  GET  /health        — Health check

Run with:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.memory_agent import MemoryAgent
from config.settings import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"

# ── Session registry ──────────────────────────────────────────────────────────
_agents: dict[str, MemoryAgent] = {}


def _get_agent(session_id: str) -> MemoryAgent:
    if session_id not in _agents:
        logger.info("Creating new MemoryAgent for session=%s", session_id)
        _agents[session_id] = MemoryAgent(session_id=session_id)
    return _agents[session_id]


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Memory Agent API starting up...")
    _get_agent(settings.session_id)
    yield
    logger.info("Memory Agent API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Persistent Memory Study Assistant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,          # disable Swagger UI
    redoc_url=None,         # disable ReDoc
    openapi_url=None,       # disable OpenAPI schema endpoint
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS, JS, images if any)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str = Field(default="default")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    turn_count: int
    token_estimate: int
    compression_triggered: bool
    latency_ms: float

class StatusResponse(BaseModel):
    session_id: str
    turn_count: int
    stm_turns: int
    episodic_entries: int
    semantic_facts: int
    stm_token_estimate: int

class ProfileResponse(BaseModel):
    profile: dict

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    goal: Optional[str] = None
    project: Optional[str] = None
    interest: Optional[str] = None
    constraint: Optional[str] = None
    preference_key: Optional[str] = None
    preference_value: Optional[str] = None
    custom_key: Optional[str] = None
    custom_value: Optional[str] = None


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the chat UI at the root URL."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found. Check api/static/index.html exists.")
    return FileResponse(str(index), media_type="text/html")


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    agent = _get_agent(request.session_id)
    try:
        result = agent.chat(request.message)
    except Exception as exc:
        logger.exception("Chat error session=%s: %s", request.session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return ChatResponse(
        response=result.reply,
        session_id=result.session_id,
        turn_count=result.turn_count,
        token_estimate=result.token_estimate,
        compression_triggered=result.compression_triggered,
        latency_ms=result.latency_ms,
    )


@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str) -> StatusResponse:
    agent = _get_agent(session_id)
    return StatusResponse(**agent.status())


@app.get("/profile/{session_id}", response_model=ProfileResponse)
async def get_profile(session_id: str) -> ProfileResponse:
    agent = _get_agent(session_id)
    return ProfileResponse(profile=agent.user_profile.as_dict())


@app.post("/profile/{session_id}")
async def update_profile(session_id: str, req: ProfileUpdateRequest) -> dict:
    agent = _get_agent(session_id)
    p = agent.user_profile
    if req.name:             p.set_name(req.name)
    if req.goal:             p.add_goal(req.goal)
    if req.project:          p.add_project(req.project)
    if req.interest:         p.add_interest(req.interest)
    if req.constraint:       p.add_constraint(req.constraint)
    if req.preference_key and req.preference_value:
        p.set_preference(req.preference_key, req.preference_value)
    if req.custom_key and req.custom_value:
        p.set_custom_fact(req.custom_key, req.custom_value)
    return {"status": "updated"}


@app.delete("/session/{session_id}")
async def reset_session(session_id: str) -> dict:
    if session_id in _agents:
        _agents[session_id].stm.clear()
        logger.info("STM cleared for session=%s", session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}