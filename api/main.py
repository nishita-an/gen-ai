"""
FastAPI Application
────────────────────
Exposes the MemoryAgent over HTTP.

Endpoints:
  POST /chat          — main chat endpoint
  GET  /status        — memory system diagnostics
  GET  /profile       — current user profile
  DELETE /session     — reset short-term memory for a session

Run with:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.memory_agent import MemoryAgent
from config.settings import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Session registry ──────────────────────────────────────────────────────────
# For production, replace with Redis or a DB-backed store.
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
    # Pre-warm the default session
    _get_agent(settings.session_id)
    yield
    logger.info("Memory Agent API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Persistent Memory Study Assistant",
    description=(
        "LLM-powered study assistant with 4-layer persistent memory: "
        "short-term, episodic, semantic, and user profile."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="User's message")
    session_id: str = Field(default="default", description="Session identifier")

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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="Send a message to the study assistant")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    The agent maintains coherent memory across 50+ turns without exceeding
    the model context window through automatic summarisation and retrieval.
    """
    agent = _get_agent(request.session_id)
    try:
        result = agent.chat(request.message)
    except Exception as exc:
        logger.exception("Chat error for session=%s: %s", request.session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        response=result.reply,
        session_id=result.session_id,
        turn_count=result.turn_count,
        token_estimate=result.token_estimate,
        compression_triggered=result.compression_triggered,
        latency_ms=result.latency_ms,
    )


@app.get("/status/{session_id}", response_model=StatusResponse, summary="Memory system diagnostics")
async def get_status(session_id: str) -> StatusResponse:
    agent = _get_agent(session_id)
    return StatusResponse(**agent.status())


@app.get("/profile/{session_id}", response_model=ProfileResponse, summary="Get user profile")
async def get_profile(session_id: str) -> ProfileResponse:
    agent = _get_agent(session_id)
    return ProfileResponse(profile=agent.user_profile.as_dict())


@app.post("/profile/{session_id}", summary="Manually update user profile")
async def update_profile(session_id: str, req: ProfileUpdateRequest) -> dict:
    agent = _get_agent(session_id)
    profile = agent.user_profile
    if req.name:
        profile.set_name(req.name)
    if req.goal:
        profile.add_goal(req.goal)
    if req.project:
        profile.add_project(req.project)
    if req.interest:
        profile.add_interest(req.interest)
    if req.constraint:
        profile.add_constraint(req.constraint)
    if req.preference_key and req.preference_value:
        profile.set_preference(req.preference_key, req.preference_value)
    if req.custom_key and req.custom_value:
        profile.set_custom_fact(req.custom_key, req.custom_value)
    return {"status": "updated"}


@app.delete("/session/{session_id}", summary="Reset short-term memory for a session")
async def reset_session(session_id: str) -> dict:
    """Clears working memory for the session (episodic and semantic memory are preserved)."""
    if session_id in _agents:
        _agents[session_id].stm.clear()
        logger.info("STM cleared for session=%s", session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}
