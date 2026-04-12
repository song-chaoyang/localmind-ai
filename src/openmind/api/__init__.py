"""
OpenMind REST API package.

Provides a FastAPI application with endpoints for chat, model management,
RAG ingestion, conversation history, and system statistics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from openmind.core.engine import OpenMind, OllamaError
from openmind.core.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared engine instance (set via create_app or serve)
# ---------------------------------------------------------------------------

_engine: Optional[OpenMind] = None


def _get_engine() -> OpenMind:
    """Return the shared engine instance, raising if not initialised."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="OpenMind engine is not initialised. Call create_app() first.",
        )
    return _engine


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(..., min_length=1, description="The user message.")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature.")


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    response: str
    model: str
    tokens_used: int = 0


class RAGIngestRequest(BaseModel):
    """Request body for the RAG ingest endpoint."""

    file_path: str = Field(..., description="Path to the file to ingest.")


class RAGIngestResponse(BaseModel):
    """Response body for the RAG ingest endpoint."""

    chunks_created: int
    message: str


class RAGQueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    query: str = Field(..., min_length=1, description="The search query.")


class RAGQueryResponse(BaseModel):
    """Response body for the RAG query endpoint."""

    results: List[Dict[str, Any]]
    context: str


class ModelPullRequest(BaseModel):
    """Request body for pulling a model."""

    name: str = Field(..., description="Name of the model to pull.")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_app(engine: Optional[OpenMind] = None, config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: An existing :class:`OpenMind` instance. If ``None`` a new
            one is created from *config* (or defaults).
        config: Optional :class:`Config` used when *engine* is ``None``.

    Returns:
        A fully configured :class:`FastAPI` application.
    """
    global _engine  # noqa: PLW0603

    if engine is not None:
        _engine = engine
    elif _engine is None:
        cfg = config or Config()
        _engine = OpenMind(
            model=cfg.model.name,
            host=cfg.server.host,
            port=cfg.server.port,
            config=cfg,
        )

    app = FastAPI(
        title="OpenMind API",
        description="One-click local AI chat API powered by Ollama.",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_engine._config.server.cors_origins,  # noqa: SLF001
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Return basic API information."""
        return {
            "name": "OpenMind API",
            "version": "0.1.0",
            "description": "One-click local AI chat API powered by Ollama.",
            "model": _engine.model,
            "endpoints": {
                "chat": "/api/v1/chat",
                "chat_stream": "/api/v1/chat/stream",
                "models": "/api/v1/models",
                "history": "/api/v1/history",
                "rag_ingest": "/api/v1/rag/ingest",
                "rag_query": "/api/v1/rag/query",
                "stats": "/api/v1/stats",
            },
        }

    @app.get("/api/v1/models")
    async def list_models() -> List[Dict[str, Any]]:
        """List all locally available Ollama models."""
        try:
            return _engine.list_models()
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        """Send a message and receive a complete response."""
        try:
            response = _engine.chat(
                message=req.message,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
            )
            stats = _engine.get_stats()
            return ChatResponse(
                response=response,
                model=_engine.model,
                tokens_used=stats.get("total_tokens", 0),
            )
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/v1/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Send a message and receive a Server-Sent Events stream."""

        async def event_generator():
            try:
                async for chunk in _engine.chat_stream(
                    message=req.message,
                    system_prompt=req.system_prompt,
                    temperature=req.temperature,
                ):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except OllamaError as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/history")
    async def get_history(limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve conversation history."""
        return _engine.get_history(limit=limit)

    @app.delete("/api/v1/history")
    async def clear_history() -> Dict[str, str]:
        """Clear all conversation history."""
        _engine.clear_history()
        return {"status": "ok", "message": "History cleared."}

    @app.post("/api/v1/rag/ingest", response_model=RAGIngestResponse)
    async def rag_ingest(req: RAGIngestRequest) -> RAGIngestResponse:
        """Ingest a document into the RAG pipeline."""
        try:
            chunks = _engine.ingest(req.file_path)
            return RAGIngestResponse(
                chunks_created=chunks,
                message=f"Successfully ingested {chunks} chunks from '{req.file_path}'.",
            )
        except OllamaError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/rag/query", response_model=RAGQueryResponse)
    async def rag_query(req: RAGQueryRequest) -> RAGQueryResponse:
        """Query the RAG pipeline for relevant document chunks."""
        if _engine.rag is None:
            raise HTTPException(status_code=400, detail="RAG is not enabled.")
        results = _engine.rag.query(req.query)
        context = _engine.rag.build_context(req.query)
        return RAGQueryResponse(
            results=[{"text": r.text, "score": r.score, "source": r.source} for r in results],
            context=context,
        )

    @app.get("/api/v1/stats")
    async def get_stats() -> Dict[str, Any]:
        """Return system and runtime statistics."""
        return _engine.get_stats()

    return app


# ---------------------------------------------------------------------------
# Convenience serve function
# ---------------------------------------------------------------------------

def serve(
    host: str = "0.0.0.0",
    port: int = 3000,
    config: Optional[Config] = None,
    engine: Optional[OpenMind] = None,
    log_level: str = "info",
) -> None:
    """Start the OpenMind API server using uvicorn.

    Args:
        host: Host address to bind to.
        port: Port number.
        config: Optional configuration.
        engine: Optional pre-built engine.
        log_level: Uvicorn log level.
    """
    import uvicorn

    cfg = config or Config()
    cfg.server.host = host
    cfg.server.port = port

    app = create_app(engine=engine, config=cfg)

    logger.info("Starting OpenMind API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


__all__ = ["create_app", "serve", "ChatRequest", "ChatResponse"]
