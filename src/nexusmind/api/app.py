"""FastAPI REST API for NexusMind.

Provides HTTP endpoints for chat, model management, memory operations,
skill management, task scheduling, and document ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from nexusmind.core.config import Config
from nexusmind.core.engine import NexusMind

logger = logging.getLogger(__name__)

# Module-level engine instance (initialized by create_app)
_engine: NexusMind | None = None


def get_engine() -> NexusMind:
    """Get the global NexusMind engine instance.

    Returns:
        The NexusMind engine.

    Raises:
        RuntimeError: If the engine has not been initialized.
    """
    if _engine is None:
        raise RuntimeError("NexusMind engine not initialized. Call create_app() first.")
    return _engine


def create_app(config: Config | None = None) -> Any:
    """Create and configure the FastAPI application.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI is required for the API. "
            "Install it with: pip install nexusmind[api]"
        )

    global _engine
    _engine = NexusMind(config=config)

    app = FastAPI(
        title="NexusMind API",
        description="AI Agent with Persistent Memory, Auto Skill Evolution, "
                    "Offline Scheduling, and Multi-Provider LLM Support",
        version="0.1.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_engine.config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files if directory exists
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # Root endpoint
    # ------------------------------------------------------------------

    @app.get("/")
    async def api_info() -> dict[str, Any]:
        """Get API information and status."""
        return {
            "name": "NexusMind API",
            "version": "0.1.0",
            "status": "running",
            "active_model": _engine._active_model,
            "active_provider": _engine.config.model.provider,
        }

    # ------------------------------------------------------------------
    # Chat endpoints
    # ------------------------------------------------------------------

    @app.post("/api/v1/chat")
    async def chat_endpoint(
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a chat message and get a response.

        Request body:
            messages: list of {"role": str, "content": str} or a single string
            model: optional model override
            stream: boolean (ignored, use /chat/stream for streaming)
        """
        try:
            messages = body.get("messages", "")
            model = body.get("model")
            provider = body.get("provider")

            response = await _engine.chat(
                messages=messages,
                model=model,
                provider=provider,
            )
            return {
                "content": response.content,
                "model": response.model,
                "provider": response.provider,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            }
        except Exception as e:
            logger.error("Chat error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/chat/stream")
    async def chat_stream_endpoint(body: dict[str, Any]) -> Any:
        """Stream a chat response using Server-Sent Events.

        Request body:
            messages: list of {"role": str, "content": str} or a single string
            model: optional model override
        """
        try:
            from sse_starlette.sse import EventSourceResponse

            messages = body.get("messages", "")
            model = body.get("model")
            provider = body.get("provider")

            async def event_generator():
                async for chunk in _engine.chat_stream(
                    messages=messages,
                    model=model,
                    provider=provider,
                ):
                    yield {"data": chunk}
                yield {"data": "[DONE]"}

            return EventSourceResponse(event_generator())
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="sse-starlette is required for streaming. "
                       "Install with: pip install nexusmind[api]",
            )
        except Exception as e:
            logger.error("Stream error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Model endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/models")
    async def list_models() -> dict[str, Any]:
        """List available models from all providers."""
        try:
            models = await _engine.list_models()
            return {"models": models}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/models/switch")
    async def switch_model(body: dict[str, Any]) -> dict[str, Any]:
        """Switch the active model.

        Request body:
            model: str - model name
            provider: optional str - provider name
        """
        try:
            model = body.get("model", "")
            provider = body.get("provider")
            if not model:
                raise HTTPException(status_code=400, detail="model is required")
            await _engine.switch_model(model, provider)
            return {
                "status": "ok",
                "model": model,
                "provider": provider or _engine.config.model.provider,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Memory endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/memory")
    async def list_memories(
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List stored memories."""
        try:
            memories = _engine.memory.long_term.list_all(
                category=category, limit=limit, offset=offset
            )
            return {
                "memories": [
                    {
                        "key": m.key,
                        "value": m.value,
                        "category": m.category,
                        "tags": m.tags,
                        "importance": m.importance,
                        "updated_at": m.updated_at,
                    }
                    for m in memories
                ],
                "total": _engine.memory.long_term.count(category),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/memory")
    async def store_memory(body: dict[str, Any]) -> dict[str, Any]:
        """Store a memory.

        Request body:
            key: str - memory key
            value: str - memory value
            category: optional str
            tags: optional list[str]
            importance: optional float (0.0-1.0)
        """
        try:
            key = body.get("key", "")
            value = body.get("value", "")
            if not key or not value:
                raise HTTPException(status_code=400, detail="key and value are required")

            entry = _engine.memory.remember(
                key=key,
                value=value,
                category=body.get("category", "general"),
                tags=body.get("tags"),
                importance=body.get("importance", 0.5),
            )
            return {"status": "ok", "key": entry.key}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/memory/{key:path}")
    async def delete_memory(key: str) -> dict[str, Any]:
        """Delete a specific memory by key."""
        try:
            deleted = _engine.memory.forget(key)
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Memory '{key}' not found")
            return {"status": "ok", "deleted": key}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Skill endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/skills")
    async def list_skills(tag: Optional[str] = None) -> dict[str, Any]:
        """List available skills."""
        try:
            skills = _engine.skills.list_skills(tag=tag)
            return {
                "skills": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "usage_count": s.usage_count,
                        "success_rate": round(s.success_rate, 3),
                        "tags": s.tags,
                        "source": s.source,
                    }
                    for s in skills
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/skills/execute")
    async def execute_skill(body: dict[str, Any]) -> dict[str, Any]:
        """Execute a skill.

        Request body:
            name: str - skill name
            context: dict - execution context
        """
        try:
            name = body.get("name", "")
            context = body.get("context", {})
            if not name:
                raise HTTPException(status_code=400, detail="name is required")

            result = await _engine.skills.execute_skill(name, context)
            return result
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Scheduler endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/scheduler/tasks")
    async def list_tasks(
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """List scheduled tasks."""
        try:
            from nexusmind.core.scheduler import TaskStatus

            task_status = TaskStatus(status) if status else None
            tasks = _engine.scheduler.list_tasks(status=task_status)
            return {
                "tasks": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "prompt": t.prompt[:200],
                        "schedule": t.schedule,
                        "status": t.status.value,
                        "next_run": t.next_run,
                        "run_count": t.run_count,
                        "tags": t.tags,
                    }
                    for t in tasks
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/scheduler/tasks")
    async def create_task(body: dict[str, Any]) -> dict[str, Any]:
        """Create a scheduled task.

        Request body:
            name: str - task name
            prompt: str - task prompt
            schedule: str - schedule expression
            model: optional str
            notify: optional list[str]
            tags: optional list[str]
        """
        try:
            name = body.get("name", "")
            prompt = body.get("prompt", "")
            schedule = body.get("schedule", "")
            if not name or not prompt:
                raise HTTPException(status_code=400, detail="name and prompt are required")

            task = _engine.scheduler.schedule_task(
                name=name,
                prompt=prompt,
                schedule=schedule,
                model=body.get("model"),
                notify=body.get("notify"),
                tags=body.get("tags"),
            )
            return {
                "status": "ok",
                "task": {
                    "id": task.id,
                    "name": task.name,
                    "schedule": task.schedule,
                    "next_run": task.next_run,
                },
            }
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Stats endpoint
    # ------------------------------------------------------------------

    @app.get("/api/v1/stats")
    async def get_stats() -> dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            return _engine.get_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Ingest endpoint
    # ------------------------------------------------------------------

    @app.post("/api/v1/ingest")
    async def ingest_documents(body: dict[str, Any]) -> dict[str, Any]:
        """Ingest documents into the RAG pipeline.

        Request body:
            paths: list[str] - file paths or directories to ingest
        """
        try:
            paths = body.get("paths", [])
            if not paths:
                raise HTTPException(status_code=400, detail="paths is required")

            chunks = _engine.ingest_documents(paths)
            return {
                "status": "ok",
                "chunks_created": chunks,
                "rag_stats": _engine.rag.get_stats(),
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
