"""
REST API and WebSocket server for LocalMind.

Provides HTTP endpoints for chat, model management, agents,
plugins, and system operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LocalMindAPI:
    """
    LocalMind API server.

    Provides REST and WebSocket endpoints for interacting with
    LocalMind programmatically.
    """

    def __init__(self, engine=None, host: str = "0.0.0.0", port: int = 8080):
        self.engine = engine
        self.host = host
        self.port = port
        self._app = None
        self._server = None

    def create_app(self):
        """Create the FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException, UploadFile, File
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse
            from pydantic import BaseModel
        except ImportError:
            logger.error(
                "FastAPI is required for the API server. "
                "Install with: pip install 'localmind[api]'"
            )
            return None

        app = FastAPI(
            title="LocalMind API",
            description="Your Private AI Operating System — API",
            version="0.1.0",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ─── Request/Response Models ─────────────────────────────

        class ChatRequest(BaseModel):
            message: str
            system_prompt: Optional[str] = None
            temperature: Optional[float] = None
            max_tokens: Optional[int] = None
            stream: bool = False

        class ChatResponse(BaseModel):
            response: str
            model: str
            message: str

        class ModelRequest(BaseModel):
            name: str
            provider: Optional[str] = None

        class AgentRequest(BaseModel):
            agent_name: str
            task: str
            context: Optional[Dict[str, Any]] = None

        class WorkflowStepRequest(BaseModel):
            step_id: str
            name: Optional[str] = None
            agent: Optional[str] = None
            depends_on: Optional[List[str]] = None

        class WorkflowCreateRequest(BaseModel):
            name: str
            description: Optional[str] = None
            steps: List[WorkflowStepRequest]

        class DocumentIngestRequest(BaseModel):
            file_path: str
            metadata: Optional[Dict[str, Any]] = None

        # ─── Routes ──────────────────────────────────────────────

        @app.get("/")
        async def root():
            """API root endpoint."""
            return {
                "name": "LocalMind API",
                "version": "0.1.0",
                "status": "running",
                "docs": "/docs",
            }

        @app.get("/api/v1/stats")
        async def get_stats():
            """Get system statistics."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            return self.engine.get_stats()

        # ─── Model Endpoints ─────────────────────────────────────

        @app.get("/api/v1/models")
        async def list_models():
            """List available models."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            return {"models": self.engine.list_available_models()}

        @app.post("/api/v1/models/load")
        async def load_model(request: ModelRequest):
            """Load a model."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            try:
                self.engine.load_model(request.name, request.provider)
                return {"status": "ok", "model": request.name}
            except Exception as e:
                raise HTTPException(500, str(e))

        @app.post("/api/v1/models/unload")
        async def unload_model():
            """Unload the current model."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            self.engine.unload_model()
            return {"status": "ok"}

        # ─── Chat Endpoints ──────────────────────────────────────

        @app.post("/api/v1/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Send a chat message."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")

            try:
                response = await self.engine.chat(
                    message=request.message,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                return ChatResponse(
                    response=response,
                    model=self.engine._model_name or "unknown",
                    message=request.message,
                )
            except Exception as e:
                raise HTTPException(500, str(e))

        @app.post("/api/v1/chat/stream")
        async def chat_stream(request: ChatRequest):
            """Stream a chat response."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")

            async def generate():
                async for chunk in self.engine.chat_stream(
                    message=request.message,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                ):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )

        @app.get("/api/v1/chat/history")
        async def get_chat_history():
            """Get conversation history."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            return {"history": self.engine.memory.get_context()}

        @app.delete("/api/v1/chat/history")
        async def clear_chat_history():
            """Clear conversation history."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            self.engine.clear_conversation()
            return {"status": "ok"}

        # ─── Agent Endpoints ─────────────────────────────────────

        @app.get("/api/v1/agents")
        async def list_agents():
            """List registered agents."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            return {"agents": self.engine.list_agents()}

        @app.post("/api/v1/agents/execute")
        async def execute_agent(request: AgentRequest):
            """Execute an agent."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            try:
                result = await self.engine.execute_agent(
                    agent_name=request.agent_name,
                    task=request.task,
                    context=request.context,
                )
                return {"result": result}
            except Exception as e:
                raise HTTPException(500, str(e))

        # ─── RAG Endpoints ───────────────────────────────────────

        @app.post("/api/v1/rag/ingest")
        async def ingest_document(request: DocumentIngestRequest):
            """Ingest a document."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            try:
                chunks = self.engine.ingest_document(
                    request.file_path,
                    request.metadata,
                )
                return {"status": "ok", "chunks": chunks}
            except Exception as e:
                raise HTTPException(500, str(e))

        @app.post("/api/v1/rag/query")
        async def query_documents(query: str, top_k: int = 5):
            """Query documents."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            results = await self.engine.query_documents(query, top_k=top_k)
            return {"results": results}

        # ─── Plugin Endpoints ────────────────────────────────────

        @app.get("/api/v1/plugins")
        async def list_plugins():
            """List loaded plugins."""
            # This would need access to the plugin manager
            return {"plugins": []}

        # ─── Memory Endpoints ────────────────────────────────────

        @app.get("/api/v1/memory/stats")
        async def memory_stats():
            """Get memory statistics."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            return self.engine.memory.get_stats()

        @app.delete("/api/v1/memory")
        async def clear_memory():
            """Clear all memory."""
            if not self.engine:
                raise HTTPException(500, "Engine not initialized")
            self.engine.memory.clear_all()
            return {"status": "ok"}

        self._app = app
        return app

    async def start(self) -> None:
        """Start the API server."""
        app = self.create_app()
        if not app:
            logger.error("Failed to create API app")
            return

        try:
            import uvicorn
            self._server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=self.host,
                    port=self.port,
                    log_level="info",
                )
            )
            await self._server.serve()
        except ImportError:
            logger.error(
                "uvicorn is required for the API server. "
                "Install with: pip install 'localmind[api]'"
            )

    def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            self._server.should_exit = True
