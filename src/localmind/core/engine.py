"""
LocalMind — Main Engine

The central orchestrator that coordinates all components:
model management, agents, plugins, memory, and workflows.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from localmind.core.config import Config
from localmind.core.events import EventBus, Event, EventType
from localmind.core.memory import MemoryManager, MemoryType

logger = logging.getLogger(__name__)


class LocalMind:
    """
    LocalMind — Your Private AI Operating System.

    The main entry point for interacting with LocalMind. Coordinates
    model management, intelligent agents, plugins, memory, and workflows.

    Example:
        >>> mind = LocalMind()
        >>> mind.load_model("llama3")
        >>> response = await mind.chat("Hello, LocalMind!")
        >>> print(response)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LocalMind.

        Args:
            config: Configuration object. If None, loads from default locations.
        """
        self.config = config or Config.load()
        self.config.ensure_directories()

        # Core components
        self.event_bus = EventBus()
        self.memory = MemoryManager(
            short_term_max=self.config.memory.short_term_max_messages,
            long_term_db=str(
                Path(self.config.data_dir) / "memory" / "long_term.db"
            ),
        )

        # State
        self._model_loaded: bool = False
        self._model_name: Optional[str] = None
        self._model_provider: Optional[str] = None
        self._running: bool = False
        self._agents: Dict[str, Any] = {}
        self._plugins_loaded: bool = False

        # Setup logging
        self._setup_logging()

        logger.info("LocalMind initialized (v0.1.0)")
        logger.info(f"Data directory: {self.config.data_dir}")

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )

    # ─── Model Management ────────────────────────────────────────────

    def load_model(
        self,
        model_name: str,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Load an AI model.

        Args:
            model_name: Name of the model (e.g., "llama3", "mistral", "qwen2")
            provider: Model provider ("ollama", "llama.cpp", etc.)
            **kwargs: Additional model parameters
        """
        provider = provider or self.config.model.provider
        self._model_name = model_name
        self._model_provider = provider

        logger.info(f"Loading model '{model_name}' via {provider}...")

        try:
            if provider == "ollama":
                self._load_ollama_model(model_name, **kwargs)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            self._model_loaded = True
            self.config.model.name = model_name
            self.config.model.provider = provider

            self.event_bus.emit_sync(Event(
                type=EventType.MODEL_LOADED,
                data={"model": model_name, "provider": provider},
                source="engine",
            ))

            logger.info(f"Model '{model_name}' loaded successfully")

        except Exception as e:
            self.event_bus.emit_sync(Event(
                type=EventType.MODEL_ERROR,
                data={"model": model_name, "error": str(e)},
                source="engine",
            ))
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def _load_ollama_model(self, model_name: str, **kwargs: Any) -> None:
        """Load a model via Ollama."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for Ollama integration. "
                "Install it with: pip install httpx"
            )

        # Check if Ollama is running
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama server returned non-200 status")
        except httpx.ConnectError:
            raise ConnectionError(
                "Ollama is not running. Please start Ollama first:\n"
                "  ollama serve\n"
                "Then pull the model:\n"
                f"  ollama pull {model_name}"
            )

        # Check if model is available, pull if not
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        if model_name not in model_names:
            logger.info(f"Model '{model_name}' not found locally. Pulling...")
            try:
                pull_response = httpx.post(
                    "http://localhost:11434/api/pull",
                    json={"name": model_name, "stream": False},
                    timeout=600,
                )
                if pull_response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to pull model: {pull_response.text}"
                    )
                logger.info(f"Model '{model_name}' pulled successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to pull model '{model_name}': {e}. "
                    f"Please run: ollama pull {model_name}"
                ) from e

    def unload_model(self) -> None:
        """Unload the current model."""
        if self._model_loaded:
            logger.info(f"Unloading model '{self._model_name}'")
            self._model_loaded = False
            self._model_name = None
            self.event_bus.emit_sync(Event(
                type=EventType.MODEL_UNLOADED,
                source="engine",
            ))

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models from the configured provider."""
        if self._model_provider == "ollama":
            return self._list_ollama_models()
        return []

    def _list_ollama_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [
                    {
                        "name": m.get("name", ""),
                        "size": m.get("size", 0),
                        "modified_at": m.get("modified_at", ""),
                        "details": m.get("details", {}),
                    }
                    for m in models
                ]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
        return []

    # ─── Chat Interface ──────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat message and get a response.

        Args:
            message: The user's message
            system_prompt: Optional system prompt override
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            The model's response text
        """
        if not self._model_loaded:
            raise RuntimeError(
                "No model loaded. Call load_model() first, or start LocalMind "
                "with a model specified."
            )

        # Add user message to memory
        self.memory.add_message("user", message)

        # Set system prompt if provided
        if system_prompt:
            self.memory.short_term.set_system_prompt(system_prompt)

        # Emit event
        self.event_bus.emit_sync(Event(
            type=EventType.CHAT_MESSAGE,
            data={"message": message, "model": self._model_name},
            source="engine",
        ))

        # Get response from model
        try:
            response_text = await self._generate_response(
                message=message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Add assistant response to memory
            self.memory.add_message("assistant", response_text)

            # Emit response event
            self.event_bus.emit_sync(Event(
                type=EventType.CHAT_RESPONSE,
                data={
                    "message": message,
                    "response": response_text,
                    "model": self._model_name,
                },
                source="engine",
            ))

            return response_text

        except Exception as e:
            self.event_bus.emit_sync(Event(
                type=EventType.CHAT_ERROR,
                data={"message": message, "error": str(e)},
                source="engine",
            ))
            raise

    async def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Send a chat message and stream the response.

        Yields chunks of the response as they are generated.

        Args:
            message: The user's message
            system_prompt: Optional system prompt override
            temperature: Sampling temperature (0.0-2.0)
            **kwargs: Additional generation parameters

        Yields:
            Chunks of the response text
        """
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        self.memory.add_message("user", message)
        if system_prompt:
            self.memory.short_term.set_system_prompt(system_prompt)

        self.event_bus.emit_sync(Event(
            type=EventType.CHAT_MESSAGE,
            data={"message": message, "model": self._model_name},
            source="engine",
        ))

        full_response = ""

        try:
            async for chunk in self._generate_response_stream(
                message=message,
                temperature=temperature,
                **kwargs,
            ):
                full_response += chunk
                self.event_bus.emit_sync(Event(
                    type=EventType.CHAT_STREAM_CHUNK,
                    data={"chunk": chunk, "model": self._model_name},
                    source="engine",
                ))
                yield chunk

            self.memory.add_message("assistant", full_response)

            self.event_bus.emit_sync(Event(
                type=EventType.CHAT_RESPONSE,
                data={
                    "message": message,
                    "response": full_response,
                    "model": self._model_name,
                },
                source="engine",
            ))

        except Exception as e:
            self.event_bus.emit_sync(Event(
                type=EventType.CHAT_ERROR,
                data={"message": message, "error": str(e)},
                source="engine",
            ))
            raise

    async def _generate_response(
        self,
        message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model."""
        if self._model_provider == "ollama":
            return await self._ollama_generate(
                message=message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        raise ValueError(f"Unknown provider: {self._model_provider}")

    async def _generate_response_stream(
        self,
        message: str,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the model."""
        if self._model_provider == "ollama":
            async for chunk in self._ollama_generate_stream(
                message=message,
                temperature=temperature,
                **kwargs,
            ):
                yield chunk
            return
        raise ValueError(f"Unknown provider: {self._model_provider}")

    async def _ollama_generate(
        self,
        message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Ollama API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for Ollama integration")

        context = self.memory.get_context()
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": context + [{"role": "user", "content": message}],
            "stream": False,
        }

        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        if max_tokens is not None:
            payload.setdefault("options", {})["num_predict"] = max_tokens

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    async def _ollama_generate_stream(
        self,
        message: str,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Ollama API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for Ollama integration")

        context = self.memory.get_context()
        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": context + [{"role": "user", "content": message}],
            "stream": True,
        }

        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        import json
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    # ─── Agent System ────────────────────────────────────────────────

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent."""
        self._agents[name] = agent
        self.event_bus.emit_sync(Event(
            type=EventType.AGENT_CREATED,
            data={"agent_name": name},
            source="engine",
        ))
        logger.info(f"Agent '{name}' registered")

    def get_agent(self, name: str) -> Optional[Any]:
        """Get a registered agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self._agents.keys())

    async def execute_agent(
        self,
        agent_name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute a registered agent.

        Args:
            agent_name: Name of the agent to execute
            task: Task description for the agent
            context: Additional context for the agent

        Returns:
            The agent's result
        """
        agent = self._agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found. "
                           f"Available: {list(self._agents.keys())}")

        self.event_bus.emit_sync(Event(
            type=EventType.AGENT_STARTED,
            data={"agent_name": agent_name, "task": task},
            source="engine",
        ))

        try:
            if hasattr(agent, "execute"):
                result = await agent.execute(task, context or {})
            elif callable(agent):
                result = agent(task, context or {})
            else:
                raise TypeError(f"Agent '{agent_name}' is not callable")

            self.event_bus.emit_sync(Event(
                type=EventType.AGENT_FINISHED,
                data={"agent_name": agent_name, "result": str(result)[:500]},
                source="engine",
            ))

            return str(result)

        except Exception as e:
            self.event_bus.emit_sync(Event(
                type=EventType.AGENT_ERROR,
                data={"agent_name": agent_name, "error": str(e)},
                source="engine",
            ))
            raise

    async def collaborate(
        self,
        agents: List[Any],
        task: str,
    ) -> str:
        """
        Have multiple agents collaborate on a task.

        Each agent processes the task sequentially, building on
        the results of previous agents.

        Args:
            agents: List of agents to collaborate
            task: The task description

        Returns:
            Combined result from all agents
        """
        results = []
        current_task = task

        for i, agent in enumerate(agents):
            agent_name = getattr(agent, "name", f"agent_{i}")
            logger.info(f"Agent '{agent_name}' working on task...")

            try:
                if hasattr(agent, "execute"):
                    result = await agent.execute(current_task, {})
                elif callable(agent):
                    result = agent(current_task)
                else:
                    continue

                results.append(f"[{agent_name}]: {result}")
                # Feed result to next agent
                current_task = f"Original task: {task}\n\nPrevious results:\n" + "\n".join(results)

            except Exception as e:
                logger.error(f"Agent '{agent_name}' failed: {e}")
                results.append(f"[{agent_name}]: ERROR - {e}")

        return "\n\n".join(results)

    # ─── RAG (Document Ingestion) ────────────────────────────────────

    def ingest_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Ingest a document into the RAG system.

        Args:
            file_path: Path to the document
            metadata: Optional metadata for the document

        Returns:
            Number of chunks created
        """
        from localmind.core.rag import RAGPipeline

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        rag = RAGPipeline(self.config)
        chunks = rag.ingest_file(file_path, metadata)

        self.event_bus.emit_sync(Event(
            type=EventType.RAG_DOCUMENT_INGESTED,
            data={
                "file": str(file_path),
                "chunks": chunks,
            },
            source="engine",
        ))

        return chunks

    async def query_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system for relevant documents.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of relevant document chunks with scores
        """
        from localmind.core.rag import RAGPipeline

        rag = RAGPipeline(self.config)
        return await rag.query(query, top_k=top_k)

    # ─── Plugin System ───────────────────────────────────────────────

    def load_plugins(self) -> None:
        """Load all configured plugins."""
        from localmind.plugins import PluginManager

        manager = PluginManager(self.config)
        manager.load_all()
        self._plugins_loaded = True
        logger.info("All plugins loaded")

    # ─── System ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "version": "0.1.0",
            "model_loaded": self._model_loaded,
            "model_name": self._model_name,
            "model_provider": self._model_provider,
            "agents_registered": len(self._agents),
            "plugins_loaded": self._plugins_loaded,
            "memory": self.memory.get_stats(),
            "config": {
                "data_dir": self.config.data_dir,
                "log_level": self.config.log_level,
            },
        }

    def clear_conversation(self) -> None:
        """Clear the current conversation history."""
        self.memory.clear_conversation()
        logger.info("Conversation cleared")

    def shutdown(self) -> None:
        """Shut down LocalMind gracefully."""
        logger.info("Shutting down LocalMind...")
        self.event_bus.emit_sync(Event(
            type=EventType.SYSTEM_SHUTDOWN,
            source="engine",
        ))
        self.event_bus.shutdown()
        self._running = False
        logger.info("LocalMind shut down complete")

    def __repr__(self) -> str:
        model_info = f"model='{self._model_name}'" if self._model_loaded else "no model"
        return f"LocalMind({model_info}, agents={len(self._agents)})"
