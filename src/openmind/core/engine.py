"""
Main OpenMind engine.

Provides the :class:`OpenMind` class -- the primary interface for chatting
with local AI models via the Ollama REST API, with optional RAG augmentation
and persistent memory.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from openmind.core.config import Config, ModelConfig, DEFAULT_DATA_DIR
from openmind.core.memory import MemoryManager
from openmind.core.rag import RAGPipeline
from openmind.utils.helpers import timer

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_LIST_MODELS_ENDPOINT = "/api/tags"
OLLAMA_MODEL_INFO_ENDPOINT = "/api/show"
OLLAMA_PULL_ENDPOINT = "/api/pull"


class OllamaError(Exception):
    """Raised when the Ollama API returns an error or is unreachable."""


class OpenMind:
    """High-level interface to local AI via Ollama.

    Wraps the Ollama REST API with conversation memory, RAG support, and
    convenient helpers.  The engine auto-detects whether Ollama is running
    and provides actionable error messages when it is not.

    Args:
        model: Name of the Ollama model (e.g. ``"llama3"``).
        host: Host for the OpenMind API server (not Ollama).
        port: Port for the OpenMind API server.
        config: Optional :class:`Config` object. When provided, its values
            take precedence over the positional arguments.
        ollama_base_url: Base URL of the Ollama API.

    Example::

        engine = OpenMind(model="llama3")
        response = engine.chat("Hello, how are you?")
        print(response)

        # With RAG
        engine.ingest("knowledge_base.md")
        response = engine.chat("What does the knowledge base say about X?")
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "0.0.0.0",
        port: int = 3000,
        config: Optional[Config] = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        # Resolve configuration
        if config is not None:
            self._config = config
            model = self._config.model.name
            host = self._config.server.host
            port = self._config.server.port
        else:
            self._config = Config()

        self._model: str = model
        self._host: str = host
        self._port: int = port
        self._ollama_url: str = ollama_base_url
        self._client: httpx.Client = httpx.Client(
            base_url=self._ollama_url,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        self._async_client: Optional[httpx.AsyncClient] = None

        # Memory
        memory_path = (
            self._config.memory.long_term_db_path
            if self._config.memory.long_term_enabled
            else None
        )
        self.memory = MemoryManager(
            short_term_max=self._config.memory.short_term_max_messages,
            long_term_path=memory_path,
        )

        # RAG
        self.rag: Optional[RAGPipeline] = None
        if self._config.rag.enabled:
            self.rag = RAGPipeline(
                chunk_size=self._config.rag.chunk_size,
                chunk_overlap=self._config.rag.chunk_overlap,
                embedding_model=self._config.rag.embedding_model,
                max_chunks=self._config.rag.max_chunks,
            )

        # Stats tracking
        self._total_queries: int = 0
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Name of the currently active model."""
        return self._model

    @property
    def host(self) -> str:
        """API server host."""
        return self._host

    @property
    def port(self) -> int:
        """API server port."""
        return self._port

    # ------------------------------------------------------------------
    # Ollama connectivity
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        """Verify that Ollama is reachable.

        Returns:
            ``True`` if Ollama responded to a health check.

        Raises:
            OllamaError: If Ollama is not running.
        """
        try:
            resp = self._client.get("/")
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        raise OllamaError(
            "Ollama is not running. Please start it first:\n"
            "  1. Install Ollama from https://ollama.ai\n"
            "  2. Run 'ollama serve' in a terminal\n"
            "  3. Pull a model with 'ollama pull llama3'\n"
            "Then try again."
        )

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lazily create and return the async HTTP client."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self._ollama_url,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._async_client

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a message and return the model's response.

        Args:
            message: The user message.
            system_prompt: Optional system prompt to set behaviour.
            temperature: Override the default sampling temperature.

        Returns:
            The assistant's response text.

        Raises:
            OllamaError: If Ollama is unreachable or returns an error.
        """
        self._check_ollama()

        # Build RAG context if available
        augmented_message = self._augment_with_rag(message)

        # Build messages list
        messages = self._build_messages(augmented_message, system_prompt)

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        elif self._config.model.temperature != 0.7:
            payload["options"] = {"temperature": self._config.model.temperature}

        with timer() as t:
            resp = self._client.post(OLLAMA_CHAT_ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()

        response_text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)

        # Update memory
        self.memory.add("user", message)
        self.memory.add("assistant", response_text)

        # Update stats
        self._total_queries += 1
        self._total_tokens += eval_count

        logger.info(
            "Chat completed in %.2fs (model=%s, tokens=%d)",
            t.elapsed,
            self._model,
            eval_count,
        )
        return response_text

    async def chat_stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """Send a message and yield response tokens as they arrive.

        Args:
            message: The user message.
            system_prompt: Optional system prompt.
            temperature: Override the default sampling temperature.

        Yields:
            Response text chunks as they are generated.

        Raises:
            OllamaError: If Ollama is unreachable or returns an error.
        """
        self._check_ollama()

        augmented_message = self._augment_with_rag(message)
        messages = self._build_messages(augmented_message, system_prompt)

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        elif self._config.model.temperature != 0.7:
            payload["options"] = {"temperature": self._config.model.temperature}

        client = self._get_async_client()
        full_response = ""

        async with client.stream("POST", OLLAMA_CHAT_ENDPOINT, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    import json

                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        full_response += delta
                        yield delta
                except Exception:
                    continue

        # Update memory and stats after streaming completes
        self.memory.add("user", message)
        self.memory.add("assistant", full_response)
        self._total_queries += 1

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> int:
        """Ingest a document into the RAG pipeline.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Number of chunks created.

        Raises:
            OllamaError: If RAG is not enabled.
            FileNotFoundError: If the file does not exist.
        """
        if self.rag is None:
            raise OllamaError("RAG is not enabled. Enable it in your configuration.")
        return self.rag.ingest_file(file_path)

    def _augment_with_rag(self, message: str) -> str:
        """Augment a user message with RAG context if available."""
        if self.rag is None or self.rag.store.size == 0:
            return message
        context = self.rag.build_context(message)
        if context:
            return f"{context}\n\nUser Question: {message}"
        return message

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def switch_model(self, model_name: str) -> None:
        """Switch to a different Ollama model.

        Args:
            model_name: Name of the model to switch to.
        """
        self._model = model_name
        self._config.model.name = model_name
        logger.info("Switched to model: %s", model_name)

    def list_models(self) -> List[Dict[str, Any]]:
        """List locally available Ollama models.

        Returns:
            A list of model info dictionaries.

        Raises:
            OllamaError: If Ollama is unreachable.
        """
        self._check_ollama()
        resp = self._client.get(OLLAMA_LIST_MODELS_ENDPOINT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", [])

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model.

        Args:
            model_name: Model name. Defaults to the current model.

        Returns:
            A dictionary with model details.

        Raises:
            OllamaError: If Ollama is unreachable.
        """
        self._check_ollama()
        name = model_name or self._model
        resp = self._client.post(OLLAMA_MODEL_INFO_ENDPOINT, json={"name": name})
        resp.raise_for_status()
        return resp.json()

    def pull_model(self, model_name: str) -> str:
        """Pull (download) a model from the Ollama registry.

        Args:
            model_name: Name of the model to pull.

        Returns:
            A status message.

        Raises:
            OllamaError: If Ollama is unreachable.
        """
        self._check_ollama()
        resp = self._client.post(OLLAMA_PULL_ENDPOINT, json={"name": model_name}, timeout=600.0)
        resp.raise_for_status()
        return f"Model '{model_name}' pulled successfully."

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics.

        Returns:
            A dictionary with query count, token usage, model info,
            memory stats, and RAG stats.
        """
        stats: Dict[str, Any] = {
            "model": self._model,
            "host": self._host,
            "port": self._port,
            "total_queries": self._total_queries,
            "total_tokens": self._total_tokens,
            "memory": {
                "short_term_messages": len(self.memory.short_term),
                "long_term_enabled": self.memory.long_term is not None,
                "long_term_count": (
                    self.memory.long_term.count() if self.memory.long_term else 0
                ),
            },
        }
        if self.rag is not None:
            stats["rag"] = self.rag.stats
        return stats

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve conversation history.

        Args:
            limit: Maximum messages to return.

        Returns:
            A list of message dictionaries.
        """
        messages = self.memory.get_history(limit=limit)
        return [m.to_dict() for m in messages]

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.memory.clear()
        self._total_queries = 0
        self._total_tokens = 0
        logger.info("Conversation history cleared.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        message: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages payload for the Ollama chat API.

        Constructs the full message list from the system prompt (if any),
        conversation history, and the new user message.

        Args:
            message: The current user message.
            system_prompt: Optional system prompt.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        messages: List[Dict[str, str]] = []

        # System prompt
        effective_system = system_prompt or "You are a helpful AI assistant."
        messages.append({"role": "system", "content": effective_system})

        # Conversation context from short-term memory
        context = self.memory.get_context()
        messages.extend(context)

        # Current user message
        messages.append({"role": "user", "content": message})

        return messages

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources (HTTP clients, database connections)."""
        self._client.close()
        if self._async_client and not self._async_client.is_closed:
            asyncio.get_event_loop().run_until_complete(self._async_client.aclose())
        self.memory.close()

    def __repr__(self) -> str:
        return (
            f"OpenMind(model={self._model!r}, host={self._host!r}, port={self._port})"
        )

    def __enter__(self) -> "OpenMind":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
