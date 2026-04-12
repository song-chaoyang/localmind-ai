"""
Configuration module for OpenMind.

Provides dataclass-based configuration for model settings, server settings,
RAG parameters, and a unified Config class with file/environment loading.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_DATA_DIR = Path.home() / ".openmind"


@dataclass
class ModelConfig:
    """Configuration for the AI model.

    Attributes:
        name: Name of the Ollama model to use (e.g. ``"llama3"``).
        temperature: Sampling temperature between 0.0 (deterministic) and 1.0 (creative).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        num_ctx: Context window size (number of tokens).
        max_tokens: Maximum tokens to generate per response.
        repeat_penalty: Penalty for repeating tokens.
    """

    name: str = "llama3"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 4096
    max_tokens: int = 2048
    repeat_penalty: float = 1.1


@dataclass
class ServerConfig:
    """Configuration for the OpenMind API server.

    Attributes:
        host: Host address to bind to.
        port: Port number to listen on.
        debug: Whether to run in debug mode.
        cors_origins: Allowed CORS origins (``"*"`` permits all).
    """

    host: str = "0.0.0.0"
    port: int = 3000
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class RAGConfig:
    """Configuration for the Retrieval-Augmented Generation pipeline.

    Attributes:
        enabled: Whether RAG is active.
        chunk_size: Maximum number of characters per text chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        max_chunks: Maximum number of chunks to retrieve per query.
        embedding_model: Name of the sentence-transformers model, or ``"hash"`` for
            a deterministic hash-based fallback.
    """

    enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks: int = 5
    embedding_model: str = "hash"


@dataclass
class MemoryConfig:
    """Configuration for the memory subsystem.

    Attributes:
        short_term_max_messages: Maximum messages kept in the sliding-window
            short-term memory.
        long_term_enabled: Whether to persist conversation history to SQLite.
        long_term_db_path: Path to the SQLite database file.
    """

    short_term_max_messages: int = 50
    long_term_enabled: bool = True
    long_term_db_path: str = str(DEFAULT_DATA_DIR / "memory.db")


class Config:
    """Unified configuration for the OpenMind application.

    Aggregates :class:`ModelConfig`, :class:`ServerConfig`, :class:`RAGConfig`,
    and :class:`MemoryConfig` into a single object that can be loaded from a JSON
    file, populated from environment variables, or constructed programmatically.

    Example::

        cfg = Config.from_env()
        cfg = Config.from_file("~/.openmind/config.json")
        cfg.save()  # writes to cfg.config_path
    """

    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        server: Optional[ServerConfig] = None,
        rag: Optional[RAGConfig] = None,
        memory: Optional[MemoryConfig] = None,
        data_dir: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.model: ModelConfig = model or ModelConfig()
        self.server: ServerConfig = server or ServerConfig()
        self.rag: RAGConfig = rag or RAGConfig()
        self.memory: MemoryConfig = memory or MemoryConfig()
        self.data_dir: Path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.config_path: Path = (
            Path(config_path) if config_path else self.data_dir / "config.json"
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of this configuration."""
        return {
            "model": asdict(self.model),
            "server": asdict(self.server),
            "rag": asdict(self.rag),
            "memory": asdict(self.memory),
            "data_dir": str(self.data_dir),
        }

    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """Apply values from a dictionary to the matching sub-configs."""
        if "model" in data and isinstance(data["model"], dict):
            for k, v in data["model"].items():
                if hasattr(self.model, k):
                    setattr(self.model, k, v)
        if "server" in data and isinstance(data["server"], dict):
            for k, v in data["server"].items():
                if hasattr(self.server, k):
                    setattr(self.server, k, v)
        if "rag" in data and isinstance(data["rag"], dict):
            for k, v in data["rag"].items():
                if hasattr(self.rag, k):
                    setattr(self.rag, k, v)
        if "memory" in data and isinstance(data["memory"], dict):
            for k, v in data["memory"].items():
                if hasattr(self.memory, k):
                    setattr(self.memory, k, v)
        if "data_dir" in data:
            self.data_dir = Path(data["data_dir"])

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Create a :class:`Config` from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            A new :class:`Config` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(path).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        config = cls(config_path=path)
        config._apply_dict(data)
        return config

    @classmethod
    def from_env(cls) -> Config:
        """Create a :class:`Config` populated from environment variables.

        Recognised environment variables (all optional):

        - ``OPENMIND_MODEL`` -- model name
        - ``OPENMIND_TEMPERATURE`` -- sampling temperature
        - ``OPENMIND_HOST`` -- server host
        - ``OPENMIND_PORT`` -- server port
        - ``OPENMIND_DATA_DIR`` -- data directory
        - ``OPENMIND_RAG_ENABLED`` -- ``"true"`` / ``"false"``
        - ``OPENMIND_RAG_CHUNK_SIZE`` -- chunk size in characters
        - ``OPENMIND_MEMORY_ENABLED`` -- ``"true"`` / ``"false"``

        Returns:
            A new :class:`Config` instance.
        """
        config = cls()

        env = os.environ
        if "OPENMIND_MODEL" in env:
            config.model.name = env["OPENMIND_MODEL"]
        if "OPENMIND_TEMPERATURE" in env:
            try:
                config.model.temperature = float(env["OPENMIND_TEMPERATURE"])
            except ValueError:
                pass
        if "OPENMIND_HOST" in env:
            config.server.host = env["OPENMIND_HOST"]
        if "OPENMIND_PORT" in env:
            try:
                config.server.port = int(env["OPENMIND_PORT"])
            except ValueError:
                pass
        if "OPENMIND_DATA_DIR" in env:
            config.data_dir = Path(env["OPENMIND_DATA_DIR"])
        if "OPENMIND_RAG_ENABLED" in env:
            config.rag.enabled = env["OPENMIND_RAG_ENABLED"].lower() in ("true", "1", "yes")
        if "OPENMIND_RAG_CHUNK_SIZE" in env:
            try:
                config.rag.chunk_size = int(env["OPENMIND_RAG_CHUNK_SIZE"])
            except ValueError:
                pass
        if "OPENMIND_MEMORY_ENABLED" in env:
            config.memory.long_term_enabled = env["OPENMIND_MEMORY_ENABLED"].lower() in (
                "true",
                "1",
                "yes",
            )

        return config

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> Config:
        """Load configuration, falling back to environment variables.

        If *path* is given and the file exists, it is loaded first and then
        overridden by any matching environment variables.

        Args:
            path: Optional path to a JSON config file.

        Returns:
            A new :class:`Config` instance.
        """
        if path is None:
            path = DEFAULT_DATA_DIR / "config.json"

        path = Path(path).expanduser().resolve()
        if path.exists():
            config = cls.from_file(path)
        else:
            config = cls.from_env()
        return config

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> None:
        """Write the current configuration to a JSON file.

        Args:
            path: Destination path. Defaults to :attr:`config_path`.
        """
        dest = Path(path) if path else self.config_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
