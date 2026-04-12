"""
Configuration management for LocalMind.

Handles loading, saving, and accessing configuration settings
from environment variables, config files, and CLI arguments.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Default configuration directory
DEFAULT_CONFIG_DIR = Path.home() / ".localmind"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str = "llama3"
    provider: str = "ollama"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "context_window": 8192,
    })
    gpu_layers: int = -1  # -1 means auto
    quantization: str = "auto"  # auto, q4_0, q4_1, q5_0, q5_1, q8_0


@dataclass
class ServerConfig:
    """Configuration for the API server."""

    host: str = "0.0.0.0"
    port: int = 8080
    ui_port: int = 3000
    workers: int = 1
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""

    short_term_max_messages: int = 50
    long_term_enabled: bool = True
    long_term_backend: str = "sqlite"  # sqlite, chromadb, redis
    long_term_max_entries: int = 10000
    semantic_memory_enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store: str = "chromadb"  # chromadb, faiss, qdrant
    top_k: int = 5
    reranker_enabled: bool = False


@dataclass
class AgentConfig:
    """Configuration for the agent system."""

    max_iterations: int = 10
    timeout_seconds: int = 300
    default_tools: List[str] = field(default_factory=lambda: [
        "web_search",
        "file_reader",
        "file_writer",
        "code_executor",
        "calculator",
    ])
    sandbox_enabled: bool = True
    verbose_logging: bool = False


@dataclass
class PluginConfig:
    """Configuration for the plugin system."""

    auto_load: bool = True
    plugin_dir: str = str(DEFAULT_CONFIG_DIR / "plugins")
    marketplace_url: str = "https://plugins.localmind.ai"
    hot_reload: bool = True


@dataclass
class Config:
    """
    Main configuration for LocalMind.

    Configuration is loaded from (in order of priority):
    1. CLI arguments
    2. Environment variables (LOCALMIND_*)
    3. Config file (~/.localmind/config.json)
    4. Default values
    """

    # Core settings
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = str(DEFAULT_CONFIG_DIR)
    config_dir: str = str(DEFAULT_CONFIG_DIR)

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    plugin: PluginConfig = field(default_factory=PluginConfig)

    @classmethod
    def from_file(cls, path: Union[str, Path] = DEFAULT_CONFIG_FILE) -> Config:
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        config = cls()

        # Core settings
        config.debug = os.getenv("LOCALMIND_DEBUG", "false").lower() == "true"
        config.log_level = os.getenv("LOCALMIND_LOG_LEVEL", config.log_level)
        config.data_dir = os.getenv("LOCALMIND_DATA_DIR", config.data_dir)

        # Model settings
        if model_name := os.getenv("LOCALMIND_MODEL"):
            config.model.name = model_name
        if provider := os.getenv("LOCALMIND_MODEL_PROVIDER"):
            config.model.provider = provider

        # Server settings
        if port := os.getenv("LOCALMIND_PORT"):
            config.server.port = int(port)
        if host := os.getenv("LOCALMIND_HOST"):
            config.server.host = host

        return config

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> Config:
        """
        Load configuration with full priority chain.

        Priority: CLI args > env vars > config file > defaults
        """
        # Start with defaults
        config = cls()

        # Load from file
        if path:
            config = cls.from_file(path)
        elif DEFAULT_CONFIG_FILE.exists():
            config = cls.from_file(DEFAULT_CONFIG_FILE)

        # Override with environment variables
        env_config = cls.from_env()
        for attr_name in ["debug", "log_level", "data_dir"]:
            env_val = getattr(env_config, attr_name)
            default_val = getattr(cls(), attr_name)
            if env_val != default_val:
                setattr(config, attr_name, env_val)

        # Override model settings from env
        if os.getenv("LOCALMIND_MODEL"):
            config.model.name = os.getenv("LOCALMIND_MODEL")
        if os.getenv("LOCALMIND_MODEL_PROVIDER"):
            config.model.provider = os.getenv("LOCALMIND_MODEL_PROVIDER")

        return config

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to a JSON file."""
        path = Path(path or DEFAULT_CONFIG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Config:
        """Create configuration from a dictionary."""
        config = cls()

        # Core settings
        for key in ["debug", "log_level", "data_dir", "config_dir"]:
            if key in data:
                setattr(config, key, data[key])

        # Model config
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        # Server config
        if "server" in data:
            for key, value in data["server"].items():
                if hasattr(config.server, key):
                    setattr(config.server, key, value)

        # Memory config
        if "memory" in data:
            for key, value in data["memory"].items():
                if hasattr(config.memory, key):
                    setattr(config.memory, key, value)

        # RAG config
        if "rag" in data:
            for key, value in data["rag"].items():
                if hasattr(config.rag, key):
                    setattr(config.rag, key, value)

        # Agent config
        if "agent" in data:
            for key, value in data["agent"].items():
                if hasattr(config.agent, key):
                    setattr(config.agent, key, value)

        # Plugin config
        if "plugin" in data:
            for key, value in data["plugin"].items():
                if hasattr(config.plugin, key):
                    setattr(config.plugin, key, value)

        return config

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        dirs = [
            Path(self.data_dir),
            Path(self.data_dir) / "models",
            Path(self.data_dir) / "memory",
            Path(self.data_dir) / "plugins",
            Path(self.data_dir) / "documents",
            Path(self.data_dir) / "workflows",
            Path(self.data_dir) / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
