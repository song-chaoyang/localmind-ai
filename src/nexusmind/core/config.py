"""Configuration management for NexusMind.

Provides dataclass-based configuration with support for multiple LLM providers,
memory settings, scheduler configuration, and notification integrations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional


_DEFAULT_DATA_DIR = Path.home() / ".nexusmind"


@dataclass
class ModelConfig:
    """Configuration for the active LLM model."""

    default_model: str = "llama3"
    provider: str = "ollama"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    """Configuration for the web server."""

    host: str = "127.0.0.1"
    port: int = 8420
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None


@dataclass
class MemoryConfig:
    """Configuration for the persistent memory system."""

    max_short_term: int = 50
    max_long_term: int = 10000
    persist_dir: str = str(_DEFAULT_DATA_DIR / "memory")
    auto_extract_entities: bool = True
    similarity_threshold: float = 0.7


@dataclass
class SchedulerConfig:
    """Configuration for the offline task scheduler."""

    enabled: bool = True
    max_concurrent: int = 3
    persist_dir: str = str(_DEFAULT_DATA_DIR / "scheduler")
    default_model: Optional[str] = None
    retry_failed: bool = True
    max_retries: int = 3


@dataclass
class NotificationConfig:
    """Configuration for notification integrations."""

    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook: Optional[str] = None
    slack_token: Optional[str] = None
    slack_channel: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[str] = None


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""

    name: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    models: list[str] = field(default_factory=list)
    timeout: float = 120.0
    max_retries: int = 3


class Config:
    """Main configuration class for NexusMind.

    Aggregates all sub-configurations and supports loading from files,
    environment variables, and saving/loading from the default data directory.

    Example::

        config = Config.from_env()
        config = Config.from_file("config.json")
        config.save()
    """

    def __init__(
        self,
        model: ModelConfig | None = None,
        server: ServerConfig | None = None,
        memory: MemoryConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        notification: NotificationConfig | None = None,
        providers: dict[str, ProviderConfig] | None = None,
    ) -> None:
        self.model = model or ModelConfig()
        self.server = server or ServerConfig()
        self.memory = memory or MemoryConfig()
        self.scheduler = scheduler or SchedulerConfig()
        self.notification = notification or NotificationConfig()
        self.providers = providers or self._default_providers()
        self._config_path: Path | None = None

    @staticmethod
    def _default_providers() -> dict[str, ProviderConfig]:
        """Create default provider configurations."""
        return {
            "ollama": ProviderConfig(
                name="ollama",
                base_url="http://localhost:11434",
            ),
            "openai": ProviderConfig(
                name="openai",
                base_url="https://api.openai.com/v1",
                api_key=os.environ.get("OPENAI_API_KEY"),
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                base_url="https://api.anthropic.com/v1",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            ),
            "openrouter": ProviderConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
        }

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            A Config instance populated from the file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = cls._from_dict(data)
        config._config_path = path
        return config

    @classmethod
    def from_env(cls) -> Config:
        """Create configuration from environment variables.

        Environment variable mapping:
            NEXUSMIND_CONFIG - Path to config file (takes priority)
            NEXUSMIND_MODEL - Default model name
            NEXUSMIND_PROVIDER - Active provider name
            NEXUSMIND_HOST - Server host
            NEXUSMIND_PORT - Server port
            OPENAI_API_KEY - OpenAI API key
            ANTHROPIC_API_KEY - Anthropic API key
            OPENROUTER_API_KEY - OpenRouter API key

        Returns:
            A Config instance.
        """
        config_path = os.environ.get("NEXUSMIND_CONFIG")
        if config_path:
            return cls.from_file(config_path)

        model = ModelConfig()
        model.default_model = os.environ.get("NEXUSMIND_MODEL", model.default_model)
        model.provider = os.environ.get("NEXUSMIND_PROVIDER", model.provider)

        server = ServerConfig()
        server.host = os.environ.get("NEXUSMIND_HOST", server.host)
        if port_env := os.environ.get("NEXUSMIND_PORT"):
            server.port = int(port_env)

        memory = MemoryConfig()
        if persist := os.environ.get("NEXUSMIND_MEMORY_DIR"):
            memory.persist_dir = persist

        scheduler = SchedulerConfig()
        if sched_dir := os.environ.get("NEXUSMIND_SCHEDULER_DIR"):
            scheduler.persist_dir = sched_dir

        notification = NotificationConfig()
        notification.telegram_token = os.environ.get("TELEGRAM_TOKEN")
        notification.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        notification.discord_webhook = os.environ.get("DISCORD_WEBHOOK")
        notification.slack_token = os.environ.get("SLACK_TOKEN")

        providers = cls._default_providers()
        if key := os.environ.get("OPENAI_API_KEY"):
            providers["openai"].api_key = key
        if key := os.environ.get("ANTHROPIC_API_KEY"):
            providers["anthropic"].api_key = key
        if key := os.environ.get("OPENROUTER_API_KEY"):
            providers["openrouter"].api_key = key

        return cls(
            model=model,
            server=server,
            memory=memory,
            scheduler=scheduler,
            notification=notification,
            providers=providers,
        )

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            A Config instance.
        """
        def _dataclass_from_dict(cls_dc, d):
            if not d:
                return cls_dc()
            valid = {f.name: d[f.name] for f in fields(cls_dc) if f.name in d}
            return cls_dc(**valid)

        model = _dataclass_from_dict(ModelConfig, data.get("model"))
        server = _dataclass_from_dict(ServerConfig, data.get("server"))
        memory = _dataclass_from_dict(MemoryConfig, data.get("memory"))
        scheduler = _dataclass_from_dict(SchedulerConfig, data.get("scheduler"))
        notification = _dataclass_from_dict(NotificationConfig, data.get("notification"))

        providers: dict[str, ProviderConfig] = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = _dataclass_from_dict(ProviderConfig, pdata)

        return cls(
            model=model,
            server=server,
            memory=memory,
            scheduler=scheduler,
            notification=notification,
            providers=providers or None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "model": asdict(self.model),
            "server": asdict(self.server),
            "memory": asdict(self.memory),
            "scheduler": asdict(self.scheduler),
            "notification": asdict(self.notification),
            "providers": {name: asdict(p) for name, p in self.providers.items()},
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Save configuration to a JSON file.

        Args:
            path: Destination path. Defaults to ~/.nexusmind/config.json.

        Returns:
            The path where the config was saved.
        """
        if path is None:
            path = _DEFAULT_DATA_DIR / "config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        self._config_path = path
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load configuration from a JSON file.

        Args:
            path: Source path. Defaults to ~/.nexusmind/config.json.

        Returns:
            A Config instance.

        Raises:
            FileNotFoundError: If no config file exists at the path.
        """
        if path is None:
            path = _DEFAULT_DATA_DIR / "config.json"
        return cls.from_file(path)

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'ollama').

        Returns:
            ProviderConfig for the specified provider.

        Raises:
            KeyError: If the provider is not configured.
        """
        if provider_name not in self.providers:
            raise KeyError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[provider_name]

    def ensure_data_dirs(self) -> list[Path]:
        """Ensure all required data directories exist.

        Returns:
            List of created/existing directory paths.
        """
        dirs = [
            Path(self.memory.persist_dir),
            Path(self.scheduler.persist_dir),
            _DEFAULT_DATA_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def __repr__(self) -> str:
        return (
            f"Config(model={self.model.default_model!r}, "
            f"provider={self.model.provider!r}, "
            f"port={self.server.port})"
        )
