"""
Tests for LocalMind Core - Configuration
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from localmind.core.config import (
    AgentConfig,
    Config,
    MemoryConfig,
    ModelConfig,
    PluginConfig,
    RAGConfig,
    ServerConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.name == "llama3"
        assert config.provider == "ollama"
        assert config.parameters["temperature"] == 0.7
        assert config.gpu_layers == -1

    def test_custom_values(self):
        config = ModelConfig(
            name="mistral",
            provider="ollama",
            parameters={"temperature": 0.5},
        )
        assert config.name == "mistral"
        assert config.parameters["temperature"] == 0.5


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.ui_port == 3000

    def test_custom_port(self):
        config = ServerConfig(port=9090)
        assert config.port == 9090


class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_values(self):
        config = MemoryConfig()
        assert config.short_term_max_messages == 50
        assert config.long_term_enabled is True
        assert config.long_term_backend == "sqlite"


class TestRAGConfig:
    """Tests for RAGConfig."""

    def test_default_values(self):
        config = RAGConfig()
        assert config.enabled is True
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.top_k == 5


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300
        assert config.sandbox_enabled is True
        assert len(config.default_tools) > 0


class TestConfig:
    """Tests for the main Config class."""

    def test_default_config(self):
        config = Config()
        assert config.debug is False
        assert config.log_level == "INFO"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.rag, RAGConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.plugin, PluginConfig)

    def test_from_dict(self):
        data = {
            "debug": True,
            "log_level": "DEBUG",
            "model": {"name": "mistral", "temperature": 0.5},
            "server": {"port": 9090},
        }
        config = Config._from_dict(data)
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.model.name == "mistral"
        assert config.server.port == 9090

    def test_to_dict(self):
        config = Config()
        data = config._to_dict()
        assert "debug" in data
        assert "model" in data
        assert "server" in data
        assert "memory" in data

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        try:
            config = Config()
            config.debug = True
            config.model.name = "mistral"
            config.save(temp_path)

            loaded = Config.from_file(temp_path)
            assert loaded.debug is True
            assert loaded.model.name == "mistral"
        finally:
            os.unlink(temp_path)

    def test_from_env(self):
        os.environ["LOCALMIND_DEBUG"] = "true"
        os.environ["LOCALMIND_LOG_LEVEL"] = "DEBUG"
        os.environ["LOCALMIND_MODEL"] = "qwen2"
        os.environ["LOCALMIND_PORT"] = "9999"

        try:
            config = Config.from_env()
            assert config.debug is True
            assert config.log_level == "DEBUG"
            assert config.model.name == "qwen2"
            assert config.server.port == 9999
        finally:
            os.environ.pop("LOCALMIND_DEBUG", None)
            os.environ.pop("LOCALMIND_LOG_LEVEL", None)
            os.environ.pop("LOCALMIND_MODEL", None)
            os.environ.pop("LOCALMIND_PORT", None)

    def test_ensure_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(data_dir=tmpdir)
            config.ensure_directories()

            expected_dirs = [
                Path(tmpdir) / "models",
                Path(tmpdir) / "memory",
                Path(tmpdir) / "plugins",
                Path(tmpdir) / "documents",
                Path(tmpdir) / "workflows",
                Path(tmpdir) / "logs",
            ]
            for d in expected_dirs:
                assert d.exists()
                assert d.is_dir()
