"""
Tests for LocalMind Core - Engine
"""

import tempfile

import pytest

from localmind.core.config import Config
from localmind.core.engine import LocalMind


class TestLocalMind:
    """Tests for the main LocalMind engine."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(data_dir=self.temp_dir)

    def test_initialization(self):
        mind = LocalMind(config=self.config)
        assert mind is not None
        assert mind.config == self.config
        assert mind._model_loaded is False

    def test_initialization_creates_directories(self):
        mind = LocalMind(config=self.config)
        from pathlib import Path
        assert Path(self.temp_dir).exists()
        assert Path(self.temp_dir / "models").exists()
        assert Path(self.temp_dir / "memory").exists()

    def test_register_agent(self):
        from localmind.agents import ResearchAgent

        mind = LocalMind(config=self.config)
        agent = ResearchAgent(engine=mind)
        mind.register_agent("researcher", agent)

        assert "researcher" in mind.list_agents()
        assert mind.get_agent("researcher") is agent

    def test_register_multiple_agents(self):
        from localmind.agents import ResearchAgent, CodeAgent, DataAgent

        mind = LocalMind(config=self.config)
        mind.register_agent("researcher", ResearchAgent(engine=mind))
        mind.register_agent("coder", CodeAgent(engine=mind))
        mind.register_agent("analyst", DataAgent(engine=mind))

        assert len(mind.list_agents()) == 3

    def test_get_nonexistent_agent(self):
        mind = LocalMind(config=self.config)
        assert mind.get_agent("nonexistent") is None

    def test_execute_nonexistent_agent(self):
        mind = LocalMind(config=self.config)

        with pytest.raises(ValueError, match="not found"):
            import asyncio
            asyncio.run(mind.execute_agent("nonexistent", "some task"))

    def test_chat_without_model(self):
        mind = LocalMind(config=self.config)

        with pytest.raises(RuntimeError, match="No model loaded"):
            import asyncio
            asyncio.run(mind.chat("Hello"))

    def test_clear_conversation(self):
        mind = LocalMind(config=self.config)
        mind.memory.add_message("user", "Hello")
        mind.clear_conversation()

        assert mind.memory.short_term.message_count == 0

    def test_get_stats(self):
        mind = LocalMind(config=self.config)
        stats = mind.get_stats()

        assert stats["version"] == "0.1.0"
        assert stats["model_loaded"] is False
        assert "memory" in stats

    def test_repr(self):
        mind = LocalMind(config=self.config)
        assert "LocalMind" in repr(mind)
        assert "no model" in repr(mind)

    def test_shutdown(self):
        mind = LocalMind(config=self.config)
        mind.shutdown()
        assert mind._running is False

    def test_load_model_no_ollama(self):
        mind = LocalMind(config=self.config)

        with pytest.raises(RuntimeError):
            mind.load_model("llama3")

    def test_list_available_models_no_ollama(self):
        mind = LocalMind(config=self.config)
        models = mind.list_available_models()
        assert models == []

    def test_unload_model_when_none_loaded(self):
        mind = LocalMind(config=self.config)
        mind.unload_model()  # Should not raise
        assert mind._model_loaded is False
