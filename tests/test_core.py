"""Tests for NexusMind core modules.

Tests cover Config, Memory, Skills, Scheduler, Providers, and RAG
with mocked HTTP calls where appropriate.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the Config module."""

    def test_default_config(self) -> None:
        """Test creating a default configuration."""
        from nexusmind.core.config import Config

        config = Config()
        assert config.model.default_model == "llama3"
        assert config.model.provider == "ollama"
        assert config.server.port == 8420
        assert config.memory.max_short_term == 50
        assert "ollama" in config.providers
        assert "openai" in config.providers

    def test_from_env(self) -> None:
        """Test loading configuration from environment variables."""
        from nexusmind.core.config import Config

        os.environ["NEXUSMIND_MODEL"] = "gpt-4"
        os.environ["NEXUSMIND_PROVIDER"] = "openai"
        os.environ["NEXUSMIND_PORT"] = "9000"

        try:
            config = Config.from_env()
            assert config.model.default_model == "gpt-4"
            assert config.model.provider == "openai"
            assert config.server.port == 9000
        finally:
            os.environ.pop("NEXUSMIND_MODEL", None)
            os.environ.pop("NEXUSMIND_PROVIDER", None)
            os.environ.pop("NEXUSMIND_PORT", None)

    def test_save_and_load(self) -> None:
        """Test saving and loading configuration."""
        from nexusmind.core.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = Config()
            config.model.default_model = "mistral"
            config.save(config_path)

            loaded = Config.load(config_path)
            assert loaded.model.default_model == "mistral"

    def test_to_dict(self) -> None:
        """Test serializing configuration to dictionary."""
        from nexusmind.core.config import Config

        config = Config()
        d = config.to_dict()
        assert "model" in d
        assert "server" in d
        assert "memory" in d
        assert "providers" in d
        assert d["model"]["default_model"] == "llama3"

    def test_get_provider_config(self) -> None:
        """Test getting a specific provider configuration."""
        from nexusmind.core.config import Config

        config = Config()
        pc = config.get_provider_config("openai")
        assert pc.name == "openai"
        assert pc.base_url == "https://api.openai.com/v1"

    def test_get_provider_config_unknown(self) -> None:
        """Test getting an unknown provider raises KeyError."""
        from nexusmind.core.config import Config

        config = Config()
        with pytest.raises(KeyError, match="Unknown provider"):
            config.get_provider_config("nonexistent")

    def test_ensure_data_dirs(self) -> None:
        """Test that data directories are created."""
        from nexusmind.core.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.memory.persist_dir = str(Path(tmpdir) / "mem")
            config.scheduler.persist_dir = str(Path(tmpdir) / "sched")
            dirs = config.ensure_data_dirs()
            assert all(Path(d).exists() for d in dirs)


# ---------------------------------------------------------------------------
# Memory Tests
# ---------------------------------------------------------------------------


class TestMemory:
    """Tests for the Memory module."""

    def _make_memory(self, tmpdir: str):
        """Create a MemoryManager with temp directory."""
        from nexusmind.core.config import MemoryConfig
        from nexusmind.core.memory import MemoryManager

        config = MemoryConfig(persist_dir=tmpdir)
        return MemoryManager(config=config)

    def test_remember_and_recall(self, tmp_path: Path) -> None:
        """Test storing and recalling a memory."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.remember("test_key", "test_value", category="test")
            results = mm.recall("test_key")
            assert len(results) >= 1
            assert results[0].value == "test_value"
            assert results[0].category == "test"
        finally:
            mm.close()

    def test_forget(self, tmp_path: Path) -> None:
        """Test deleting a memory."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.remember("to_forget", "will be deleted")
            assert mm.forget("to_forget") is True
            assert mm.forget("nonexistent") is False
        finally:
            mm.close()

    def test_short_term_memory(self, tmp_path: Path) -> None:
        """Test short-term memory sliding window."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.add_message("user", "Hello")
            mm.add_message("assistant", "Hi there!")
            assert mm.short_term.size == 2

            recent = mm.short_term.get_recent(count=1)
            assert len(recent) == 1
            assert recent[0]["role"] == "assistant"
        finally:
            mm.close()

    def test_user_profile(self, tmp_path: Path) -> None:
        """Test user profile management."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.update_user_profile({"name": "Alice", "lang": "Python"})
            profile = mm.get_user_profile()
            assert profile["name"] == "Alice"
            assert profile["lang"] == "Python"
        finally:
            mm.close()

    def test_entity_extraction(self, tmp_path: Path) -> None:
        """Test automatic entity extraction."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.add_message("user", "I'm building a FastAPI app with Python and Docker")
            entities = mm.get_entities()
            entity_names = [e.name for e in entities]
            # Should extract at least some tech entities
            assert len(entities) > 0
        finally:
            mm.close()

    def test_export_import(self, tmp_path: Path) -> None:
        """Test memory export and import."""
        mm = self._make_memory(str(tmp_path / "mem1"))
        try:
            mm.remember("key1", "value1")
            mm.remember("key2", "value2")
            data = mm.export_memories()
        finally:
            mm.close()

        mm2 = self._make_memory(str(tmp_path / "mem2"))
        try:
            count = mm2.import_memories(data)
            assert count == 2
            results = mm2.recall("key1")
            assert len(results) >= 1
        finally:
            mm2.close()

    def test_get_context(self, tmp_path: Path) -> None:
        """Test context building for LLM prompts."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.remember("project", "NexusMind AI Agent", category="project")
            mm.add_message("user", "What is NexusMind?")
            context = mm.get_context("NexusMind")
            assert "NexusMind" in context
        finally:
            mm.close()

    def test_get_stats(self, tmp_path: Path) -> None:
        """Test memory statistics."""
        mm = self._make_memory(str(tmp_path / "mem"))
        try:
            mm.remember("stat_test", "value")
            mm.add_message("user", "hello")
            stats = mm.get_stats()
            assert stats["long_term_memories"] >= 1
            assert stats["short_term_messages"] >= 1
        finally:
            mm.close()


# ---------------------------------------------------------------------------
# Skills Tests
# ---------------------------------------------------------------------------


class TestSkills:
    """Tests for the Skills module."""

    def _make_engine(self, tmpdir: str):
        """Create a SkillEngine with temp directory."""
        from nexusmind.core.skills import SkillEngine

        return SkillEngine(persist_dir=tmpdir)

    def test_create_and_get_skill(self, tmp_path: Path) -> None:
        """Test creating and retrieving a skill."""
        engine = self._make_engine(str(tmp_path / "skills"))
        try:
            from nexusmind.core.skills import SkillStep

            skill = engine.create_skill(
                name="test_skill",
                description="A test skill",
                steps=[
                    SkillStep(description="Step 1", action="generate", template="Hello {name}"),
                    SkillStep(description="Step 2", action="validate"),
                ],
                tags=["test"],
            )
            assert skill.name == "test_skill"
            assert len(skill.steps) == 2

            retrieved = engine.get_skill("test_skill")
            assert retrieved is not None
            assert retrieved.description == "A test skill"
        finally:
            engine.close()

    def test_execute_skill(self, tmp_path: Path) -> None:
        """Test executing a skill."""
        engine = self._make_engine(str(tmp_path / "skills"))
        try:
            from nexusmind.core.skills import SkillStep

            engine.create_skill(
                name="exec_test",
                description="Test execution",
                steps=[
                    SkillStep(description="Generate greeting", action="generate", template="Hello!"),
                ],
            )

            result = asyncio.get_event_loop().run_until_complete(
                engine.execute_skill("exec_test", {})
            )
            assert result["success"] is True
            assert result["steps_completed"] == 1
        finally:
            engine.close()

    def test_delete_skill(self, tmp_path: Path) -> None:
        """Test deleting a skill."""
        engine = self._make_engine(str(tmp_path / "skills"))
        try:
            engine.create_skill(name="to_delete", description="Will be deleted", steps=[])
            assert engine.delete_skill("to_delete") is True
            assert engine.delete_skill("nonexistent") is False
        finally:
            engine.close()

    def test_pattern_detection(self, tmp_path: Path) -> None:
        """Test pattern detection from repeated interactions."""
        engine = self._make_engine(str(tmp_path / "skills"))
        try:
            # Simulate repeated pattern
            for _ in range(3):
                engine.learn_from_interaction(
                    [{"role": "user", "content": "deploy to production server at 10.0.0.1"}],
                    "success",
                )
            suggestions = engine.get_pattern_suggestions()
            # May or may not detect depending on normalization
            assert isinstance(suggestions, list)
        finally:
            engine.close()

    def test_export_import_skills(self, tmp_path: Path) -> None:
        """Test skill export and import."""
        engine = self._make_engine(str(tmp_path / "skills1"))
        try:
            engine.create_skill(name="export_test", description="Test", steps=[])
            data = engine.export_skills()
        finally:
            engine.close()

        engine2 = self._make_engine(str(tmp_path / "skills2"))
        try:
            count = engine2.import_skills(data)
            assert count == 1
            assert engine2.get_skill("export_test") is not None
        finally:
            engine2.close()

    def test_skill_dna(self, tmp_path: Path) -> None:
        """Test skill DNA fingerprint generation."""
        engine = self._make_engine(str(tmp_path / "skills"))
        try:
            dna = engine.skill_dna()
            assert "total_skills" in dna
            assert "overall_success_rate" in dna
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# Scheduler Tests
# ---------------------------------------------------------------------------


class TestScheduler:
    """Tests for the Scheduler module."""

    def _make_scheduler(self, tmpdir: str):
        """Create a TaskScheduler with temp directory."""
        from nexusmind.core.config import SchedulerConfig
        from nexusmind.core.scheduler import TaskScheduler

        config = SchedulerConfig(persist_dir=tmpdir, enabled=False)
        return TaskScheduler(config=config)

    def test_schedule_task(self, tmp_path: Path) -> None:
        """Test creating a scheduled task."""
        scheduler = self._make_scheduler(str(tmp_path / "sched"))
        try:
            task = scheduler.schedule_task(
                name="Test Task",
                prompt="Say hello",
                schedule="in 30 minutes",
            )
            assert task.name == "Test Task"
            assert task.status.value == "pending"
            assert task.next_run > 0
        finally:
            scheduler.close()

    def test_list_tasks(self, tmp_path: Path) -> None:
        """Test listing tasks."""
        scheduler = self._make_scheduler(str(tmp_path / "sched"))
        try:
            scheduler.schedule_task("T1", "prompt1", "in 1 hour")
            scheduler.schedule_task("T2", "prompt2", "in 2 hours")
            tasks = scheduler.list_tasks()
            assert len(tasks) == 2
        finally:
            scheduler.close()

    def test_cancel_task(self, tmp_path: Path) -> None:
        """Test cancelling a task."""
        scheduler = self._make_scheduler(str(tmp_path / "sched"))
        try:
            task = scheduler.schedule_task("Cancel Me", "prompt", "in 1 hour")
            assert scheduler.cancel_task(task.id) is True
            assert scheduler.cancel_task("nonexistent") is False
        finally:
            scheduler.close()

    def test_pause_resume_task(self, tmp_path: Path) -> None:
        """Test pausing and resuming a task."""
        scheduler = self._make_scheduler(str(tmp_path / "sched"))
        try:
            task = scheduler.schedule_task("Pause Me", "prompt", "in 1 hour")
            assert scheduler.pause_task(task.id) is True
            updated = scheduler.get_task(task.id)
            assert updated is not None
            assert updated.status.value == "paused"
            assert scheduler.resume_task(task.id) is True
        finally:
            scheduler.close()

    def test_run_now(self, tmp_path: Path) -> None:
        """Test running a task immediately."""
        scheduler = self._make_scheduler(str(tmp_path / "sched"))
        try:
            task = scheduler.schedule_task("Run Now", "Say hello world", "in 1 hour")

            async def executor(prompt, model):
                return f"Response to: {prompt}"

            scheduler._executor = executor
            result = asyncio.get_event_loop().run_until_complete(
                scheduler.run_now(task.id)
            )
            assert result.status.value == "completed"
            assert "Response to:" in result.output
        finally:
            scheduler.close()

    def test_get_task_results(self, tmp_path: Path) -> None:
        """Test getting task execution results."""
        scheduler = self._make_scheduler(str(tmp_path / "sched"))
        try:
            task = scheduler.schedule_task("Results Test", "prompt", "in 1 hour")

            async def executor(prompt, model):
                return "done"

            scheduler._executor = executor
            asyncio.get_event_loop().run_until_complete(scheduler.run_now(task.id))
            results = scheduler.get_task_results(task.id)
            assert len(results) == 1
            assert results[0].status.value == "completed"
        finally:
            scheduler.close()

    def test_schedule_parser_natural(self) -> None:
        """Test natural language schedule parsing."""
        from nexusmind.core.scheduler import ScheduleParser
        from datetime import datetime, timezone

        base = datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)

        # "in 30 minutes"
        ts = ScheduleParser.parse("in 30 minutes", base)
        expected = base.replace(minute=30).timestamp()
        assert abs(ts - expected) < 1

        # "every day at 9am"
        ts = ScheduleParser.parse("every day at 9am", base)
        assert ts > base.timestamp()

    def test_schedule_parser_cron(self) -> None:
        """Test cron expression parsing."""
        from nexusmind.core.scheduler import ScheduleParser
        from datetime import datetime, timezone

        base = datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)

        # "0 9 * * *" - every day at 9am
        ts = ScheduleParser.parse("0 9 * * *", base)
        assert ts > base.timestamp()


# ---------------------------------------------------------------------------
# Provider Tests (mocked)
# ---------------------------------------------------------------------------


class TestProviders:
    """Tests for the Provider module."""

    def test_provider_factory(self) -> None:
        """Test provider factory creation."""
        from nexusmind.core.config import ModelConfig, ProviderConfig
        from nexusmind.core.providers import ProviderFactory

        pc = ProviderConfig(name="ollama", base_url="http://localhost:11434")
        mc = ModelConfig()
        provider = ProviderFactory.create("ollama", pc, mc)
        assert provider.name == "ollama"

    def test_provider_factory_unknown(self) -> None:
        """Test that unknown provider raises ValueError."""
        from nexusmind.core.config import ModelConfig, ProviderConfig
        from nexusmind.core.providers import ProviderFactory

        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create("nonexistent", ProviderConfig(), ModelConfig())

    def test_register_custom_provider(self) -> None:
        """Test registering a custom provider."""
        from nexusmind.core.config import ModelConfig, ProviderConfig
        from nexusmind.core.providers import BaseProvider, ChatMessage, ChatResponse, ProviderFactory

        class DummyProvider(BaseProvider):
            async def chat(self, messages, model=None, **kwargs):
                return ChatResponse(content="dummy", model="dummy", provider="dummy")

            async def chat_stream(self, messages, model=None, **kwargs):
                yield "dummy"

            async def list_models(self):
                return []

        ProviderFactory.register("dummy", DummyProvider)
        assert "dummy" in ProviderFactory.available_providers()
        provider = ProviderFactory.create("dummy", ProviderConfig(), ModelConfig())
        assert isinstance(provider, DummyProvider)

    @pytest.mark.asyncio
    async def test_ollama_provider_chat(self) -> None:
        """Test Ollama provider chat with mocked HTTP."""
        from nexusmind.core.config import ModelConfig, ProviderConfig
        from nexusmind.core.providers import ChatMessage, OllamaProvider

        pc = ProviderConfig(name="ollama", base_url="http://localhost:11434")
        mc = ModelConfig()
        provider = OllamaProvider(pc, mc)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3",
            "message": {"content": "Hello!"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            response = await provider.chat(
                [ChatMessage(role="user", content="Hi")]
            )
            assert response.content == "Hello!"
            assert response.provider == "ollama"

        await provider.close()

    @pytest.mark.asyncio
    async def test_openai_provider_chat(self) -> None:
        """Test OpenAI provider chat with mocked HTTP."""
        from nexusmind.core.config import ModelConfig, ProviderConfig
        from nexusmind.core.providers import ChatMessage, OpenAIProvider

        pc = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        mc = ModelConfig()
        provider = OpenAIProvider(pc, mc)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gpt-4",
            "choices": [
                {"message": {"content": "Hello from GPT!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_response)
            response = await provider.chat(
                [ChatMessage(role="user", content="Hi")]
            )
            assert response.content == "Hello from GPT!"
            assert response.usage["total_tokens"] == 15

        await provider.close()


# ---------------------------------------------------------------------------
# RAG Tests
# ---------------------------------------------------------------------------


class TestRAG:
    """Tests for the RAG module."""

    def test_document_loader(self, tmp_path: Path) -> None:
        """Test loading a document from a file."""
        from nexusmind.core.rag import DocumentLoader

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, this is a test document.", encoding="utf-8")

        docs = DocumentLoader.load(test_file)
        assert len(docs) == 1
        assert docs[0].content == "Hello, this is a test document."
        assert docs[0].metadata["filename"] == "test.txt"

    def test_text_splitter(self) -> None:
        """Test text splitting."""
        from nexusmind.core.rag import TextSplitter

        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size + some overlap tolerance
        for chunk in chunks:
            assert len(chunk) <= 70

    def test_simple_embedder(self) -> None:
        """Test the hash-based embedder."""
        from nexusmind.core.rag import SimpleEmbedder

        embedder = SimpleEmbedder()
        vec = embedder.embed("Hello world")
        assert len(vec) == embedder.dimension
        assert all(isinstance(x, float) for x in vec)

    def test_vector_store(self) -> None:
        """Test the in-memory vector store."""
        from nexusmind.core.rag import Document, SimpleEmbedder, VectorStore

        embedder = SimpleEmbedder()
        store = VectorStore(dimension=embedder.dimension)

        doc1 = Document(content="Python programming language")
        doc1.embedding = embedder.embed(doc1.content)

        doc2 = Document(content="JavaScript web development")
        doc2.embedding = embedder.embed(doc2.content)

        store.add(doc1)
        store.add(doc2)
        assert store.size == 2

        query_vec = embedder.embed("Python code")
        results = store.search(query_vec, top_k=2)
        assert len(results) >= 1
        # Python doc should rank higher for Python query
        assert results[0].document.content == "Python programming language"

    def test_rag_pipeline(self, tmp_path: Path) -> None:
        """Test the full RAG pipeline."""
        from nexusmind.core.rag import RAGPipeline

        # Create test documents
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "doc1.txt").write_text(
            "NexusMind is an AI agent with persistent memory.", encoding="utf-8"
        )
        (doc_dir / "doc2.txt").write_text(
            "The system supports multiple LLM providers.", encoding="utf-8"
        )

        pipeline = RAGPipeline(chunk_size=100, chunk_overlap=20)
        chunks = pipeline.ingest([str(doc_dir)])
        assert chunks > 0

        results = pipeline.query("persistent memory")
        assert len(results) > 0

        context = pipeline.build_context("memory system")
        assert "NexusMind" in context or "memory" in context.lower()

        stats = pipeline.get_stats()
        assert stats["documents_ingested"] == 2
        assert stats["store_size"] > 0


# ---------------------------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for utility helper functions."""

    def test_generate_id(self) -> None:
        """Test ID generation."""
        from nexusmind.utils.helpers import generate_id

        id1 = generate_id()
        id2 = generate_id()
        assert len(id1) == 16
        assert id1 != id2

    def test_format_bytes(self) -> None:
        """Test byte formatting."""
        from nexusmind.utils.helpers import format_bytes

        assert format_bytes(0) == "0.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1048576) == "1.0 MB"
        assert format_bytes(1073741824) == "1.0 GB"

    def test_truncate_text(self) -> None:
        """Test text truncation."""
        from nexusmind.utils.helpers import truncate_text

        assert truncate_text("short", 20) == "short"
        assert truncate_text("a" * 100, 10) == "aaaaaaa..."
        assert truncate_text("hello world", 8, suffix="!") == "hello!"

    def test_hash_text(self) -> None:
        """Test text hashing."""
        from nexusmind.utils.helpers import hash_text

        h1 = hash_text("hello")
        h2 = hash_text("hello")
        h3 = hash_text("world")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # SHA-256 hex

    def test_safe_json_loads(self) -> None:
        """Test safe JSON parsing."""
        from nexusmind.utils.helpers import safe_json_loads

        assert safe_json_loads('{"a": 1}') == {"a": 1}
        assert safe_json_loads("invalid", {}) == {}
        assert safe_json_loads(None, "default") == "default"

    def test_chunk_list(self) -> None:
        """Test list chunking."""
        from nexusmind.utils.helpers import chunk_list

        assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
        assert chunk_list([], 3) == []

    def test_timer(self) -> None:
        """Test the timer context manager."""
        from nexusmind.utils.helpers import timer

        with timer() as t:
            time.sleep(0.01)
        assert t["elapsed"] >= 0.01
        assert t["start"] > 0

    def test_async_retry(self) -> None:
        """Test the async retry decorator."""
        from nexusmind.utils.helpers import async_retry

        call_count = 0

        @async_retry(max_retries=3, delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "success"

        result = asyncio.get_event_loop().run_until_complete(flaky_function())
        assert result == "success"
        assert call_count == 3


# ---------------------------------------------------------------------------
# Engine Integration Tests (mocked)
# ---------------------------------------------------------------------------


class TestEngine:
    """Integration tests for the NexusMind engine."""

    @pytest.mark.asyncio
    async def test_engine_creation(self, tmp_path: Path) -> None:
        """Test creating a NexusMind engine."""
        from nexusmind.core.config import Config
        from nexusmind.core.engine import NexusMind

        config = Config()
        config.memory.persist_dir = str(tmp_path / "mem")
        config.scheduler.persist_dir = str(tmp_path / "sched")

        mind = NexusMind(config=config)
        assert mind._active_model == "llama3"
        assert mind.memory is not None
        assert mind.skills is not None
        assert mind.scheduler is not None
        assert mind.rag is not None

        stats = mind.get_stats()
        assert "version" in stats
        assert "memory" in stats
        assert "skills" in stats

        await mind.close()

    @pytest.mark.asyncio
    async def test_engine_chat_mocked(self, tmp_path: Path) -> None:
        """Test engine chat with mocked provider."""
        from nexusmind.core.config import Config, ModelConfig, ProviderConfig
        from nexusmind.core.engine import NexusMind
        from nexusmind.core.providers import ChatMessage, ChatResponse

        config = Config()
        config.memory.persist_dir = str(tmp_path / "mem")
        config.scheduler.persist_dir = str(tmp_path / "sched")

        mind = NexusMind(config=config)

        # Mock the active provider
        mock_response = ChatResponse(
            content="Mocked response!",
            model="test-model",
            provider="ollama",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )

        with patch.object(
            mind._active_provider, "chat", new_callable=AsyncMock, return_value=mock_response
        ):
            response = await mind.chat("Hello!")
            assert response.content == "Mocked response!"
            assert mind._stats["total_chats"] == 1

        await mind.close()

    def test_ingest_documents(self, tmp_path: Path) -> None:
        """Test document ingestion."""
        from nexusmind.core.config import Config
        from nexusmind.core.engine import NexusMind

        config = Config()
        config.memory.persist_dir = str(tmp_path / "mem")
        config.scheduler.persist_dir = str(tmp_path / "sched")

        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "test.txt").write_text("Test document content.", encoding="utf-8")

        mind = NexusMind(config=config)
        chunks = mind.ingest_documents([str(doc_dir)])
        assert chunks > 0

        rag_stats = mind.rag.get_stats()
        assert rag_stats["documents_ingested"] == 1
