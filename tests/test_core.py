"""Tests for OpenMind core modules."""

import tempfile
from pathlib import Path

import pytest


class TestConfig:
    def test_default_config(self):
        from openmind.core.config import Config
        config = Config()
        assert config.model.name == "llama3"
        assert config.server.port == 3000

    def test_save_and_load(self):
        from openmind.core.config import Config
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            config = Config()
            config.model.name = "mistral"
            config.save(path)
            loaded = Config.from_file(path)
            assert loaded.model.name == "mistral"
        finally:
            Path(path).unlink()


class TestMemory:
    def test_short_term(self):
        from openmind.core.memory import ShortTermMemory
        mem = ShortTermMemory(max_messages=5)
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi!")
        assert mem.message_count == 2
        ctx = mem.get_context()
        assert len(ctx) == 2

    def test_short_term_overflow(self):
        from openmind.core.memory import ShortTermMemory
        mem = ShortTermMemory(max_messages=3)
        for i in range(10):
            mem.add_message("user", f"msg{i}")
        assert mem.message_count == 3

    def test_long_term(self):
        from openmind.core.memory import LongTermMemory, MemoryEntry, MemoryType
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(db_path=f"{tmpdir}/test.db")
            entry = MemoryEntry(content="test knowledge", memory_type=MemoryType.LONG_TERM)
            eid = ltm.store(entry)
            results = ltm.retrieve()
            assert len(results) == 1
            assert ltm.delete(eid) is True


class TestTextSplitter:
    def test_split(self):
        from openmind.core.rag import TextSplitter, ChunkConfig
        splitter = TextSplitter(ChunkConfig(chunk_size=50))
        chunks = splitter.split_text(" ".join(["word"] * 100))
        assert len(chunks) > 1

    def test_empty(self):
        from openmind.core.rag import TextSplitter
        splitter = TextSplitter()
        assert splitter.split_text("") == []


class TestVectorStore:
    def test_add_and_search(self):
        from openmind.core.rag import VectorStore, Document
        store = VectorStore()
        store.add([Document(content="Python programming language")])
        results = store.search("programming", top_k=1)
        assert len(results) > 0


class TestHelpers:
    def test_format_bytes(self):
        from openmind.utils.helpers import format_bytes
        assert "KB" in format_bytes(1024)
        assert "MB" in format_bytes(1048576)

    def test_generate_id(self):
        from openmind.utils.helpers import generate_id
        gid = generate_id("test")
        assert gid.startswith("test_")

    def test_truncate(self):
        from openmind.utils.helpers import truncate_text
        assert truncate_text("hello world", 5) == "he..."
