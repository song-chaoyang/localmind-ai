"""
Tests for LocalMind Core - Memory System
"""

import tempfile
import time
from datetime import datetime

import pytest

from localmind.core.memory import (
    LongTermMemory,
    MemoryEntry,
    MemoryManager,
    MemoryType,
    ShortTermMemory,
)


class TestMemoryEntry:
    """Tests for MemoryEntry."""

    def test_auto_id_generation(self):
        entry = MemoryEntry(content="test content")
        assert entry.id.startswith("mem_")
        assert len(entry.id) > 5

    def test_custom_id(self):
        entry = MemoryEntry(id="custom_id", content="test")
        assert entry.id == "custom_id"

    def test_to_dict(self):
        entry = MemoryEntry(content="test content", importance=0.8)
        data = entry.to_dict()
        assert data["content"] == "test content"
        assert data["importance"] == 0.8
        assert data["memory_type"] == "long_term"

    def test_from_dict(self):
        data = {
            "id": "test_id",
            "content": "test content",
            "metadata": {"key": "value"},
            "memory_type": "short_term",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "access_count": 5,
            "importance": 0.9,
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.id == "test_id"
        assert entry.content == "test content"
        assert entry.metadata == {"key": "value"}
        assert entry.memory_type == MemoryType.SHORT_TERM
        assert entry.importance == 0.9


class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    def test_add_message(self):
        memory = ShortTermMemory(max_messages=10)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")

        assert memory.message_count == 2

    def test_max_messages(self):
        memory = ShortTermMemory(max_messages=3)
        for i in range(5):
            memory.add_message("user", f"Message {i}")

        assert memory.message_count == 3

    def test_get_context(self):
        memory = ShortTermMemory()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi!")

        context = memory.get_context()
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"

    def test_get_context_with_system_prompt(self):
        memory = ShortTermMemory()
        memory.set_system_prompt("You are helpful.")
        memory.add_message("user", "Hello")

        context = memory.get_context()
        assert len(context) == 2
        assert context[0]["role"] == "system"

    def test_get_last_n(self):
        memory = ShortTermMemory()
        for i in range(5):
            memory.add_message("user", f"Message {i}")

        last_2 = memory.get_last_n(2)
        assert len(last_2) == 2

    def test_clear(self):
        memory = ShortTermMemory()
        memory.add_message("user", "Hello")
        memory.clear()

        assert memory.message_count == 0


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test_memory.db"
        self.memory = LongTermMemory(db_path=self.db_path)

    def teardown_method(self):
        import os
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve(self):
        entry = MemoryEntry(
            content="Important knowledge",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8,
        )
        entry_id = self.memory.store(entry)

        results = self.memory.retrieve()
        assert len(results) == 1
        assert results[0].content == "Important knowledge"

    def test_retrieve_by_type(self):
        self.memory.store(MemoryEntry(
            content="Short term info",
            memory_type=MemoryType.SHORT_TERM,
        ))
        self.memory.store(MemoryEntry(
            content="Long term info",
            memory_type=MemoryType.LONG_TERM,
        ))

        short_results = self.memory.retrieve(memory_type=MemoryType.SHORT_TERM)
        long_results = self.memory.retrieve(memory_type=MemoryType.LONG_TERM)

        assert len(short_results) == 1
        assert len(long_results) == 1

    def test_delete(self):
        entry = MemoryEntry(content="To be deleted")
        entry_id = self.memory.store(entry)

        assert self.memory.delete(entry_id) is True
        assert len(self.memory.retrieve()) == 0

    def test_update(self):
        entry = MemoryEntry(content="Original")
        entry_id = self.memory.store(entry)

        assert self.memory.update(entry_id, content="Updated") is True
        results = self.memory.retrieve()
        assert results[0].content == "Updated"

    def test_count(self):
        self.memory.store(MemoryEntry(content="Entry 1"))
        self.memory.store(MemoryEntry(content="Entry 2"))
        self.memory.store(MemoryEntry(content="Entry 3"))

        assert self.memory.count() == 3

    def test_clear(self):
        self.memory.store(MemoryEntry(content="Entry 1"))
        self.memory.store(MemoryEntry(content="Entry 2"))

        deleted = self.memory.clear()
        assert deleted == 2
        assert self.memory.count() == 0


class TestMemoryManager:
    """Tests for MemoryManager."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = MemoryManager(
            short_term_max=10,
            long_term_db=f"{self.temp_dir}/test.db",
        )

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_remember_and_recall(self):
        self.manager.remember("AI is transforming the world")
        self.manager.remember("Python is a great language")

        memories = self.manager.recall()
        assert len(memories) == 2

    def test_forget(self):
        entry_id = self.manager.remember("To be forgotten")
        assert self.manager.forget(entry_id) is True

    def test_add_message(self):
        self.manager.add_message("user", "Hello")
        self.manager.add_message("assistant", "Hi!")

        context = self.manager.get_context()
        assert len(context) == 2

    def test_clear_conversation(self):
        self.manager.add_message("user", "Hello")
        self.manager.clear_conversation()

        assert len(self.manager.get_context()) == 0

    def test_get_stats(self):
        self.manager.add_message("user", "Hello")
        self.manager.remember("Some knowledge")

        stats = self.manager.get_stats()
        assert stats["short_term_messages"] == 1
        assert stats["long_term_total"] == 1
