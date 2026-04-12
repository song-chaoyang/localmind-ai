"""
Memory subsystem for OpenMind.

Provides short-term (in-memory sliding window) and long-term (SQLite-backed)
conversation memory, unified behind a :class:`MemoryManager` interface.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A single conversation message.

    Attributes:
        role: One of ``"system"``, ``"user"``, or ``"assistant"``.
        content: The text body of the message.
        timestamp: Unix epoch seconds when the message was created.
        metadata: Optional arbitrary key-value metadata.
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Reconstruct a :class:`Message` from a dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class ShortTermMemory:
    """In-memory sliding-window conversation buffer.

    Keeps the most recent *max_messages* messages so that context is always
    available for the model without unbounded memory growth.

    Args:
        max_messages: Maximum number of messages to retain.

    Example::

        stm = ShortTermMemory(max_messages=10)
        stm.add("user", "Hello!")
        stm.add("assistant", "Hi there!")
        print(stm.get_messages())
    """

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: List[Message] = []
        self.max_messages: int = max_messages

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Append a message, evicting the oldest if the buffer is full.

        Args:
            role: Message role (``"user"``, ``"assistant"``, or ``"system"``).
            content: Message text.
            metadata: Optional metadata dictionary.
        """
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def get_messages(self) -> List[Message]:
        """Return all messages currently in the buffer, oldest first."""
        return list(self._messages)

    def get_context(self) -> List[Dict[str, str]]:
        """Return messages formatted for the Ollama chat API.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        """Remove all messages from the buffer."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ShortTermMemory(messages={len(self._messages)}, max={self.max_messages})"


class LongTermMemory:
    """SQLite-backed persistent conversation storage.

    Conversations are stored in a local SQLite database and survive across
    application restarts.

    Args:
        db_path: Path to the SQLite database file. The file is created
            automatically if it does not exist.

    Example::

        ltm = LongTermMemory("~/.openmind/memory.db")
        ltm.store("user", "Hello!")
        ltm.store("assistant", "Hi there!")
        history = ltm.search("Hello")
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create the messages table if it does not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp DESC)
            """
        )
        self._conn.commit()

    def store(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a message to the database.

        Args:
            role: Message role.
            content: Message text.
            metadata: Optional metadata dictionary.

        Returns:
            The integer row ID of the inserted message.
        """
        cursor = self._conn.execute(
            "INSERT INTO messages (role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (
                role,
                content,
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_recent(self, limit: int = 50) -> List[Message]:
        """Retrieve the most recent messages.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            A list of :class:`Message` objects, newest last.
        """
        rows = self._conn.execute(
            "SELECT role, content, timestamp, metadata FROM messages ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Message(
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def search(self, query: str, limit: int = 10) -> List[Message]:
        """Search message content for *query* using a simple LIKE match.

        Args:
            query: Search string.
            limit: Maximum results to return.

        Returns:
            Matching :class:`Message` objects.
        """
        rows = self._conn.execute(
            "SELECT role, content, timestamp, metadata FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            Message(
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def clear(self) -> None:
        """Delete all stored messages."""
        self._conn.execute("DELETE FROM messages")
        self._conn.commit()

    def count(self) -> int:
        """Return the total number of stored messages."""
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM messages").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"LongTermMemory(db_path={self.db_path!s})"


class MemoryManager:
    """Unified interface for short-term and long-term memory.

    Every incoming message is stored in both the short-term buffer (for
    immediate context) and, optionally, in long-term storage (for persistence).

    Args:
        short_term_max: Maximum messages in the short-term buffer.
        long_term_path: Path to the SQLite database. Pass ``None`` to disable
            long-term persistence.

    Example::

        mm = MemoryManager(short_term_max=20, long_term_path="~/.openmind/memory.db")
        mm.add("user", "What is Python?")
        mm.add("assistant", "Python is a programming language.")
        context = mm.get_context()
    """

    def __init__(
        self,
        short_term_max: int = 50,
        long_term_path: Optional[str | Path] = None,
    ) -> None:
        self.short_term = ShortTermMemory(max_messages=short_term_max)
        self.long_term: Optional[LongTermMemory] = None
        if long_term_path:
            self.long_term = LongTermMemory(long_term_path)

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a message in both memory stores.

        Args:
            role: Message role.
            content: Message text.
            metadata: Optional metadata.
        """
        self.short_term.add(role=role, content=content, metadata=metadata)
        if self.long_term:
            self.long_term.store(role=role, content=content, metadata=metadata)

    def get_context(self) -> List[Dict[str, str]]:
        """Return the current conversation context for the model.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts from short-term memory.
        """
        return self.short_term.get_context()

    def get_history(self, limit: int = 100) -> List[Message]:
        """Retrieve conversation history.

        If long-term memory is available the history is read from there;
        otherwise the short-term buffer is used.

        Args:
            limit: Maximum messages to return.

        Returns:
            A list of :class:`Message` objects.
        """
        if self.long_term:
            return self.long_term.get_recent(limit=limit)
        return self.short_term.get_messages()

    def search(self, query: str, limit: int = 10) -> List[Message]:
        """Search conversation history.

        Falls back to an in-memory search if long-term storage is disabled.

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            Matching :class:`Message` objects.
        """
        if self.long_term:
            return self.long_term.search(query=query, limit=limit)
        # Fallback: simple substring match on short-term messages
        return [
            m
            for m in self.short_term.get_messages()
            if query.lower() in m.content.lower()
        ][:limit]

    def clear(self) -> None:
        """Clear all memory (both short-term and long-term)."""
        self.short_term.clear()
        if self.long_term:
            self.long_term.clear()

    def close(self) -> None:
        """Release resources (close the SQLite connection if open)."""
        if self.long_term:
            self.long_term.close()

    def __repr__(self) -> str:
        return (
            f"MemoryManager(short_term={self.short_term}, "
            f"long_term={'enabled' if self.long_term else 'disabled'})"
        )
