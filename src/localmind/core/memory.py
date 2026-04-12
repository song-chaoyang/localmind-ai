"""
Memory system for LocalMind.

Provides short-term, long-term, and semantic memory capabilities
for persistent context and knowledge management.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"    # Conversation context
    LONG_TERM = "long_term"      # Persistent knowledge
    SEMANTIC = "semantic"        # Vector-indexed knowledge
    EPISODIC = "episodic"        # Past interaction records


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.LONG_TERM
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID based on content hash."""
        content_hash = hashlib.sha256(
            f"{self.content}{self.created_at.isoformat()}".encode()
        ).hexdigest()[:16]
        return f"mem_{content_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryEntry:
        """Create from dictionary."""
        data = data.copy()
        data["memory_type"] = MemoryType(data["memory_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class ShortTermMemory:
    """
    Short-term memory for managing conversation context.

    Acts as a sliding window of recent messages, maintaining
    the most relevant context for the current conversation.
    """

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self._messages: List[Dict[str, Any]] = []
        self._system_prompt: Optional[str] = None

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to short-term memory."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._messages.append(message)

        # Trim to max size
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the conversation."""
        self._system_prompt = prompt

    def get_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """Get the current conversation context."""
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(
            {"role": m["role"], "content": m["content"]}
            for m in self._messages
        )
        return messages

    def get_last_n(self, n: int) -> List[Dict[str, Any]]:
        """Get the last n messages."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Clear short-term memory."""
        self._messages.clear()
        self._system_prompt = None

    @property
    def message_count(self) -> int:
        """Return the number of messages in memory."""
        return len(self._messages)


class LongTermMemory:
    """
    Long-term memory for persistent knowledge storage.

    Uses SQLite for reliable, structured storage of memories
    that persist across sessions.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path.home() / ".localmind" / "memory" / "long_term.db"
        )
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    memory_type TEXT DEFAULT 'long_term',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC)
            """)
            conn.commit()

    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, metadata, memory_type, created_at, updated_at,
                 access_count, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.content,
                json.dumps(entry.metadata),
                entry.memory_type.value,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.access_count,
                entry.importance,
            ))
            conn.commit()
        return entry.id

    def retrieve(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """Retrieve memory entries."""
        conditions = []
        params: List[Any] = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"""
                SELECT * FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
            """, params + [limit]).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def update(self, memory_id: str, content: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None,
               importance: Optional[float] = None) -> bool:
        """Update a memory entry."""
        updates = []
        params: List[Any] = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(memory_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self, memory_type: Optional[MemoryType] = None) -> int:
        """Count memory entries."""
        with sqlite3.connect(self.db_path) as conn:
            if memory_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?",
                    (memory_type.value,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0]

    def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear memory entries."""
        with sqlite3.connect(self.db_path) as conn:
            if memory_type:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE memory_type = ?",
                    (memory_type.value,)
                )
            else:
                cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return cursor.rowcount

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            memory_type=MemoryType(row["memory_type"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"],
            importance=row["importance"],
        )


class MemoryManager:
    """
    Unified memory manager that coordinates short-term and long-term memory.

    Provides a simple interface for storing, retrieving, and managing
    memories across different storage backends.
    """

    def __init__(
        self,
        short_term_max: int = 50,
        long_term_db: Optional[str] = None,
    ):
        self.short_term = ShortTermMemory(max_messages=short_term_max)
        self.long_term = LongTermMemory(db_path=long_term_db)
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """Store a memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            importance=importance,
        )
        return self.long_term.store(entry)

    def recall(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Retrieve memories."""
        return self.long_term.retrieve(
            query=query,
            memory_type=memory_type,
            limit=limit,
        )

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        return self.long_term.delete(memory_id)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to short-term memory."""
        self.short_term.add_message(role, content)

    def get_context(self) -> List[Dict[str, str]]:
        """Get the current conversation context."""
        return self.short_term.get_context()

    def clear_conversation(self) -> None:
        """Clear the current conversation."""
        self.short_term.clear()

    def clear_all(self) -> None:
        """Clear all memories."""
        self.short_term.clear()
        self.long_term.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "session_id": self._session_id,
            "short_term_messages": self.short_term.message_count,
            "long_term_total": self.long_term.count(),
            "long_term_by_type": {
                mt.value: self.long_term.count(MemoryType(mt))
                for mt in MemoryType
            },
        }
