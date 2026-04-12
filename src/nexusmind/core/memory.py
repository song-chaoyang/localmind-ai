"""Persistent Memory System for NexusMind.

Provides short-term (sliding window) and long-term (SQLite-backed) memory
with semantic search, entity extraction, and user profile management.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from nexusmind.core.config import MemoryConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Base exception for memory operations."""


class MemoryNotFoundError(MemoryError):
    """Raised when a specific memory is not found."""


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single memory entry in long-term storage."""

    key: str
    value: str
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: list[float] = field(default_factory=list)


@dataclass
class ConversationSummary:
    """Summary of a conversation session."""

    id: str
    title: str
    messages_count: int
    created_at: float
    updated_at: float
    summary: str = ""


@dataclass
class Entity:
    """An extracted entity (person, project, tech, etc.)."""

    name: str
    entity_type: str  # person, project, tech, concept, location
    description: str = ""
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mention_count: int = 1
    attributes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Short-Term Memory
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """Sliding window buffer for recent conversation messages.

    Maintains a configurable number of recent messages for immediate context.
    Automatically evicts oldest messages when the buffer is full.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._messages: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a message to short-term memory.

        Args:
            role: Message role ('user', 'assistant', 'system').
            content: Message content.
            metadata: Optional metadata dict.

        Returns:
            The message ID.
        """
        import uuid

        msg_id = uuid.uuid4().hex[:12]
        self._messages[msg_id] = {
            "id": msg_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        # Evict oldest if over capacity
        while len(self._messages) > self.max_size:
            self._messages.popitem(last=False)
        return msg_id

    def get_recent(self, count: int | None = None) -> list[dict[str, Any]]:
        """Get the most recent messages.

        Args:
            count: Number of messages to return. Defaults to all.

        Returns:
            List of message dicts in chronological order.
        """
        messages = list(self._messages.values())
        if count is not None:
            messages = messages[-count:]
        return messages

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._messages.clear()

    @property
    def size(self) -> int:
        """Current number of messages in memory."""
        return len(self._messages)

    def to_list(self) -> list[dict[str, str]]:
        """Export messages as a list of role/content dicts.

        Returns:
            List of {'role': ..., 'content': ...} dicts.
        """
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages.values()
        ]


# ---------------------------------------------------------------------------
# Long-Term Memory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """SQLite-backed persistent long-term memory storage.

    Stores conversations, messages, user profiles, agent notes, and
    extracted entities. Provides semantic search capabilities.
    """

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_dir / "memory.db"
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                tags TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);

            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                mention_count INTEGER DEFAULT 1,
                attributes TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        """)
        conn.commit()

    # -- Memory CRUD --

    def store(self, key: str, value: str, category: str = "general",
              tags: list[str] | None = None, importance: float = 0.5) -> MemoryEntry:
        """Store a memory entry.

        Args:
            key: Unique identifier for the memory.
            value: The memory content.
            category: Category for organization.
            tags: Optional tags for search.
            importance: Importance score (0.0 - 1.0).

        Returns:
            The created MemoryEntry.
        """
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            """INSERT OR REPLACE INTO memories (key, value, category, tags, importance,
               created_at, updated_at, access_count)
               VALUES (?, ?, ?, ?, ?, COALESCE(
                   (SELECT created_at FROM memories WHERE key = ?), ?
               ), ?, COALESCE(
                   (SELECT access_count FROM memories WHERE key = ?), 0
               ))""",
            (
                key, value, category, json.dumps(tags or []),
                importance, key, now, now, key,
            ),
        )
        conn.commit()
        return MemoryEntry(
            key=key, value=value, category=category,
            tags=tags or [], importance=importance,
            created_at=now, updated_at=now, access_count=0,
        )

    def recall(self, key: str) -> MemoryEntry | None:
        """Recall a specific memory by key.

        Args:
            key: The memory key.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories by keyword matching.

        Args:
            query: Search query string.
            category: Optional category filter.
            limit: Maximum results to return.

        Returns:
            List of matching MemoryEntry objects.
        """
        conn = self._get_conn()
        query_lower = f"%{query.lower()}%"
        if category:
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE (LOWER(value) LIKE ? OR LOWER(key) LIKE ?)
                   AND category = ?
                   ORDER BY importance DESC, updated_at DESC
                   LIMIT ?""",
                (query_lower, query_lower, category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM memories
                   WHERE LOWER(value) LIKE ? OR LOWER(key) LIKE ?
                   ORDER BY importance DESC, updated_at DESC
                   LIMIT ?""",
                (query_lower, query_lower, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def forget(self, key: str) -> bool:
        """Delete a specific memory.

        Args:
            returns: True if the memory was found and deleted.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def list_all(
        self,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List all memories with optional filtering.

        Args:
            category: Optional category filter.
            limit: Maximum results.
            offset: Result offset for pagination.

        Returns:
            List of MemoryEntry objects.
        """
        conn = self._get_conn()
        if category:
            rows = conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (category, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def count(self, category: str | None = None) -> int:
        """Count memories.

        Args:
            category: Optional category filter.

        Returns:
            Number of memories.
        """
        conn = self._get_conn()
        if category:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE category = ?",
                (category,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"] if row else 0

    # -- Conversations --

    def save_conversation(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
        title: str = "",
    ) -> None:
        """Save a conversation with its messages.

        Args:
            conversation_id: Unique conversation identifier.
            messages: List of {'role': ..., 'content': ...} dicts.
            title: Optional conversation title.
        """
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            """INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at)
               VALUES (?, ?, COALESCE((SELECT created_at FROM conversations WHERE id = ?), ?), ?)""",
            (conversation_id, title, conversation_id, now, now),
        )
        for msg in messages:
            msg_id = msg.get("id", f"{conversation_id}_{len(messages)}")
            conn.execute(
                """INSERT OR REPLACE INTO messages (id, conversation_id, role, content, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, msg["role"], msg["content"], now),
            )
        conn.commit()

    def get_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages from a conversation.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            List of message dicts.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row.get("metadata") or "{}"),
            }
            for row in rows
        ]

    def list_conversations(self, limit: int = 50) -> list[ConversationSummary]:
        """List recent conversations.

        Args:
            limit: Maximum conversations to return.

        Returns:
            List of ConversationSummary objects.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT c.*, COUNT(m.id) as msg_count
               FROM conversations c
               LEFT JOIN messages m ON c.id = m.conversation_id
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            ConversationSummary(
                id=row["id"],
                title=row["title"],
                messages_count=row["msg_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                summary=row["summary"] or "",
            )
            for row in rows
        ]

    # -- User Profile --

    def get_user_profile(self) -> dict[str, str]:
        """Get the user profile as a dictionary.

        Returns:
            Dictionary of profile key-value pairs.
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def update_user_profile(self, updates: dict[str, str]) -> None:
        """Update user profile fields.

        Args:
            updates: Dictionary of profile fields to update.
        """
        conn = self._get_conn()
        now = time.time()
        for key, value in updates.items():
            conn.execute(
                """INSERT OR REPLACE INTO user_profile (key, value, updated_at)
                   VALUES (?, ?, ?)""",
                (key, value, now),
            )
        conn.commit()

    # -- Entities --

    def upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity.

        Args:
            entity: The Entity to store.
        """
        conn = self._get_conn()
        now = time.time()
        existing = conn.execute(
            "SELECT mention_count FROM entities WHERE name = ?", (entity.name,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE entities SET entity_type=?, description=?, last_seen=?,
                   mention_count=?, attributes=? WHERE name=?""",
                (
                    entity.entity_type, entity.description, now,
                    existing["mention_count"] + 1,
                    json.dumps(entity.attributes), entity.name,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO entities (name, entity_type, description, first_seen,
                   last_seen, mention_count, attributes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity.name, entity.entity_type, entity.description,
                    entity.first_seen, now, 1, json.dumps(entity.attributes),
                ),
            )
        conn.commit()

    def get_entities(
        self, entity_type: str | None = None, limit: int = 100
    ) -> list[Entity]:
        """Get stored entities.

        Args:
            entity_type: Optional type filter.
            limit: Maximum results.

        Returns:
            List of Entity objects.
        """
        conn = self._get_conn()
        if entity_type:
            rows = conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY mention_count DESC LIMIT ?",
                (entity_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Entity(
                name=row["name"],
                entity_type=row["entity_type"],
                description=row["description"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                mention_count=row["mention_count"],
                attributes=json.loads(row.get("attributes") or "{}"),
            )
            for row in rows
        ]

    # -- Export / Import --

    def export_data(self) -> dict[str, Any]:
        """Export all memory data as a dictionary.

        Returns:
            Dictionary containing all memory data.
        """
        conn = self._get_conn()
        memories = [asdict(m) for m in self.list_all(limit=100000)]
        conversations = [
            {
                "id": c.id,
                "title": c.title,
                "messages_count": c.messages_count,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "summary": c.summary,
            }
            for c in self.list_conversations(limit=100000)
        ]
        profile = self.get_user_profile()
        entities = [asdict(e) for e in self.get_entities(limit=100000)]

        return {
            "version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memories": memories,
            "conversations": conversations,
            "user_profile": profile,
            "entities": entities,
        }

    def import_data(self, data: dict[str, Any]) -> int:
        """Import memory data from a dictionary.

        Args:
            data: Dictionary with memory data (as exported by export_data).

        Returns:
            Number of memories imported.
        """
        count = 0
        for mem in data.get("memories", []):
            self.store(
                key=mem["key"],
                value=mem["value"],
                category=mem.get("category", "general"),
                tags=mem.get("tags", []),
                importance=mem.get("importance", 0.5),
            )
            count += 1

        profile = data.get("user_profile", {})
        if profile:
            self.update_user_profile(profile)

        for ent in data.get("entities", []):
            entity = Entity(
                name=ent["name"],
                entity_type=ent.get("entity_type", "concept"),
                description=ent.get("description", ""),
                attributes=ent.get("attributes", {}),
            )
            self.upsert_entity(entity)

        return count

    # -- Helpers --

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            key=row["key"],
            value=row["value"],
            category=row["category"],
            tags=json.loads(row["tags"] or "[]"),
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            embedding=json.loads(row["embedding"] or "[]"),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Extract entities from conversation text using pattern matching.

    Identifies people, projects, technologies, and other named entities
    from natural language text.
    """

    # Patterns for entity extraction
    _PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("tech", re.compile(
            r'\b(?:Python|JavaScript|TypeScript|React|Vue|Angular|Django|Flask|FastAPI|'
            r'Node\.?js|Docker|Kubernetes|AWS|GCP|Azure|PostgreSQL|MySQL|Redis|MongoDB|'
            r'Git|Linux|TensorFlow|PyTorch|Rust|Go|Java|C\+\+|Swift|Kotlin)\b',
            re.IGNORECASE,
        )),
        ("project", re.compile(
            r'(?:project|repo|app|service|module|package)\s+[\'"]?([\w\-]+)[\'"]?',
            re.IGNORECASE,
        )),
        ("person", re.compile(
            r'(?:@|mentioned|said|told|asked)\s+([\w]+(?:\s+[\w]+)?)',
            re.IGNORECASE,
        )),
        ("concept", re.compile(
            r'(?:API|REST|GraphQL|WebSocket|OAuth|JWT|CI\/CD|TDD|BDD|SOLID|DRY|'
            r'MVC|MVVM|microservice|monolith|serverless|edge computing|LLM|RAG|'
            r'fine-?tun(?:ing|e)|prompt engineering)',
            re.IGNORECASE,
        )),
    ]

    @classmethod
    def extract(cls, text: str) -> list[Entity]:
        """Extract entities from text.

        Args:
            text: The text to extract entities from.

        Returns:
            List of Entity objects found in the text.
        """
        entities: list[Entity] = []
        seen: set[str] = set()

        for entity_type, pattern in cls._PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1) if match.lastindex else match.group(0)
                name = name.strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    entities.append(
                        Entity(
                            name=name,
                            entity_type=entity_type,
                            description=f"Extracted from: {text[:100]}",
                        )
                    )

        return entities


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Unified memory management interface.

    Combines short-term and long-term memory with entity extraction,
    context building, and export/import capabilities.

    Example::

        memory = MemoryManager(MemoryConfig())
        memory.remember("user_name", "Alice")
        results = memory.recall("Alice")
        context = memory.get_context("What was I working on?")
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.short_term = ShortTermMemory(max_size=self.config.max_short_term)
        self.long_term = LongTermMemory(persist_dir=self.config.persist_dir)
        self._entity_extractor = EntityExtractor()

    def remember(
        self,
        key: str,
        value: str,
        category: str = "general",
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Store important information in long-term memory.

        Args:
            key: Unique identifier.
            value: The information to remember.
            category: Category for organization.
            tags: Optional tags.
            importance: Importance score (0.0-1.0).

        Returns:
            The stored MemoryEntry.
        """
        entry = self.long_term.store(key, value, category, tags, importance)
        logger.debug("Remembered: %s = %s", key, truncate(value, 50))
        return entry

    def recall(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search long-term memory for relevant information.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching MemoryEntry objects.
        """
        results = self.long_term.search(query, limit=limit)
        # Update access counts
        for entry in results:
            self.long_term.store(
                entry.key, entry.value, entry.category,
                entry.tags, entry.importance,
            )
        return results

    def forget(self, key: str) -> bool:
        """Delete a specific memory.

        Args:
            key: The memory key to delete.

        Returns:
            True if the memory was found and deleted.
        """
        return self.long_term.forget(key)

    def add_message(self, role: str, content: str) -> str:
        """Add a message to short-term memory and extract entities.

        Args:
            role: Message role.
            content: Message content.

        Returns:
            The message ID.
        """
        msg_id = self.short_term.add(role, content)

        # Auto-extract entities
        if self.config.auto_extract_entities:
            entities = self._entity_extractor.extract(content)
            for entity in entities:
                self.long_term.upsert_entity(entity)

        return msg_id

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Build a context window from relevant memories.

        Combines recent short-term messages with relevant long-term memories
        to create a context string for LLM prompts.

        Args:
            query: The current query to find relevant context for.
            max_tokens: Approximate maximum token budget.

        Returns:
            A formatted context string.
        """
        parts: list[str] = []

        # Search long-term memory for relevant context
        memories = self.recall(query, limit=5)
        if memories:
            memory_text = "\n".join(
                f"- [{m.category}] {m.key}: {m.value}" for m in memories
            )
            parts.append(f"## Relevant Memories\n{memory_text}")

        # Add recent short-term context
        recent = self.short_term.get_recent(count=10)
        if recent:
            chat_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent
            )
            parts.append(f"## Recent Conversation\n{chat_text}")

        context = "\n\n".join(parts)

        # Rough token truncation (~4 chars per token)
        if len(context) > max_tokens * 4:
            context = context[: max_tokens * 4] + "\n...[truncated]"

        return context

    def get_user_profile(self) -> dict[str, str]:
        """Get the user profile.

        Returns:
            Dictionary of profile key-value pairs.
        """
        return self.long_term.get_user_profile()

    def update_user_profile(self, updates: dict[str, str]) -> None:
        """Update user profile fields.

        Args:
            updates: Dictionary of profile fields to update.
        """
        self.long_term.update_user_profile(updates)

    def get_entities(self, entity_type: str | None = None) -> list[Entity]:
        """Get tracked entities.

        Args:
            entity_type: Optional type filter.

        Returns:
            List of Entity objects.
        """
        return self.long_term.get_entities(entity_type)

    def export_memories(self) -> dict[str, Any]:
        """Export all memory data.

        Returns:
            Dictionary containing all memory data.
        """
        return self.long_term.export_data()

    def import_memories(self, data: dict[str, Any]) -> int:
        """Import memory data.

        Args:
            data: Dictionary with memory data.

        Returns:
            Number of memories imported.
        """
        return self.long_term.import_data(data)

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with memory statistics.
        """
        return {
            "short_term_messages": self.short_term.size,
            "short_term_capacity": self.short_term.max_size,
            "long_term_memories": self.long_term.count(),
            "conversations": len(self.long_term.list_conversations(limit=100000)),
            "entities": len(self.long_term.get_entities()),
            "user_profile_fields": len(self.long_term.get_user_profile()),
        }

    def close(self) -> None:
        """Close all memory connections."""
        self.long_term.close()


def truncate(text: str, max_len: int) -> str:
    """Truncate text for logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
