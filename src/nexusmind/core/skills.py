"""Auto Skill Evolution System for NexusMind.

Provides automatic pattern detection, skill learning from interactions,
and skill execution. Skills are stored in both JSON and SQLite for
persistence and searchability.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillError(Exception):
    """Base exception for skill operations."""


class SkillNotFoundError(SkillError):
    """Raised when a skill is not found."""


class SkillExecutionError(SkillError):
    """Raised when skill execution fails."""


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class SkillStep:
    """A single step within a skill."""

    description: str
    action: str  # 'generate', 'search', 'execute', 'validate', 'notify'
    template: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """A reusable skill learned from user interactions.

    Attributes:
        name: Unique skill name.
        description: Human-readable description.
        trigger_pattern: Regex pattern that triggers this skill.
        steps: Ordered list of steps to execute.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
        usage_count: Number of times the skill has been executed.
        success_count: Number of successful executions.
        failure_count: Number of failed executions.
        tags: Categorization tags.
        source: How the skill was learned ('auto', 'manual').
    """

    name: str
    description: str
    trigger_pattern: str = ""
    steps: list[SkillStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    tags: list[str] = field(default_factory=list)
    source: str = "manual"
    version: int = 1

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this skill.

        Returns:
            Success rate as a float between 0.0 and 1.0.
        """
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    def to_dict(self) -> dict[str, Any]:
        """Serialize skill to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "trigger_pattern": self.trigger_pattern,
            "steps": [asdict(s) for s in self.steps],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "tags": self.tags,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        """Deserialize a skill from a dictionary.

        Args:
            data: Dictionary with skill data.

        Returns:
            A Skill instance.
        """
        steps = [
            SkillStep(**s) for s in data.get("steps", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            trigger_pattern=data.get("trigger_pattern", ""),
            steps=steps,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            tags=data.get("tags", []),
            source=data.get("source", "manual"),
            version=data.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Pattern Detector
# ---------------------------------------------------------------------------


class PatternDetector:
    """Detects repeated patterns in user interactions.

    When a user repeats a similar pattern 3+ times, suggests creating
    a reusable skill from it.
    """

    def __init__(self, min_occurrences: int = 3) -> None:
        self.min_occurrences = min_occurrences
        self._patterns: dict[str, list[dict[str, Any]]] = {}

    def observe(self, text: str, context: dict[str, Any] | None = None) -> None:
        """Observe a user interaction for pattern detection.

        Args:
            text: The user's input text.
            context: Optional context information.
        """
        normalized = self._normalize(text)
        pattern_key = self._extract_pattern(normalized)

        if pattern_key:
            if pattern_key not in self._patterns:
                self._patterns[pattern_key] = []
            self._patterns[pattern_key].append({
                "text": text,
                "context": context or {},
                "timestamp": time.time(),
            })

    def detect_repeated(self) -> list[dict[str, Any]]:
        """Detect patterns that have been repeated enough times.

        Returns:
            List of detected repeated patterns with metadata.
        """
        suggestions = []
        for pattern_key, occurrences in self._patterns.items():
            if len(occurrences) >= self.min_occurrences:
                suggestions.append({
                    "pattern": pattern_key,
                    "occurrences": len(occurrences),
                    "examples": [o["text"] for o in occurrences[:3]],
                    "last_seen": occurrences[-1]["timestamp"],
                })
        return sorted(suggestions, key=lambda x: x["occurrences"], reverse=True)

    def _normalize(self, text: str) -> str:
        """Normalize text for pattern matching.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        text = text.strip().lower()
        # Replace specific values with placeholders
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', text)
        text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', '<EMAIL>', text)
        text = re.sub(r'\bhttps?://\S+\b', '<URL>', text)
        text = re.sub(r'\b\d+\b', '<NUM>', text)
        text = re.sub(r'["\'][\w\s]+["\']', '<QUOTED>', text)
        return text

    def _extract_pattern(self, normalized: str) -> str | None:
        """Extract a pattern key from normalized text.

        Args:
            normalized: Normalized text.

        Returns:
            A pattern key string or None.
        """
        if len(normalized) < 10:
            return None
        # Use the first 60 chars as a rough pattern key
        return normalized[:60]

    def clear(self) -> None:
        """Clear all observed patterns."""
        self._patterns.clear()


# ---------------------------------------------------------------------------
# Skill Store
# ---------------------------------------------------------------------------


class SkillStore:
    """SQLite-backed persistent storage for skills."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_dir / "skills.db"
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                trigger_pattern TEXT DEFAULT '',
                steps TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                source TEXT DEFAULT 'manual',
                version INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS skill_executions (
                id TEXT PRIMARY KEY,
                skill_name TEXT NOT NULL,
                input_summary TEXT,
                output_summary TEXT,
                success INTEGER DEFAULT 0,
                duration_ms INTEGER,
                timestamp REAL NOT NULL,
                FOREIGN KEY (skill_name) REFERENCES skills(name)
            );

            CREATE INDEX IF NOT EXISTS idx_skill_exec ON skill_executions(skill_name);
        """)
        conn.commit()

    def save(self, skill: Skill) -> None:
        """Save or update a skill.

        Args:
            skill: The Skill to save.
        """
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO skills
               (name, description, trigger_pattern, steps, created_at, updated_at,
                usage_count, success_count, failure_count, tags, source, version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.name, skill.description, skill.trigger_pattern,
                json.dumps([asdict(s) for s in skill.steps]),
                skill.created_at, skill.updated_at,
                skill.usage_count, skill.success_count, skill.failure_count,
                json.dumps(skill.tags), skill.source, skill.version,
            ),
        )
        conn.commit()

    def get(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill if found, None otherwise.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        return self._row_to_skill(row)

    def list_all(self, tag: str | None = None, limit: int = 100) -> list[Skill]:
        """List all skills.

        Args:
            tag: Optional tag filter.
            limit: Maximum results.

        Returns:
            List of Skill objects.
        """
        conn = self._get_conn()
        if tag:
            rows = conn.execute(
                """SELECT * FROM skills WHERE tags LIKE ?
                   ORDER BY usage_count DESC LIMIT ?""",
                (f'%"{tag}"%', limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM skills ORDER BY usage_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_skill(row) for row in rows]

    def delete(self, name: str) -> bool:
        """Delete a skill.

        Args:
            name: Skill name.

        Returns:
            True if the skill was found and deleted.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM skills WHERE name = ?", (name,))
        conn.commit()
        return cursor.rowcount > 0

    def record_execution(
        self,
        skill_name: str,
        success: bool,
        duration_ms: int,
        input_summary: str = "",
        output_summary: str = "",
    ) -> None:
        """Record a skill execution for analytics.

        Args:
            skill_name: Name of the executed skill.
            success: Whether the execution was successful.
            duration_ms: Execution duration in milliseconds.
            input_summary: Brief summary of input.
            output_summary: Brief summary of output.
        """
        import uuid

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO skill_executions
               (id, skill_name, input_summary, output_summary, success, duration_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex[:16], skill_name,
                input_summary, output_summary,
                1 if success else 0, duration_ms, time.time(),
            ),
        )
        # Update skill counters
        if success:
            conn.execute(
                "UPDATE skills SET usage_count = usage_count + 1, success_count = success_count + 1, updated_at = ? WHERE name = ?",
                (time.time(), skill_name),
            )
        else:
            conn.execute(
                "UPDATE skills SET usage_count = usage_count + 1, failure_count = failure_count + 1, updated_at = ? WHERE name = ?",
                (time.time(), skill_name),
            )
        conn.commit()

    def _row_to_skill(self, row: sqlite3.Row) -> Skill:
        """Convert a database row to a Skill."""
        steps_data = json.loads(row.get("steps") or "[]")
        steps = [SkillStep(**s) for s in steps_data]
        return Skill(
            name=row["name"],
            description=row["description"],
            trigger_pattern=row.get("trigger_pattern", ""),
            steps=steps,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            usage_count=row["usage_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            tags=json.loads(row.get("tags") or "[]"),
            source=row.get("source", "manual"),
            version=row.get("version", 1),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Skill Engine
# ---------------------------------------------------------------------------


class SkillEngine:
    """Auto Skill Evolution Engine.

    Learns reusable patterns from user interactions, manages skills,
    and provides skill suggestions based on context.

    Example::

        engine = SkillEngine("~/.nexusmind/skills")
        engine.learn_from_interaction(messages, "success")
        suggestions = engine.suggest_skills("deploy to production")
        result = await engine.execute_skill("deploy", context)
    """

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store = SkillStore(persist_dir=self.persist_dir)
        self._pattern_detector = PatternDetector(min_occurrences=3)
        self._interaction_history: list[dict[str, Any]] = []

    def learn_from_interaction(
        self,
        conversation: list[dict[str, str]],
        outcome: str,
    ) -> list[dict[str, Any]]:
        """Analyze an interaction and potentially learn a new skill.

        Args:
            conversation: List of {'role': ..., 'content': ...} dicts.
            outcome: 'success', 'failure', or 'partial'.

        Returns:
            List of pattern suggestions (may be empty).
        """
        # Observe user messages for pattern detection
        for msg in conversation:
            if msg["role"] == "user":
                self._pattern_detector.observe(
                    msg["content"],
                    {"outcome": outcome},
                )

        # Store in interaction history
        self._interaction_history.append({
            "conversation": conversation,
            "outcome": outcome,
            "timestamp": time.time(),
        })

        # Keep history manageable
        if len(self._interaction_history) > 1000:
            self._interaction_history = self._interaction_history[-500:]

        return self._pattern_detector.detect_repeated()

    def create_skill(
        self,
        name: str,
        description: str,
        steps: list[SkillStep],
        trigger_pattern: str = "",
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> Skill:
        """Create a new skill.

        Args:
            name: Unique skill name.
            description: Human-readable description.
            steps: Ordered list of skill steps.
            trigger_pattern: Regex pattern for auto-triggering.
            tags: Categorization tags.
            source: How the skill was created.

        Returns:
            The created Skill.
        """
        skill = Skill(
            name=name,
            description=description,
            trigger_pattern=trigger_pattern,
            steps=steps,
            tags=tags or [],
            source=source,
        )
        self._store.save(skill)
        logger.info("Created skill: %s", name)
        return skill

    async def execute_skill(
        self,
        skill_name: str,
        context: dict[str, Any],
        executor: Any = None,
    ) -> dict[str, Any]:
        """Execute a skill with the given context.

        Args:
            skill_name: Name of the skill to execute.
            context: Execution context with variables.
            executor: Optional callable for executing steps.

        Returns:
            Execution result dictionary.

        Raises:
            SkillNotFoundError: If the skill doesn't exist.
            SkillExecutionError: If execution fails.
        """
        skill = self._store.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")

        start_time = time.time()
        results: list[dict[str, Any]] = []
        success = True
        error: Exception | None = None

        try:
            for i, step in enumerate(skill.steps):
                step_result: dict[str, Any] = {
                    "step": i + 1,
                    "action": step.action,
                    "description": step.description,
                    "status": "pending",
                }

                try:
                    if executor is not None:
                        step_output = await executor(step, context)
                        step_result["output"] = step_output
                        step_result["status"] = "completed"
                    else:
                        # Default: just record the step
                        step_result["output"] = {
                            "action": step.action,
                            "template": step.template.format(**context) if step.template else "",
                        }
                        step_result["status"] = "completed"

                    results.append(step_result)
                except Exception as step_error:
                    step_result["status"] = "failed"
                    step_result["error"] = str(step_error)
                    results.append(step_result)
                    success = False
                    error = step_error
                    break

        except Exception as e:
            success = False
            error = e

        duration_ms = int((time.time() - start_time) * 1000)

        self._store.record_execution(
            skill_name=skill_name,
            success=success,
            duration_ms=duration_ms,
            input_summary=str(context)[:200],
            output_summary=str(results)[:200],
        )

        if not success and error:
            raise SkillExecutionError(
                f"Skill '{skill_name}' failed at step {len(results)}: {error}"
            ) from error

        return {
            "skill": skill_name,
            "success": success,
            "duration_ms": duration_ms,
            "steps_completed": len([r for r in results if r["status"] == "completed"]),
            "steps_total": len(skill.steps),
            "results": results,
        }

    def suggest_skills(self, context: str) -> list[Skill]:
        """Suggest skills relevant to the current context.

        Matches context against trigger patterns and tags.

        Args:
            context: The current context or query string.

        Returns:
            List of relevant Skills, sorted by relevance.
        """
        all_skills = self._store.list_all(limit=100)
        scored: list[tuple[float, Skill]] = []

        for skill in all_skills:
            score = 0.0
            context_lower = context.lower()

            # Check trigger pattern
            if skill.trigger_pattern:
                try:
                    if re.search(skill.trigger_pattern, context, re.IGNORECASE):
                        score += 10.0
                except re.error:
                    pass

            # Check description keywords
            desc_words = set(skill.description.lower().split())
            context_words = set(context_lower.split())
            overlap = desc_words & context_words
            if overlap:
                score += len(overlap) * 0.5

            # Check tags
            for tag in skill.tags:
                if tag.lower() in context_lower:
                    score += 2.0

            # Boost by success rate
            if skill.usage_count > 0:
                score += skill.success_rate * 3.0

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:5]]

    def get_pattern_suggestions(self) -> list[dict[str, Any]]:
        """Get current pattern detection suggestions.

        Returns:
            List of detected repeated patterns.
        """
        return self._pattern_detector.detect_repeated()

    def list_skills(self, tag: str | None = None) -> list[Skill]:
        """List all skills.

        Args:
            tag: Optional tag filter.

        Returns:
            List of Skill objects.
        """
        return self._store.list_all(tag=tag)

    def get_skill(self, name: str) -> Skill | None:
        """Get a specific skill.

        Args:
            name: Skill name.

        Returns:
            Skill if found, None otherwise.
        """
        return self._store.get(name)

    def delete_skill(self, name: str) -> bool:
        """Delete a skill.

        Args:
            name: Skill name.

        Returns:
            True if deleted.
        """
        return self._store.delete(name)

    def export_skills(self) -> dict[str, Any]:
        """Export all skills as a dictionary.

        Returns:
            Dictionary containing all skills.
        """
        skills = self._store.list_all(limit=100000)
        return {
            "version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "skills": [s.to_dict() for s in skills],
        }

    def import_skills(self, data: dict[str, Any]) -> int:
        """Import skills from a dictionary.

        Args:
            data: Dictionary with skill data.

        Returns:
            Number of skills imported.
        """
        count = 0
        for skill_data in data.get("skills", []):
            skill = Skill.from_dict(skill_data)
            self._store.save(skill)
            count += 1
        return count

    def skill_dna(self) -> dict[str, Any]:
        """Generate a fingerprint of the user's interaction patterns.

        Analyzes all skills and interaction history to produce a
        summary of the user's typical usage patterns.

        Returns:
            Dictionary with DNA fingerprint data.
        """
        skills = self._store.list_all(limit=100000)
        patterns = self._pattern_detector.detect_repeated()

        top_categories: dict[str, int] = {}
        for skill in skills:
            for tag in skill.tags:
                top_categories[tag] = top_categories.get(tag, 0) + 1

        total_executions = sum(s.usage_count for s in skills)
        total_success = sum(s.success_count for s in skills)
        overall_success_rate = (
            total_success / total_executions if total_executions > 0 else 0.0
        )

        return {
            "total_skills": len(skills),
            "total_executions": total_executions,
            "overall_success_rate": round(overall_success_rate, 3),
            "top_categories": dict(
                sorted(top_categories.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "detected_patterns": len(patterns),
            "most_used_skills": [
                {"name": s.name, "usage_count": s.usage_count}
                for s in sorted(skills, key=lambda x: x.usage_count, reverse=True)[:5]
                if s.usage_count > 0
            ],
            "interaction_history_size": len(self._interaction_history),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get skill engine statistics.

        Returns:
            Dictionary with skill engine stats.
        """
        skills = self._store.list_all(limit=100000)
        return {
            "total_skills": len(skills),
            "total_executions": sum(s.usage_count for s in skills),
            "auto_detected_patterns": len(self._pattern_detector.detect_repeated()),
            "skill_dna": self.skill_dna(),
        }

    def close(self) -> None:
        """Close the skill store."""
        self._store.close()
