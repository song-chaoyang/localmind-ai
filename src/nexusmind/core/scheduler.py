"""Offline Task Scheduling System for NexusMind.

Provides cron-like task scheduling with support for natural language
schedules, notification integration, and persistent task storage.
Implements a simple scheduler without external dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from nexusmind.core.config import SchedulerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SchedulerError(Exception):
    """Base exception for scheduler operations."""


class TaskNotFoundError(SchedulerError):
    """Raised when a task is not found."""


class TaskExecutionError(SchedulerError):
    """Raised when task execution fails."""


# ---------------------------------------------------------------------------
# Enums and Data Types
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Status of a scheduled task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: int = 0
    model: str = ""
    tokens_used: int = 0


@dataclass
class ScheduledTask:
    """A scheduled task definition.

    Attributes:
        id: Unique task identifier.
        name: Human-readable task name.
        prompt: The prompt to send to the LLM.
        schedule: Schedule expression (cron or natural language).
        model: Model to use for execution.
        status: Current task status.
        created_at: Creation timestamp.
        last_run: Last execution timestamp.
        next_run: Next scheduled execution timestamp.
        run_count: Number of times executed.
        max_retries: Maximum retry attempts on failure.
        retry_count: Current retry count.
        notify: Notification channels.
        tags: Categorization tags.
    """

    id: str
    name: str
    prompt: str
    schedule: str = ""
    model: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    last_run: float = 0.0
    next_run: float = 0.0
    run_count: int = 0
    max_retries: int = 3
    retry_count: int = 0
    notify: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    timeout: float = 300.0  # 5 minutes default

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "schedule": self.schedule,
            "model": self.model,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "notify": self.notify,
            "tags": self.tags,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledTask:
        """Deserialize task from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            prompt=data["prompt"],
            schedule=data.get("schedule", ""),
            model=data.get("model"),
            status=TaskStatus(data.get("status", "pending")),
            created_at=data.get("created_at", time.time()),
            last_run=data.get("last_run", 0.0),
            next_run=data.get("next_run", 0.0),
            run_count=data.get("run_count", 0),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            notify=data.get("notify", []),
            tags=data.get("tags", []),
            timeout=data.get("timeout", 300.0),
        )


# ---------------------------------------------------------------------------
# Schedule Parser
# ---------------------------------------------------------------------------


class ScheduleParser:
    """Parse natural language and cron schedule expressions.

    Supported formats:
        - "every day at 9am"
        - "every monday"
        - "every 30 minutes"
        - "in 30 minutes"
        - "at 2:30pm"
        - Cron: "0 9 * * *" (minute hour day month weekday)
    """

    # Natural language patterns
    _PATTERNS: list[tuple[re.Pattern[str], Callable]] = []

    @classmethod
    def parse(cls, expression: str, base_time: datetime | None = None) -> float:
        """Parse a schedule expression and return the next run timestamp.

        Args:
            expression: Schedule expression (natural language or cron).
            base_time: Base time for calculation. Defaults to now.

        Returns:
            Unix timestamp of the next scheduled run.

        Raises:
            ValueError: If the expression cannot be parsed.
        """
        if base_time is None:
            base_time = datetime.now(timezone.utc)

        expression = expression.strip().lower()

        # Try natural language patterns
        result = cls._parse_natural(expression, base_time)
        if result is not None:
            return result

        # Try cron expression
        result = cls._parse_cron(expression, base_time)
        if result is not None:
            return result

        raise ValueError(f"Cannot parse schedule expression: {expression!r}")

    @classmethod
    def _parse_natural(
        cls, expression: str, base_time: datetime
    ) -> float | None:
        """Parse natural language schedule expressions.

        Args:
            expression: Lowercase expression string.
            base_time: Base time for calculation.

        Returns:
            Timestamp of next run, or None if not matched.
        """
        now = base_time

        # "in X minutes/hours/days"
        m = re.match(r"in\s+(\d+)\s+(minute|minutes|hour|hours|day|days)", expression)
        if m:
            amount = int(m.group(1))
            unit = m.group(2)
            if unit.startswith("minute"):
                delta = timedelta(minutes=amount)
            elif unit.startswith("hour"):
                delta = timedelta(hours=amount)
            else:
                delta = timedelta(days=amount)
            return (now + delta).timestamp()

        # "every X minutes"
        m = re.match(r"every\s+(\d+)\s+minutes?", expression)
        if m:
            minutes = int(m.group(1))
            next_run = now + timedelta(minutes=minutes)
            return next_run.timestamp()

        # "every hour"
        if expression == "every hour":
            return (now + timedelta(hours=1)).timestamp()

        # "every day at HH:MM(am|pm)?"
        m = re.match(r"every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", expression)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            ampm = m.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run.timestamp()

        # "every monday/tuesday/..." etc.
        weekday_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
        }
        m = re.match(r"every\s+(\w+)", expression)
        if m:
            day_name = m.group(1)
            if day_name in weekday_map:
                target_weekday = weekday_map[day_name]
                days_ahead = (target_weekday - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                next_run = now + timedelta(days=days_ahead)
                next_run = next_run.replace(hour=9, minute=0, second=0, microsecond=0)
                return next_run.timestamp()

        # "at HH:MM(am|pm)?"
        m = re.match(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", expression)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            ampm = m.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run.timestamp()

        return None

    @classmethod
    def _parse_cron(
        cls, expression: str, base_time: datetime
    ) -> float | None:
        """Parse a cron expression and return the next run time.

        Supports basic 5-field cron: minute hour day month weekday

        Args:
            expression: Cron expression string.
            base_time: Base time for calculation.

        Returns:
            Timestamp of next run, or None if not a valid cron expression.
        """
        parts = expression.split()
        if len(parts) != 5:
            return None

        # Validate all parts are cron-like
        cron_re = re.compile(r"^[\d,\-*/]+$")
        if not all(cron_re.match(p) for p in parts):
            return None

        try:
            minute_field, hour_field, day_field, month_field, weekday_field = parts
            now = base_time

            # Simple cron evaluation: check each minute for the next 366 days
            check_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            end_time = now + timedelta(days=366)

            while check_time < end_time:
                if (
                    cls._cron_matches(check_time.minute, minute_field, 0, 59)
                    and cls._cron_matches(check_time.hour, hour_field, 0, 23)
                    and cls._cron_matches(check_time.day, day_field, 1, 31)
                    and cls._cron_matches(check_time.month, month_field, 1, 12)
                    and cls._cron_matches(check_time.weekday(), weekday_field, 0, 6)
                ):
                    return check_time.timestamp()
                check_time += timedelta(minutes=1)

            return None
        except (ValueError, IndexError):
            return None

    @classmethod
    def _cron_matches(
        cls, value: int, field: str, min_val: int, max_val: int
    ) -> bool:
        """Check if a value matches a cron field.

        Args:
            value: The value to check.
            field: Cron field expression.
            min_val: Minimum valid value.
            max_val: Maximum valid value.

        Returns:
            True if the value matches the cron field.
        """
        if field == "*":
            return True

        # Handle step values: */5
        if field.startswith("*/"):
            step = int(field[2:])
            return value % step == 0

        # Handle ranges: 1-5
        if "-" in field:
            start, end = field.split("-", 1)
            return int(start) <= value <= int(end)

        # Handle lists: 1,3,5
        if "," in field:
            return value in [int(x) for x in field.split(",")]

        # Handle specific values
        try:
            return value == int(field)
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier:
    """Send notifications via multiple channels.

    Supports Telegram, Discord webhooks, and Slack.
    """

    def __init__(self, config: dict[str, str | None]) -> None:
        self.config = config
        self._telegram_url = (
            f"https://api.telegram.org/bot{config.get('telegram_token')}/sendMessage"
            if config.get("telegram_token")
            else None
        )
        self._discord_webhook = config.get("discord_webhook")
        self._slack_token = config.get("slack_token")
        self._slack_channel = config.get("slack_channel")

    async def send(self, message: str, channels: list[str] | None = None) -> bool:
        """Send a notification message.

        Args:
            message: The message to send.
            channels: List of channel names ('telegram', 'discord', 'slack').
                     If None, sends to all configured channels.

        Returns:
            True if at least one notification was sent successfully.
        """
        if channels is None:
            channels = []
            if self._telegram_url:
                channels.append("telegram")
            if self._discord_webhook:
                channels.append("discord")
            if self._slack_token:
                channels.append("slack")

        if not channels:
            logger.debug("No notification channels configured")
            return False

        any_success = False
        for channel in channels:
            try:
                if channel == "telegram" and self._telegram_url:
                    await self._send_telegram(message)
                    any_success = True
                elif channel == "discord" and self._discord_webhook:
                    await self._send_discord(message)
                    any_success = True
                elif channel == "slack" and self._slack_token:
                    await self._send_slack(message)
                    any_success = True
            except Exception as e:
                logger.warning("Failed to send %s notification: %s", channel, e)

        return any_success

    async def _send_telegram(self, message: str) -> None:
        """Send a Telegram notification."""
        import httpx

        chat_id = self.config.get("telegram_chat_id")
        async with httpx.AsyncClient() as client:
            await client.post(
                self._telegram_url,
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=30.0,
            )

    async def _send_discord(self, message: str) -> None:
        """Send a Discord webhook notification."""
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                self._discord_webhook,
                json={"content": message},
                timeout=30.0,
            )

    async def _send_slack(self, message: str) -> None:
        """Send a Slack notification."""
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self._slack_token}"},
                json={"channel": self._slack_channel, "text": message},
                timeout=30.0,
            )


# ---------------------------------------------------------------------------
# Task Store
# ---------------------------------------------------------------------------


class TaskStore:
    """SQLite-backed persistent storage for scheduled tasks."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_dir / "tasks.db"
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
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                schedule TEXT DEFAULT '',
                model TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                last_run REAL DEFAULT 0,
                next_run REAL DEFAULT 0,
                run_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                retry_count INTEGER DEFAULT 0,
                notify TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                timeout REAL DEFAULT 300.0
            );

            CREATE TABLE IF NOT EXISTS task_results (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                status TEXT NOT NULL,
                output TEXT DEFAULT '',
                error TEXT DEFAULT '',
                started_at REAL,
                completed_at REAL,
                duration_ms INTEGER DEFAULT 0,
                model TEXT DEFAULT '',
                tokens_used INTEGER DEFAULT 0,
                timestamp REAL NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON tasks(next_run);
            CREATE INDEX IF NOT EXISTS idx_results_task ON task_results(task_id);
        """)
        conn.commit()

    def save_task(self, task: ScheduledTask) -> None:
        """Save or update a task.

        Args:
            task: The ScheduledTask to save.
        """
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, name, prompt, schedule, model, status, created_at, last_run,
                next_run, run_count, max_retries, retry_count, notify, tags, timeout)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id, task.name, task.prompt, task.schedule, task.model,
                task.status.value, task.created_at, task.last_run, task.next_run,
                task.run_count, task.max_retries, task.retry_count,
                json.dumps(task.notify), json.dumps(task.tags), task.timeout,
            ),
        )
        conn.commit()

    def get_task(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            ScheduledTask if found, None otherwise.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(
        self, status: TaskStatus | None = None, limit: int = 100
    ) -> list[ScheduledTask]:
        """List tasks with optional status filter.

        Args:
            status: Optional status filter.
            limit: Maximum results.

        Returns:
            List of ScheduledTask objects.
        """
        conn = self._get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY next_run ASC LIMIT ?",
                (status.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY next_run ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_due_tasks(self) -> list[ScheduledTask]:
        """Get tasks that are due for execution.

        Returns:
            List of due ScheduledTask objects.
        """
        conn = self._get_conn()
        now = time.time()
        rows = conn.execute(
            """SELECT * FROM tasks
               WHERE status = 'pending' AND next_run > 0 AND next_run <= ?
               ORDER BY next_run ASC""",
            (now,),
        ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def delete_task(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if deleted.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount > 0

    def save_result(self, result: TaskResult) -> None:
        """Save a task execution result.

        Args:
            result: The TaskResult to save.
        """
        import uuid

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO task_results
               (id, task_id, status, output, error, started_at, completed_at,
                duration_ms, model, tokens_used, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex[:16], result.task_id, result.status.value,
                result.output, result.error, result.started_at, result.completed_at,
                result.duration_ms, result.model, result.tokens_used, time.time(),
            ),
        )
        conn.commit()

    def get_results(self, task_id: str, limit: int = 20) -> list[TaskResult]:
        """Get execution results for a task.

        Args:
            task_id: Task identifier.
            limit: Maximum results.

        Returns:
            List of TaskResult objects.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM task_results WHERE task_id = ? ORDER BY timestamp DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
        return [
            TaskResult(
                task_id=row["task_id"],
                status=TaskStatus(row["status"]),
                output=row.get("output", ""),
                error=row.get("error", ""),
                started_at=row.get("started_at", 0.0),
                completed_at=row.get("completed_at", 0.0),
                duration_ms=row.get("duration_ms", 0),
                model=row.get("model", ""),
                tokens_used=row.get("tokens_used", 0),
            )
            for row in rows
        ]

    def _row_to_task(self, row: sqlite3.Row) -> ScheduledTask:
        """Convert a database row to a ScheduledTask."""
        return ScheduledTask(
            id=row["id"],
            name=row["name"],
            prompt=row["prompt"],
            schedule=row.get("schedule", ""),
            model=row.get("model"),
            status=TaskStatus(row.get("status", "pending")),
            created_at=row.get("created_at", time.time()),
            last_run=row.get("last_run", 0.0),
            next_run=row.get("next_run", 0.0),
            run_count=row.get("run_count", 0),
            max_retries=row.get("max_retries", 3),
            retry_count=row.get("retry_count", 0),
            notify=json.loads(row.get("notify") or "[]"),
            tags=json.loads(row.get("tags") or "[]"),
            timeout=row.get("timeout", 300.0),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Task Scheduler
# ---------------------------------------------------------------------------


class TaskScheduler:
    """Offline Task Scheduling Engine.

    Schedules, manages, and executes tasks with cron-like scheduling,
    notification support, and persistent storage.

    Example::

        scheduler = TaskScheduler(SchedulerConfig())
        task = scheduler.schedule_task(
            "Daily Report", "Generate a summary of yesterday's work",
            "every day at 9am"
        )
        scheduler.run_now(task.id)
        results = scheduler.get_task_results(task.id)
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        executor: Callable[[str, str | None], Coroutine[Any, Any, str]] | None = None,
        notification_config: dict[str, str | None] | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self._store = TaskStore(persist_dir=self.config.persist_dir)
        self._executor = executor
        self._notifier = Notifier(notification_config or {})
        self._running = False
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    def schedule_task(
        self,
        name: str,
        prompt: str,
        schedule: str,
        model: str | None = None,
        notify: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> ScheduledTask:
        """Schedule a new task.

        Args:
            name: Human-readable task name.
            prompt: The prompt to send to the LLM.
            schedule: Schedule expression (natural language or cron).
            model: Model to use. Defaults to config default.
            notify: Notification channels.
            tags: Categorization tags.

        Returns:
            The created ScheduledTask.

        Raises:
            ValueError: If the schedule expression cannot be parsed.
        """
        import uuid

        task_id = uuid.uuid4().hex[:16]
        next_run = ScheduleParser.parse(schedule)

        task = ScheduledTask(
            id=task_id,
            name=name,
            prompt=prompt,
            schedule=schedule,
            model=model or self.config.default_model,
            next_run=next_run,
            notify=notify or [],
            tags=tags or [],
            max_retries=self.config.max_retries,
        )

        self._store.save_task(task)
        logger.info(
            "Scheduled task '%s' (id=%s) for %s",
            name, task_id,
            datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat(),
        )
        return task

    async def run_now(self, task_id: str) -> TaskResult:
        """Immediately execute a scheduled task.

        Args:
            task_id: Task identifier.

        Returns:
            TaskResult with execution details.

        Raises:
            TaskNotFoundError: If the task doesn't exist.
        """
        task = self._store.get_task(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found")

        async with self._semaphore:
            start_time = time.time()
            task.status = TaskStatus.RUNNING
            task.last_run = start_time
            self._store.save_task(task)

            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                started_at=start_time,
                model=task.model or "",
            )

            try:
                if self._executor:
                    output = await asyncio.wait_for(
                        self._executor(task.prompt, task.model),
                        timeout=task.timeout,
                    )
                    result.output = output
                    result.status = TaskStatus.COMPLETED
                else:
                    result.output = f"[Task '{task.name}' executed - no executor configured]"
                    result.status = TaskStatus.COMPLETED

                task.retry_count = 0
            except asyncio.TimeoutError:
                result.error = f"Task timed out after {task.timeout}s"
                result.status = TaskStatus.FAILED
            except Exception as e:
                result.error = str(e)
                result.status = TaskStatus.FAILED
                if self.config.retry_failed and task.retry_count < task.max_retries:
                    task.retry_count += 1
                    logger.info(
                        "Retrying task '%s' (attempt %d/%d)",
                        task.name, task.retry_count, task.max_retries,
                    )

            result.completed_at = time.time()
            result.duration_ms = int((result.completed_at - start_time) * 1000)

            # Update task status
            if result.status == TaskStatus.COMPLETED:
                task.status = TaskStatus.PENDING
                task.run_count += 1
                # Recalculate next run
                if task.schedule:
                    try:
                        task.next_run = ScheduleParser.parse(
                            task.schedule,
                            datetime.fromtimestamp(time.time(), tz=timezone.utc),
                        )
                    except ValueError:
                        pass
            elif result.status == TaskStatus.FAILED:
                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                else:
                    task.status = TaskStatus.PENDING

            self._store.save_task(task)
            self._store.save_result(result)

            # Send notifications
            if task.notify:
                status_emoji = "OK" if result.status == TaskStatus.COMPLETED else "FAIL"
                message = (
                    f"[NexusMind] Task '{task.name}' {status_emoji}\n"
                    f"Duration: {result.duration_ms}ms\n"
                    f"Output: {result.output[:500]}"
                )
                if result.error:
                    message += f"\nError: {result.error}"
                await self._notifier.send(message, channels=task.notify)

            return result

    def list_tasks(self, status: TaskStatus | None = None) -> list[ScheduledTask]:
        """List all scheduled tasks.

        Args:
            status: Optional status filter.

        Returns:
            List of ScheduledTask objects.
        """
        return self._store.list_tasks(status=status)

    def get_task(self, task_id: str) -> ScheduledTask | None:
        """Get a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            ScheduledTask if found, None otherwise.
        """
        return self._store.get_task(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task.

        Args:
            task_id: Task identifier.

        Returns:
            True if the task was found and cancelled.
        """
        task = self._store.get_task(task_id)
        if task is None:
            return False
        task.status = TaskStatus.CANCELLED
        self._store.save_task(task)
        return True

    def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task.

        Args:
            task_id: Task identifier.

        Returns:
            True if the task was found and paused.
        """
        task = self._store.get_task(task_id)
        if task is None:
            return False
        task.status = TaskStatus.PAUSED
        self._store.save_task(task)
        return True

    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task.

        Args:
            task_id: Task identifier.

        Returns:
            True if the task was found and resumed.
        """
        task = self._store.get_task(task_id)
        if task is None:
            return False
        if task.status != TaskStatus.PAUSED:
            return False
        task.status = TaskStatus.PENDING
        if task.schedule and task.next_run == 0:
            try:
                task.next_run = ScheduleParser.parse(task.schedule)
            except ValueError:
                pass
        self._store.save_task(task)
        return True

    def get_task_results(self, task_id: str) -> list[TaskResult]:
        """Get execution results for a task.

        Args:
            task_id: Task identifier.

        Returns:
            List of TaskResult objects.
        """
        return self._store.get_results(task_id)

    async def start(self) -> None:
        """Start the scheduler loop.

        Checks for due tasks periodically and executes them.
        """
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_event_loop()
        logger.info("Task scheduler started")
        while self._running:
            try:
                due_tasks = self._store.get_due_tasks()
                for task in due_tasks:
                    asyncio_task = asyncio.create_task(
                        self.run_now(task.id),
                        name=f"scheduler-{task.id}",
                    )
                    self._active_tasks[task.id] = asyncio_task
                    asyncio_task.add_done_callback(
                        lambda t, tid=task.id: self._active_tasks.pop(tid, None)
                    )
            except Exception as e:
                logger.error("Scheduler loop error: %s", e)
            await asyncio.sleep(30)  # Check every 30 seconds

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        for task in self._active_tasks.values():
            task.cancel()
        self._active_tasks.clear()
        logger.info("Task scheduler stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats.
        """
        all_tasks = self._store.list_tasks(limit=100000)
        return {
            "total_tasks": len(all_tasks),
            "pending": sum(1 for t in all_tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in all_tasks if t.status == TaskStatus.RUNNING),
            "completed": sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in all_tasks if t.status == TaskStatus.FAILED),
            "paused": sum(1 for t in all_tasks if t.status == TaskStatus.PAUSED),
            "active_executions": len(self._active_tasks),
            "scheduler_running": self._running,
        }

    def close(self) -> None:
        """Close the task store."""
        self._store.close()
