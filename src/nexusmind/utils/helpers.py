"""Utility helpers for NexusMind."""

from __future__ import annotations

import functools
import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable, TypeVar

T = TypeVar("T")


def generate_id() -> str:
    """Generate a unique identifier string."""
    return uuid.uuid4().hex[:16]


def format_bytes(size: int) -> str:
    """Format byte count to human-readable string.

    Args:
        size: Number of bytes.

    Returns:
        Human-readable string like '1.5 MB'.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to a maximum length with an optional suffix.

    Args:
        text: The text to truncate.
        max_length: Maximum character length.
        suffix: Suffix to append when truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def hash_text(text: str) -> str:
    """Compute SHA-256 hash of a text string.

    Args:
        text: The text to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure.

    Args:
        text: JSON string to parse.
        default: Value to return on parse failure.

    Returns:
        Parsed JSON object or default.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def chunk_list(lst: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of the given size.

    Args:
        lst: The list to split.
        chunk_size: Maximum size of each chunk.

    Returns:
        List of chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


@contextmanager
def timer(label: str | None = None):
    """Context manager to measure elapsed time.

    Args:
        label: Optional label for the timer.

    Yields:
        A dict with 'start', 'end', and 'elapsed' keys.
    """
    result: dict[str, float] = {"start": 0.0, "end": 0.0, "elapsed": 0.0}
    result["start"] = time.perf_counter()
    try:
        yield result
    finally:
        result["end"] = time.perf_counter()
        result["elapsed"] = result["end"] - result["start"]


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator for async functions with retry logic.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
        exceptions: Tuple of exception types to catch.

    Returns:
        Decorated async function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        import asyncio

                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
