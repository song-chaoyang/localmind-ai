"""
Utility helpers for LocalMind.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from functools import wraps
from typing import Any, Dict, Generator, Optional, TypeVar

T = TypeVar("T")


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


@contextlib.contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.3f}s")


def safe_json_loads(text: str, default: Optional[T] = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}" if prefix else uid


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator for async functions with retry logic."""
    def decorator(func):
        import asyncio
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception
        return wrapper
    return decorator


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of the given size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
