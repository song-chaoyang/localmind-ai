"""
Utility helpers for OpenMind.

Provides commonly used helper functions including byte formatting, text
truncation, ID generation, a timer context manager, and safe JSON parsing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import string
import time
import uuid
from typing import Any, Dict, Generator, Iterator, Optional


def format_bytes(size: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size: Number of bytes.

    Returns:
        A string such as ``"1.23 MB"`` or ``"456 B"``.

    Example::

        >>> format_bytes(1536)
        '1.50 KB'
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate *text* to *max_length* characters, appending *suffix* if truncated.

    Args:
        text: The input string.
        max_length: Maximum allowed length (including suffix).
        suffix: String appended when truncation occurs.

    Returns:
        The possibly truncated string.

    Example::

        >>> truncate_text("Hello, world!", max_length=8)
        'Hello...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def generate_id(length: int = 12) -> str:
    """Generate a random alphanumeric identifier.

    Args:
        length: Number of characters in the generated ID.

    Returns:
        A random string of ASCII letters and digits.

    Example::

        >>> generate_id(8)
        'aB3x9Qz1'
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=length))


def generate_uuid() -> str:
    """Generate a UUID4 string.

    Returns:
        A UUID4 string without dashes for compactness.
    """
    return uuid.uuid4().hex


class timer:
    """Context manager that measures elapsed wall-clock time.

    Attributes:
        elapsed: The number of seconds spent inside the ``with`` block
            (available after the block exits).

    Example::

        with timer() as t:
            time.sleep(0.1)
        print(f"Took {t.elapsed:.3f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def safe_json_loads(text: str, default: Optional[Any] = None) -> Any:
    """Attempt to parse *text* as JSON, returning *default* on failure.

    Args:
        text: The JSON string to parse.
        default: Value returned when parsing fails. Defaults to ``None``.

    Returns:
        The parsed Python object, or *default*.

    Example::

        >>> safe_json_loads('{"a": 1}')
        {'a': 1}
        >>> safe_json_loads('not json', default={})
        {}
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text*.

    Useful for generating deterministic document/chunk identifiers.

    Args:
        text: Input string.

    Returns:
        A 64-character hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
