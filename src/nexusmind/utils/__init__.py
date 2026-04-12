"""NexusMind utilities module."""

__all__ = [
    "format_bytes",
    "truncate_text",
    "generate_id",
    "timer",
    "safe_json_loads",
    "hash_text",
    "chunk_list",
    "async_retry",
]


def __getattr__(name: str):
    """Lazy import pattern."""
    if name in (
        "format_bytes",
        "truncate_text",
        "generate_id",
        "timer",
        "safe_json_loads",
        "hash_text",
        "chunk_list",
        "async_retry",
    ):
        from nexusmind.utils.helpers import (
            async_retry,
            chunk_list,
            format_bytes,
            generate_id,
            hash_text,
            safe_json_loads,
            timer,
            truncate_text,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
