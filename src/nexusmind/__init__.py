"""
NexusMind - AI Agent with Persistent Memory, Auto Skill Evolution,
Offline Scheduling, and Multi-Provider LLM Support.

This package uses lazy imports to avoid loading heavy dependencies
(like httpx) at module level. Access the main class via:

    from nexusmind import NexusMind
"""

__version__ = "0.1.0"
__all__ = ["NexusMind"]


def __getattr__(name: str):
    """Lazy import pattern to defer heavy module loading."""
    if name == "NexusMind":
        from nexusmind.core.engine import NexusMind
        return NexusMind
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
