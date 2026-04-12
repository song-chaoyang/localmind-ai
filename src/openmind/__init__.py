"""
OpenMind - One-click local AI chat application.

Replaces ChatGPT/Claude/Gemini with free, private, local AI powered by Ollama.
"""

__version__ = "0.1.0"
__author__ = "OpenMind Contributors"


def __getattr__(name):
    """Lazy import to avoid requiring httpx at import time."""
    if name == "OpenMind":
        from openmind.core.engine import OpenMind
        return OpenMind
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OpenMind", "__version__"]
