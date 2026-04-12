"""
OpenMind core package.

Provides the main engine, configuration, RAG pipeline, and memory systems.
"""

from openmind.core.config import Config
from openmind.core.memory import MemoryManager
from openmind.core.rag import RAGPipeline

__all__ = ["Config", "MemoryManager", "RAGPipeline"]


def __getattr__(name):
    if name == "OpenMind":
        from openmind.core.engine import OpenMind
        return OpenMind
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
