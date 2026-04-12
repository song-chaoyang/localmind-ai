"""
LocalMind — Your Private AI Operating System

A unified platform for local AI model management, intelligent agents,
plugin ecosystem, and workflow orchestration.
"""

__version__ = "0.1.0"
__author__ = "LocalMind Contributors"
__license__ = "MIT"

from localmind.core.engine import LocalMind
from localmind.core.config import Config
from localmind.core.memory import MemoryManager
from localmind.core.workflow import Workflow

__all__ = [
    "LocalMind",
    "Config",
    "MemoryManager",
    "Workflow",
]
