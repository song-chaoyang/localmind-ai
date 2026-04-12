"""NexusMind core module - lazy imports for all core components."""

__all__ = ["Config", "MemoryManager", "SkillEngine", "TaskScheduler"]


def __getattr__(name: str):
    """Lazy import pattern to defer heavy module loading."""
    if name == "Config":
        from nexusmind.core.config import Config
        return Config
    if name == "MemoryManager":
        from nexusmind.core.memory import MemoryManager
        return MemoryManager
    if name == "SkillEngine":
        from nexusmind.core.skills import SkillEngine
        return SkillEngine
    if name == "TaskScheduler":
        from nexusmind.core.scheduler import TaskScheduler
        return TaskScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
