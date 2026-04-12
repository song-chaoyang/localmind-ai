"""NexusMind API module - lazy imports."""

__all__ = ["create_app"]


def __getattr__(name: str):
    """Lazy import pattern to defer heavy module loading."""
    if name == "create_app":
        from nexusmind.api.app import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
