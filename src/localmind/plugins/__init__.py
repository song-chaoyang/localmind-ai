"""
Plugin System for LocalMind.

Provides a flexible plugin architecture for extending LocalMind's
capabilities with community-built plugins.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = "MIT"
    min_localmind_version: str = "0.1.0"
    tags: List[str] = field(default_factory=list)


class Plugin(ABC):
    """
    Abstract base class for LocalMind plugins.

    Plugins can extend LocalMind with new tools, agents, commands,
    and event handlers.
    """

    metadata: PluginMetadata = PluginMetadata(name="unnamed")

    def __init__(self):
        self._logger = logging.getLogger(f"plugin.{self.metadata.name}")
        self._enabled: bool = True

    @property
    def logger(self) -> logging.Logger:
        """Get the plugin's logger."""
        return self._logger

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return self.metadata.name

    def on_load(self) -> None:
        """Called when the plugin is loaded."""
        self._logger.info(f"Plugin '{self.metadata.name}' loaded")

    def on_unload(self) -> None:
        """Called when the plugin is unloaded."""
        self._logger.info(f"Plugin '{self.metadata.name}' unloaded")

    def on_chat_message(self, message: str, context: Dict[str, Any]) -> str:
        """
        Called on every chat message. Can modify or enhance the message.

        Args:
            message: The chat message
            context: Additional context

        Returns:
            Potentially modified message
        """
        return message

    def register_tools(self) -> List[Dict[str, Any]]:
        """
        Register tools that agents can use.

        Returns:
            List of tool definitions
        """
        return []

    def register_commands(self) -> Dict[str, Callable]:
        """
        Register CLI commands.

        Returns:
            Dictionary of command name -> handler function
        """
        return {}

    def register_event_handlers(self) -> Dict[str, Callable]:
        """
        Register event handlers.

        Returns:
            Dictionary of event type -> handler function
        """
        return {}

    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the plugin's configuration schema.

        Returns:
            JSON schema for plugin configuration
        """
        return {}

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update plugin configuration."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "enabled": self._enabled,
        }


def plugin_metadata(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    homepage: str = "",
    license: str = "MIT",
    min_localmind_version: str = "0.1.0",
    tags: Optional[List[str]] = None,
) -> Callable[[Type], Type]:
    """
    Decorator to set plugin metadata.

    Usage:
        @plugin_metadata(
            name="my-plugin",
            version="1.0.0",
            description="My awesome plugin"
        )
        class MyPlugin(Plugin):
            ...
    """
    def decorator(cls: Type) -> Type:
        cls.metadata = PluginMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            homepage=homepage,
            license=license,
            min_localmind_version=min_localmind_version,
            tags=tags or [],
        )
        return cls
    return decorator


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Handles finding plugins from:
    - The built-in plugins directory
    - The user's plugin directory (~/.localmind/plugins/)
    - Installed Python packages
    """

    def __init__(self, config=None):
        self.config = config
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_dir = Path(
            config.plugin.plugin_dir if config
            else str(Path.home() / ".localmind" / "plugins")
        )
        self._builtin_plugin_dir = Path(__file__).parent / "builtin"

    def load_all(self) -> None:
        """Load all available plugins."""
        self._load_builtin_plugins()
        self._load_user_plugins()
        logger.info(f"Loaded {len(self._plugins)} plugins")

    def _load_builtin_plugins(self) -> None:
        """Load built-in plugins."""
        if self._builtin_plugin_dir.exists():
            self._load_plugins_from_dir(self._builtin_plugin_dir)

    def _load_user_plugins(self) -> None:
        """Load user-installed plugins."""
        if self._plugin_dir.exists():
            self._load_plugins_from_dir(self._plugin_dir)

    def _load_plugins_from_dir(self, directory: Path) -> None:
        """Load all plugins from a directory."""
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                self._load_plugin_file(file_path)
            except Exception as e:
                logger.error(f"Failed to load plugin from {file_path}: {e}")

    def _load_plugin_file(self, file_path: Path) -> None:
        """Load a plugin from a Python file."""
        spec = importlib.util.spec_from_file_location(
            f"localmind_plugin_{file_path.stem}",
            file_path,
        )
        if not spec or not spec.loader:
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Plugin subclasses in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
                and hasattr(attr, "metadata")
            ):
                self._register_plugin(attr())

    def _register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin instance."""
        name = plugin.name
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already loaded, skipping")
            return

        try:
            plugin.on_load()
            self._plugins[name] = plugin
            logger.info(f"Registered plugin: {name} v{plugin.metadata.version}")
        except Exception as e:
            logger.error(f"Failed to initialize plugin '{name}': {e}")

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        try:
            plugin.on_unload()
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload plugin '{name}': {e}")
            return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins."""
        return [plugin.get_info() for plugin in self._plugins.values()]

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools registered by plugins."""
        tools = []
        for plugin in self._plugins.values():
            if plugin._enabled:
                tools.extend(plugin.register_tools())
        return tools

    def get_all_commands(self) -> Dict[str, Callable]:
        """Get all commands registered by plugins."""
        commands = {}
        for plugin in self._plugins.values():
            if plugin._enabled:
                commands.update(plugin.register_commands())
        return commands

    def install_plugin(self, source: str) -> bool:
        """
        Install a plugin from a file path or URL.

        Args:
            source: Path to plugin file or directory

        Returns:
            True if installation succeeded
        """
        source_path = Path(source)

        if source_path.is_file() and source_path.suffix == ".py":
            # Copy to plugin directory
            self._plugin_dir.mkdir(parents=True, exist_ok=True)
            dest = self._plugin_dir / source_path.name
            import shutil
            shutil.copy2(source_path, dest)

            # Load the new plugin
            try:
                self._load_plugin_file(dest)
                logger.info(f"Installed plugin from {source_path.name}")
                return True
            except Exception as e:
                logger.error(f"Failed to install plugin: {e}")
                return False

        return False

    def uninstall_plugin(self, name: str) -> bool:
        """Uninstall and remove a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        self.unload_plugin(name)

        # Remove plugin file
        plugin_file = self._plugin_dir / f"{name}.py"
        if plugin_file.exists():
            plugin_file.unlink()

        return True
