"""
OpenMind command-line interface.

Provides the ``openmind`` CLI command for starting the application, chatting
interactively, managing models, and listing plugins.

Usage examples::

    openmind                          # Start the full app (web UI + model)
    openmind start                    # Same as above
    openmind serve --port 8080        # Start API server only
    openmind chat --model llama3      # Interactive CLI chat
    openmind model list               # List available models
    openmind model pull mistral       # Download a model
    openmind plugin list              # List installed plugins
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

__version__ = "0.1.0"

logger = logging.getLogger("openmind")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> None:
    """Start the full OpenMind application (web UI + API server)."""
    from openmind.api import serve
    from openmind.core.config import Config

    config = Config.load()
    if args.model:
        config.model.name = args.model
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host

    print(f"Starting OpenMind v{__version__} ...")
    print(f"  Model : {config.model.name}")
    print(f"  Host  : {config.server.host}")
    print(f"  Port  : {config.server.port}")
    print(f"  UI    : http://{config.server.host}:{config.server.port}")
    print()
    print("Press Ctrl+C to stop.\n")

    serve(
        host=config.server.host,
        port=config.server.port,
        config=config,
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the API server only (no web UI)."""
    from openmind.api import serve
    from openmind.core.config import Config

    config = Config.load()
    if args.model:
        config.model.name = args.model
    if args.port:
        config.server.port = args.port
    if args.host:
        config.server.host = args.host

    print(f"Starting OpenMind API server v{__version__} ...")
    print(f"  Model : {config.model.name}")
    print(f"  Host  : {config.server.host}")
    print(f"  Port  : {config.server.port}")
    print(f"  Docs  : http://{config.server.host}:{config.server.port}/docs")
    print()
    print("Press Ctrl+C to stop.\n")

    serve(
        host=config.server.host,
        port=config.server.port,
        config=config,
    )


def cmd_chat(args: argparse.Namespace) -> None:
    """Run an interactive CLI chat session."""
    from openmind.core.engine import OpenMind

    model = args.model or "llama3"
    print(f"OpenMind v{__version__} -- Interactive Chat")
    print(f"Model: {model}")
    print("Type 'exit', 'quit', or Ctrl+C to stop.\n")

    engine = OpenMind(model=model)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Handle slash commands
            if user_input.startswith("/"):
                _handle_slash_command(engine, user_input)
                continue

            try:
                response = engine.chat(user_input)
                print(f"\nAssistant: {response}\n")
            except Exception as exc:
                print(f"\nError: {exc}\n")
    finally:
        engine.close()


def _handle_slash_command(engine: "OpenMind", command: str) -> None:
    """Handle in-chat slash commands.

    Supported commands:
        /model <name>  -- Switch to a different model
        /clear         -- Clear conversation history
        /stats         -- Show runtime statistics
        /help          -- Show available commands
    """
    from openmind.core.engine import OpenMind

    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/model" and arg:
        try:
            engine.switch_model(arg)
            print(f"Switched to model: {arg}\n")
        except Exception as exc:
            print(f"Error switching model: {exc}\n")
    elif cmd == "/clear":
        engine.clear_history()
        print("History cleared.\n")
    elif cmd == "/stats":
        stats = engine.get_stats()
        print(f"  Model       : {stats['model']}")
        print(f"  Queries     : {stats['total_queries']}")
        print(f"  Tokens      : {stats['total_tokens']}")
        print(f"  Memory msgs : {stats['memory']['short_term_messages']}")
        print()
    elif cmd == "/help":
        print("Available commands:")
        print("  /model <name>  Switch model")
        print("  /clear         Clear history")
        print("  /stats         Show statistics")
        print("  /help          Show this help")
        print()
    else:
        print(f"Unknown command: {command}. Type /help for available commands.\n")


def cmd_model_list(args: argparse.Namespace) -> None:
    """List locally available Ollama models."""
    from openmind.models.manager import ModelManager

    try:
        mm = ModelManager()
        models = mm.list_available_models()
        if not models:
            print("No models found. Pull one with: openmind model pull <name>")
            return
        print(f"Available models ({len(models)}):\n")
        for m in models:
            name = m.get("name", "unknown")
            size = m.get("size", 0)
            modified = m.get("modified_at", "unknown")
            from openmind.utils.helpers import format_bytes

            size_str = format_bytes(size)
            print(f"  {name:<40s} {size_str:>12s}  (modified: {modified})")
        print()
        mm.close()
    except ConnectionError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_model_pull(args: argparse.Namespace) -> None:
    """Pull (download) an Ollama model."""
    from openmind.models.manager import ModelManager

    name = args.name
    if not name:
        print("Error: Please specify a model name. Usage: openmind model pull <name>")
        sys.exit(1)

    try:
        mm = ModelManager()
        print(f"Pulling model '{name}' ... (this may take a while)")
        mm.pull_model(name)
        print(f"Model '{name}' pulled successfully.")
        mm.close()
    except ConnectionError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_plugin_list(args: argparse.Namespace) -> None:
    """List installed plugins (placeholder for future plugin system)."""
    print("OpenMind Plugin System")
    print("  (No plugins installed yet. Plugin support is coming in a future release.)\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="openmind",
        description="OpenMind -- One-click local AI chat application.",
        epilog="Run 'openmind <command> --help' for details on a specific command.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"OpenMind v{__version__}",
    )

    # Global flags (used by multiple subcommands)
    def add_model_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model", "-m", type=str, default=None, help="AI model to use (default: llama3)")
        sub.add_argument("--port", "-p", type=int, default=None, help="Port number")
        sub.add_argument("--host", type=str, default=None, help="Host address")

    # -- start
    sub_start = parser.add_parser("start", help="Start the full OpenMind application")
    add_model_args(sub_start)

    # -- serve
    sub_serve = parser.add_parser("serve", help="Start the API server only")
    add_model_args(sub_serve)

    # -- chat
    sub_chat = parser.add_parser("chat", help="Interactive CLI chat")
    add_model_args(sub_chat)

    # -- model group
    sub_model = parser.add_parser("model", help="Manage Ollama models")
    model_sub = sub_model.add_subparsers(dest="model_action")

    sub_model_list = model_sub.add_parser("list", help="List available models")
    sub_model_pull = model_sub.add_parser("pull", help="Pull a model")
    sub_model_pull.add_argument("name", type=str, nargs="?", default="", help="Model name to pull")

    # -- plugin group
    sub_plugin = parser.add_parser("plugin", help="Manage plugins")
    plugin_sub = sub_plugin.add_subparsers(dest="plugin_action")
    plugin_sub.add_parser("list", help="List installed plugins")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and dispatch to the appropriate command handler.

    This is the primary entry point for the ``openmind`` CLI.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Dispatch
    command = args.__dict__.get("model_action") or args.__dict__.get("plugin_action")

    if args.__dict__.get("model_action") == "list":
        cmd_model_list(args)
    elif args.__dict__.get("model_action") == "pull":
        cmd_model_pull(args)
    elif args.__dict__.get("plugin_action") == "list":
        cmd_plugin_list(args)
    elif hasattr(args, "model_action") or hasattr(args, "plugin_action"):
        # A subcommand group was invoked but no action matched
        parser.print_help()
        sys.exit(1)
    else:
        # Determine which top-level command was used
        # When no subcommand is given, default to "start"
        action = getattr(args, "model_action", None) or getattr(args, "plugin_action", None)
        # Use the first positional arg to determine the command
        if argv and len(argv) > 0:
            first = argv[0]
            if first == "serve":
                cmd_serve(args)
            elif first == "chat":
                cmd_chat(args)
            elif first == "model":
                # Already handled above
                parser.print_help()
            elif first == "plugin":
                parser.print_help()
            else:
                cmd_start(args)
        else:
            # No arguments -- default to start
            cmd_start(args)


if __name__ == "__main__":
    main()
