"""
LocalMind Command Line Interface.

Provides commands for starting, configuring, and managing LocalMind.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="localmind",
        description="🧠 LocalMind — Your Private AI Operating System",
        epilog="Example: localmind start --model llama3",
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version="LocalMind v0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── Start Command ───────────────────────────────────────────
    start_parser = subparsers.add_parser(
        "start",
        help="Start LocalMind",
    )
    start_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model to load (e.g., llama3, mistral, qwen2)",
    )
    start_parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        choices=["ollama", "llama.cpp"],
        help="Model provider",
    )
    start_parser.add_argument(
        "--cli",
        action="store_true",
        help="Start in CLI-only mode (no web UI)",
    )
    start_parser.add_argument(
        "--headless",
        action="store_true",
        help="Start headless (API server only)",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="API server port (default: 8080)",
    )
    start_parser.add_argument(
        "--ui-port",
        type=int,
        default=3000,
        help="Web UI port (default: 3000)",
    )
    start_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    start_parser.add_argument(
        "--dev",
        action="store_true",
        help="Start in development mode (auto-reload)",
    )
    start_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # ─── Chat Command ────────────────────────────────────────────
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session",
    )
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model to use",
    )

    # ─── Model Commands ──────────────────────────────────────────
    model_parser = subparsers.add_parser(
        "model",
        help="Manage models",
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command")

    model_list_parser = model_subparsers.add_parser("list", help="List available models")
    model_pull_parser = model_subparsers.add_parser("pull", help="Pull a model")
    model_pull_parser.add_argument("name", type=str, help="Model name to pull")
    model_rm_parser = model_subparsers.add_parser("remove", help="Remove a model")
    model_rm_parser.add_argument("name", type=str, help="Model name to remove")

    # ─── Plugin Commands ─────────────────────────────────────────
    plugin_parser = subparsers.add_parser(
        "plugin",
        help="Manage plugins",
    )
    plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_command")

    plugin_list_parser = plugin_subparsers.add_parser("list", help="List plugins")
    plugin_install_parser = plugin_subparsers.add_parser("install", help="Install a plugin")
    plugin_install_parser.add_argument("source", type=str, help="Plugin source (path or URL)")
    plugin_uninstall_parser = plugin_subparsers.add_parser("uninstall", help="Uninstall a plugin")
    plugin_uninstall_parser.add_argument("name", type=str, help="Plugin name")

    # ─── Agent Commands ──────────────────────────────────────────
    agent_parser = subparsers.add_parser(
        "agent",
        help="Manage and run agents",
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")

    agent_list_parser = agent_subparsers.add_parser("list", help="List available agents")
    agent_run_parser = agent_subparsers.add_parser("run", help="Run an agent")
    agent_run_parser.add_argument("name", type=str, help="Agent name")
    agent_run_parser.add_argument("task", type=str, help="Task description")
    agent_run_parser.add_argument("--model", "-m", type=str, default=None, help="Model to use")

    # ─── Document Commands ───────────────────────────────────────
    doc_parser = subparsers.add_parser(
        "documents",
        help="Manage documents for RAG",
    )
    doc_subparsers = doc_parser.add_subparsers(dest="doc_command")

    doc_ingest_parser = doc_subparsers.add_parser("ingest", help="Ingest a document")
    doc_ingest_parser.add_argument("path", type=str, help="Document path")
    doc_query_parser = doc_subparsers.add_parser("query", help="Query documents")
    doc_query_parser.add_argument("query", type=str, help="Search query")
    doc_query_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    # ─── Config Command ──────────────────────────────────────────
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_show_parser = config_subparsers.add_parser("show", help="Show current config")
    config_set_parser = config_subparsers.add_parser("set", help="Set a config value")
    config_set_parser.add_argument("key", type=str, help="Config key")
    config_set_parser.add_argument("value", type=str, help="Config value")

    # ─── Info Command ────────────────────────────────────────────
    subparsers.add_parser(
        "info",
        help="Show system information",
    )

    return parser


async def cmd_start(args: argparse.Namespace) -> None:
    """Handle the 'start' command."""
    from localmind import LocalMind, Config

    # Load config
    config = Config.load()
    config.log_level = args.log_level
    config.server.port = args.port
    config.server.ui_port = args.ui_port
    config.server.host = args.host
    config.server.reload = args.dev

    # Initialize engine
    mind = LocalMind(config=config)

    # Load model if specified
    if args.model:
        try:
            mind.load_model(args.model, args.provider)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if not args.headless:
                logger.info("Starting without a model. You can load one later via the API.")

    # Start appropriate mode
    if args.headless:
        logger.info(f"Starting LocalMind in headless mode on port {args.port}")
        from localmind.api import LocalMindAPI
        api = LocalMindAPI(engine=mind, host=args.host, port=args.port)
        await api.start()
    elif args.cli:
        logger.info("Starting LocalMind in CLI mode")
        await start_cli(mind)
    else:
        logger.info(
            f"Starting LocalMind with Web UI\n"
            f"  API: http://localhost:{args.port}\n"
            f"  UI:  http://localhost:{args.ui_port}\n"
            f"  Docs: http://localhost:{args.port}/docs"
        )
        from localmind.api import LocalMindAPI
        api = LocalMindAPI(engine=mind, host=args.host, port=args.port)
        await api.start()


async def start_cli(mind) -> None:
    """Start an interactive CLI chat session."""
    print("\n🧠 LocalMind CLI — Type 'exit' to quit, 'help' for commands\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye! 👋")
                break

            if user_input.lower() == "help":
                print(
                    "\nCommands:\n"
                    "  exit    — Quit LocalMind\n"
                    "  help    — Show this help\n"
                    "  clear   — Clear conversation\n"
                    "  model   — Show current model\n"
                    "  stats   — Show system stats\n"
                    "  /agent <name> <task> — Run an agent\n"
                    "  Anything else — Chat with the AI\n"
                )
                continue

            if user_input.lower() == "clear":
                mind.clear_conversation()
                print("✅ Conversation cleared\n")
                continue

            if user_input.lower() == "model":
                print(f"Current model: {mind._model_name or 'None'}")
                continue

            if user_input.lower() == "stats":
                stats = mind.get_stats()
                print(f"\n📊 System Stats:")
                print(f"  Version: {stats['version']}")
                print(f"  Model: {stats['model_name'] or 'None'}")
                print(f"  Agents: {stats['agents_registered']}")
                print(f"  Memory: {stats['memory']['short_term_messages']} messages\n")
                continue

            if user_input.startswith("/agent "):
                parts = user_input.split(maxsplit=2)
                if len(parts) >= 3:
                    agent_name, task = parts[1], parts[2]
                    try:
                        result = await mind.execute_agent(agent_name, task)
                        print(f"\n🤖 [{agent_name}]: {result}\n")
                    except Exception as e:
                        print(f"\n❌ Error: {e}\n")
                else:
                    print("Usage: /agent <name> <task>")
                continue

            # Regular chat
            if not mind._model_loaded:
                print("⚠️  No model loaded. Use 'localmind start --model <name>'\n")
                continue

            print("AI: ", end="", flush=True)
            async for chunk in mind.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def cmd_chat(args: argparse.Namespace) -> None:
    """Handle the 'chat' command."""
    from localmind import LocalMind

    mind = LocalMind()

    if args.model:
        mind.load_model(args.model)

    await start_cli(mind)


async def cmd_model(args: argparse.Namespace) -> None:
    """Handle the 'model' command."""
    from localmind import LocalMind

    mind = LocalMind()

    if args.model_command == "list":
        models = mind.list_available_models()
        if models:
            print("\n📦 Available Models:")
            for m in models:
                size_gb = m.get("size", 0) / (1024 ** 3)
                print(f"  • {m['name']} ({size_gb:.1f} GB)")
        else:
            print("\nNo models found. Make sure Ollama is running.")
            print("Install: https://ollama.ai")

    elif args.model_command == "pull":
        print(f"Pulling model '{args.name}'...")
        try:
            mind.load_model(args.name)
            print(f"✅ Model '{args.name}' pulled successfully!")
        except Exception as e:
            print(f"❌ Failed to pull model: {e}")

    elif args.model_command == "remove":
        print(f"Removing model '{args.name}'...")
        print("⚠️  Model removal is managed by your model provider (e.g., Ollama)")
        print(f"   Run: ollama rm {args.name}")


async def cmd_plugin(args: argparse.Namespace) -> None:
    """Handle the 'plugin' command."""
    from localmind import Config
    from localmind.plugins import PluginManager

    manager = PluginManager(Config.load())

    if args.plugin_command == "list":
        plugins = manager.list_plugins()
        if plugins:
            print("\n🔌 Loaded Plugins:")
            for p in plugins:
                print(f"  • {p['name']} v{p['version']} — {p['description']}")
        else:
            print("\nNo plugins loaded.")

    elif args.plugin_command == "install":
        success = manager.install_plugin(args.source)
        if success:
            print(f"✅ Plugin installed from '{args.source}'")
        else:
            print(f"❌ Failed to install plugin from '{args.source}'")

    elif args.plugin_command == "uninstall":
        success = manager.uninstall_plugin(args.name)
        if success:
            print(f"✅ Plugin '{args.name}' uninstalled")
        else:
            print(f"❌ Plugin '{args.name}' not found")


async def cmd_agent(args: argparse.Namespace) -> None:
    """Handle the 'agent' command."""
    from localmind.agents import BUILTIN_AGENTS

    if args.agent_command == "list":
        print("\n🤖 Available Agents:")
        for name, cls in BUILTIN_AGENTS.items():
            print(f"  • {name:12s} — {cls.description}")
            print(f"    Capabilities: {', '.join(cls.capabilities)}")

    elif args.agent_command == "run":
        from localmind import LocalMind
        from localmind.agents import create_agent

        mind = LocalMind()
        if args.model:
            mind.load_model(args.model)

        agent = create_agent(args.name, engine=mind)
        print(f"\n🤖 Running agent '{args.name}' on task: {args.task}\n")
        result = await agent.execute(args.task, {})
        print(f"\n📋 Result:\n{result}")


async def cmd_documents(args: argparse.Namespace) -> None:
    """Handle the 'documents' command."""
    from localmind import LocalMind

    mind = LocalMind()

    if args.doc_command == "ingest":
        try:
            chunks = mind.ingest_document(args.path)
            print(f"✅ Ingested document into {chunks} chunks")
        except Exception as e:
            print(f"❌ Failed to ingest document: {e}")

    elif args.doc_command == "query":
        results = await mind.query_documents(args.query, top_k=args.top_k)
        if results:
            print(f"\n📄 Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"\n  [{i}] Score: {r['score']:.3f}")
                print(f"  Source: {r['metadata'].get('source', 'unknown')}")
                print(f"  Content: {r['content'][:200]}...")
        else:
            print("No results found.")


async def cmd_config(args: argparse.Namespace) -> None:
    """Handle the 'config' command."""
    from localmind import Config

    if args.config_command == "show":
        config = Config.load()
        print("\n⚙️  LocalMind Configuration:")
        print(f"  Data directory: {config.data_dir}")
        print(f"  Log level: {config.log_level}")
        print(f"  Model: {config.model.name}")
        print(f"  Provider: {config.model.provider}")
        print(f"  Server: {config.server.host}:{config.server.port}")
        print(f"  Memory: {config.memory.short_term_max_messages} messages (short-term)")

    elif args.config_command == "set":
        print(f"⚠️  Config editing is best done by editing ~/.localmind/config.json")
        print(f"   Or use environment variables: LOCALMIND_{args.key.upper()}={args.value}")


async def cmd_info(args: argparse.Namespace) -> None:
    """Handle the 'info' command."""
    print("\n🧠 LocalMind — Your Private AI Operating System")
    print(f"   Version: 0.1.0")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Data directory: ~/.localmind")
    print(f"\n   For help: localmind --help")
    print(f"   For docs: https://github.com/song-chaoyang/localmind-ai")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Map commands to handlers
    handlers = {
        "start": cmd_start,
        "chat": cmd_chat,
        "model": cmd_model,
        "plugin": cmd_plugin,
        "agent": cmd_agent,
        "documents": cmd_documents,
        "config": cmd_config,
        "info": cmd_info,
    }

    handler = handlers.get(args.command)
    if handler:
        asyncio.run(handler(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
