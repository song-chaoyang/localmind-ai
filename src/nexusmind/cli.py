"""NexusMind Command-Line Interface.

Provides a rich terminal interface for interacting with NexusMind,
including interactive chat, model management, memory operations,
skill management, and task scheduling.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _echo(msg: str, style: str = "") -> None:
    """Print a message using rich if available, otherwise plain text."""
    try:
        from rich.console import Console

        console = Console()
        if style:
            console.print(msg, style=style)
        else:
            console.print(msg)
    except ImportError:
        print(msg)


def _echo_error(msg: str) -> None:
    """Print an error message."""
    try:
        from rich.console import Console

        Console().print(f"[bold red]Error:[/bold red] {msg}")
    except ImportError:
        print(f"Error: {msg}", file=sys.stderr)


def _get_config(config_path: Optional[str], host: Optional[str],
                port: Optional[int], model: Optional[str],
                provider: Optional[str]):
    """Create a Config from options."""
    from nexusmind.core.config import Config

    if config_path:
        config = Config.from_file(config_path)
    else:
        config = Config.from_env()

    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if model:
        config.model.default_model = model
    if provider:
        config.model.provider = provider

    return config


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="nexusmind")
def main() -> None:
    """NexusMind - AI Agent with Persistent Memory and Multi-Provider LLM Support."""
    pass


# ---------------------------------------------------------------------------
# Start command - launch web UI + API
# ---------------------------------------------------------------------------


@main.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", "-p", default=None, type=int, help="Server port")
@click.option("--model", "-m", default=None, help="Default model")
@click.option("--provider", default=None, help="LLM provider")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def start(
    host: Optional[str],
    port: Optional[int],
    model: Optional[str],
    provider: Optional[str],
    config_path: Optional[str],
) -> None:
    """Launch the NexusMind web UI and API server."""
    try:
        import uvicorn
    except ImportError:
        _echo_error(
            "uvicorn is required. Install with: pip install nexusmind[api]"
        )
        sys.exit(1)

    config = _get_config(config_path, host, port, model, provider)
    config.ensure_data_dirs()

    _echo(f"[bold green]Starting NexusMind v0.1.0[/bold green]")
    _echo(f"  Provider: {config.model.provider}")
    _echo(f"  Model:    {config.model.default_model}")
    _echo(f"  API:      http://{config.server.host}:{config.server.port}")
    _echo(f"  Docs:     http://{config.server.host}:{config.server.port}/docs")
    _echo("")

    uvicorn.run(
        "nexusmind.api.app:create_app",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Serve command - API only
# ---------------------------------------------------------------------------


@main.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", "-p", default=None, type=int, help="Server port")
@click.option("--model", "-m", default=None, help="Default model")
@click.option("--provider", default=None, help="LLM provider")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def serve(
    host: Optional[str],
    port: Optional[int],
    model: Optional[str],
    provider: Optional[str],
    config_path: Optional[str],
) -> None:
    """Start the API server only (no web UI)."""
    try:
        import uvicorn
    except ImportError:
        _echo_error(
            "uvicorn is required. Install with: pip install nexusmind[api]"
        )
        sys.exit(1)

    config = _get_config(config_path, host, port, model, provider)
    config.ensure_data_dirs()

    _echo(f"[bold green]NexusMind API Server v0.1.0[/bold green]")
    _echo(f"  Listening on http://{config.server.host}:{config.server.port}")

    uvicorn.run(
        "nexusmind.api.app:create_app",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Chat command - interactive REPL
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--provider", default=None, help="LLM provider")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def chat(
    model: Optional[str],
    provider: Optional[str],
    config_path: Optional[str],
) -> None:
    """Start an interactive chat session in the terminal."""
    from nexusmind.core.config import Config
    from nexusmind.core.engine import NexusMind

    config = _get_config(config_path, None, None, model, provider)
    config.ensure_data_dirs()

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt

        console = Console()
    except ImportError:
        _echo_error("rich is required. Install with: pip install rich")
        sys.exit(1)

    _echo(f"[bold green]NexusMind Chat v0.1.0[/bold green]")
    _echo(f"  Provider: {config.model.provider}")
    _echo(f"  Model:    {config.model.default_model}")
    _echo('  Type "exit" or "quit" to leave, "/help" for commands')
    _echo("")

    async def _chat_loop() -> None:
        mind = NexusMind(config=config)
        try:
            while True:
                try:
                    user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if user_input.strip() == "/help":
                    console.print(Panel(
                        "[bold]Commands:[/bold]\n"
                        "  /help     - Show this help\n"
                        "  /stats    - Show system statistics\n"
                        "  /memory   - Show memory stats\n"
                        "  /skills   - List available skills\n"
                        "  /models   - List available models\n"
                        "  /clear    - Clear conversation history\n"
                        "  /save     - Save configuration\n"
                        "  exit/quit - Exit chat",
                        title="NexusMind Help",
                    ))
                    continue
                if user_input.strip() == "/stats":
                    stats = mind.get_stats()
                    console.print(Panel(json.dumps(stats, indent=2, default=str), title="System Stats"))
                    continue
                if user_input.strip() == "/memory":
                    stats = mind.memory.get_stats()
                    console.print(Panel(json.dumps(stats, indent=2), title="Memory Stats"))
                    continue
                if user_input.strip() == "/skills":
                    skills = mind.skills.list_skills()
                    if skills:
                        for s in skills:
                            console.print(f"  [bold]{s.name}[/bold]: {s.description}")
                    else:
                        console.print("  No skills learned yet.")
                    continue
                if user_input.strip() == "/models":
                    models = await mind.list_models()
                    for m in models:
                        active = " [green](active)[/green]" if m.get("active") else ""
                        console.print(f"  {m['provider']}/{m['id']}{active}")
                    continue
                if user_input.strip() == "/clear":
                    mind.memory.short_term.clear()
                    console.print("[yellow]Conversation history cleared.[/yellow]")
                    continue
                if user_input.strip() == "/save":
                    path = config.save()
                    console.print(f"[green]Configuration saved to {path}[/green]")
                    continue

                # Send message
                with console.status("[bold green]Thinking...[/bold green]"):
                    try:
                        response = await mind.chat(user_input)
                    except Exception as e:
                        console.print(f"[bold red]Error:[/bold red] {e}")
                        continue

                console.print(Panel(
                    response.content,
                    title=f"[bold green]NexusMind[/bold green] ({response.model})",
                    border_style="green",
                ))

        finally:
            await mind.close()

    asyncio.run(_chat_loop())


# ---------------------------------------------------------------------------
# Model commands
# ---------------------------------------------------------------------------


@main.group()
def model() -> None:
    """Manage LLM models."""
    pass


@model.command("list")
@click.option("--provider", default=None, help="Filter by provider")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def model_list(provider: Optional[str], config_path: Optional[str]) -> None:
    """List available models."""
    from nexusmind.core.config import Config
    from nexusmind.core.engine import NexusMind

    config = _get_config(config_path, None, None, None, provider)

    async def _list() -> None:
        mind = NexusMind(config=config)
        try:
            models = await mind.list_models()
            if not models:
                _echo("No models found. Check your provider configuration.")
                return

            for m in models:
                active = " *" if m.get("active") else ""
                provider_str = f"[{m['provider']}]"
                ctx = f" (ctx: {m['context_length']})" if m.get("context_length") else ""
                _echo(f"  {provider_str:20s} {m['id']:40s}{ctx}{active}")

            _echo(f"\n  Total: {len(models)} models")
            _echo("  * = active model")
        finally:
            await mind.close()

    asyncio.run(_list())


@model.command("switch")
@click.argument("model_name")
@click.option("--provider", default=None, help="Provider name")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def model_switch(model_name: str, provider: Optional[str], config_path: Optional[str]) -> None:
    """Switch the active model."""
    from nexusmind.core.config import Config
    from nexusmind.core.engine import NexusMind

    config = _get_config(config_path, None, None, None, provider)

    async def _switch() -> None:
        mind = NexusMind(config=config)
        try:
            await mind.switch_model(model_name, provider)
            _echo(f"[green]Switched to model: {model_name}[/green]")
            config.save()
        finally:
            await mind.close()

    asyncio.run(_switch())


# ---------------------------------------------------------------------------
# Memory commands
# ---------------------------------------------------------------------------


@main.group()
def memory() -> None:
    """Manage persistent memory."""
    pass


@memory.command("list")
@click.option("--category", default=None, help="Filter by category")
@click.option("--limit", "-n", default=20, help="Number of results")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def memory_list(category: Optional[str], limit: int, config_path: Optional[str]) -> None:
    """List stored memories."""
    from nexusmind.core.config import Config
    from nexusmind.core.memory import MemoryManager

    config = _get_config(config_path, None, None, None, None)
    mm = MemoryManager(config=config.memory)
    try:
        memories = mm.long_term.list_all(category=category, limit=limit)
        if not memories:
            _echo("No memories stored yet.")
            return

        for m in memories:
            _echo(f"  [{m.category}] {m.key}: {m.value[:80]}")

        _echo(f"\n  Total: {mm.long_term.count(category)} memories")
    finally:
        mm.close()


@memory.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def memory_search(query: str, limit: int, config_path: Optional[str]) -> None:
    """Search memories by keyword."""
    from nexusmind.core.config import Config
    from nexusmind.core.memory import MemoryManager

    config = _get_config(config_path, None, None, None, None)
    mm = MemoryManager(config=config.memory)
    try:
        results = mm.recall(query, limit=limit)
        if not results:
            _echo(f"No memories found matching '{query}'.")
            return

        for m in results:
            _echo(f"  [{m.category}] {m.key}: {m.value[:100]}")
    finally:
        mm.close()


@memory.command("export")
@click.argument("output_path", type=click.Path())
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def memory_export(output_path: str, config_path: Optional[str]) -> None:
    """Export all memories to a JSON file."""
    from nexusmind.core.config import Config
    from nexusmind.core.memory import MemoryManager

    config = _get_config(config_path, None, None, None, None)
    mm = MemoryManager(config=config.memory)
    try:
        data = mm.export_memories()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _echo(f"[green]Exported {len(data.get('memories', []))} memories to {output_path}[/green]")
    finally:
        mm.close()


# ---------------------------------------------------------------------------
# Skill commands
# ---------------------------------------------------------------------------


@main.group()
def skill() -> None:
    """Manage auto-evolved skills."""
    pass


@skill.command("list")
@click.option("--tag", default=None, help="Filter by tag")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def skill_list(tag: Optional[str], config_path: Optional[str]) -> None:
    """List available skills."""
    from nexusmind.core.config import Config
    from nexusmind.core.skills import SkillEngine

    config = _get_config(config_path, None, None, None, None)
    engine = SkillEngine(persist_dir=config.memory.persist_dir)
    try:
        skills = engine.list_skills(tag=tag)
        if not skills:
            _echo("No skills learned yet. Skills are auto-detected from repeated patterns.")
            return

        for s in skills:
            rate = f"{s.success_rate:.0%}" if s.usage_count > 0 else "N/A"
            _echo(f"  [bold]{s.name}[/bold]: {s.description}")
            _echo(f"    Usage: {s.usage_count}, Success: {rate}, Source: {s.source}")

        _echo(f"\n  Total: {len(skills)} skills")
    finally:
        engine.close()


@skill.command("export")
@click.argument("output_path", type=click.Path())
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def skill_export(output_path: str, config_path: Optional[str]) -> None:
    """Export all skills to a JSON file."""
    from nexusmind.core.config import Config
    from nexusmind.core.skills import SkillEngine

    config = _get_config(config_path, None, None, None, None)
    engine = SkillEngine(persist_dir=config.memory.persist_dir)
    try:
        data = engine.export_skills()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _echo(f"[green]Exported {len(data.get('skills', []))} skills to {output_path}[/green]")
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Schedule commands
# ---------------------------------------------------------------------------


@main.group()
def schedule() -> None:
    """Manage scheduled tasks."""
    pass


@schedule.command("list")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def schedule_list(config_path: Optional[str]) -> None:
    """List scheduled tasks."""
    from nexusmind.core.config import Config
    from nexusmind.core.scheduler import TaskScheduler

    config = _get_config(config_path, None, None, None, None)
    scheduler = TaskScheduler(config=config.scheduler)
    try:
        tasks = scheduler.list_tasks()
        if not tasks:
            _echo("No scheduled tasks.")
            return

        for t in tasks:
            next_str = (
                f"next: {t.next_run}" if t.next_run > 0 else "no schedule"
            )
            _echo(f"  [bold]{t.name}[/bold] ({t.id}) - {t.status.value} - {next_str}")
            _echo(f"    Schedule: {t.schedule or 'one-time'}")
            _echo(f"    Runs: {t.run_count}")

        _echo(f"\n  Total: {len(tasks)} tasks")
    finally:
        scheduler.close()


@schedule.command("create")
@click.option("--name", required=True, help="Task name")
@click.option("--prompt", required=True, help="Task prompt")
@click.option("--schedule", default="", help="Schedule expression")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def schedule_create(
    name: str,
    prompt: str,
    schedule: str,
    model: Optional[str],
    config_path: Optional[str],
) -> None:
    """Create a new scheduled task."""
    from nexusmind.core.config import Config
    from nexusmind.core.scheduler import TaskScheduler

    config = _get_config(config_path, None, None, None, None)
    scheduler = TaskScheduler(config=config.scheduler)
    try:
        task = scheduler.schedule_task(
            name=name,
            prompt=prompt,
            schedule=schedule,
            model=model,
        )
        _echo(f"[green]Created task '{task.name}' (id: {task.id})[/green]")
        if task.next_run > 0:
            from datetime import datetime, timezone

            next_dt = datetime.fromtimestamp(task.next_run, tz=timezone.utc)
            _echo(f"  Next run: {next_dt.isoformat()}")
    finally:
        scheduler.close()


@schedule.command("run")
@click.argument("task_id")
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
def schedule_run(task_id: str, config_path: Optional[str]) -> None:
    """Run a scheduled task immediately."""
    from nexusmind.core.config import Config
    from nexusmind.core.scheduler import TaskScheduler

    config = _get_config(config_path, None, None, None, None)

    async def _run() -> None:
        from nexusmind.core.engine import NexusMind

        mind = NexusMind(config=config)
        try:
            result = await mind.scheduler.run_now(task_id)
            _echo(f"Task '{task_id}' completed: {result.status.value}")
            if result.output:
                _echo(f"  Output: {result.output[:500]}")
            if result.error:
                _echo(f"  Error: {result.error}")
        finally:
            await mind.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
