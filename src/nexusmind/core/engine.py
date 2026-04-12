"""NexusMind Engine - The main orchestration layer.

Combines all subsystems (providers, memory, skills, scheduler, agents, RAG)
into a unified AI agent experience.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from nexusmind.core.agents import AgentOrchestrator
from nexusmind.core.config import Config, ModelConfig
from nexusmind.core.memory import MemoryManager
from nexusmind.core.providers import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ProviderFactory,
)
from nexusmind.core.rag import RAGPipeline
from nexusmind.core.scheduler import TaskScheduler
from nexusmind.core.skills import SkillEngine

logger = logging.getLogger(__name__)

# Default system prompt
_DEFAULT_SYSTEM_PROMPT = """You are NexusMind, an advanced AI assistant with persistent memory,
auto-evolving skills, and multi-provider LLM support. You have access to the user's
conversation history and can remember important information across sessions.

Key capabilities:
- Persistent memory: You remember important details from past conversations
- Skill evolution: You can learn and improve from repeated patterns
- Multi-agent collaboration: You can delegate to specialized agents
- RAG: You can search through ingested documents for context

Always be helpful, accurate, and proactive. When you detect repeated patterns,
suggest creating reusable skills. Store important user preferences and context
in memory for future reference."""


class NexusMind:
    """Main NexusMind AI Agent Engine.

    Orchestrates all subsystems to provide a unified AI agent experience
    with persistent memory, skill evolution, scheduling, and multi-provider
    LLM support.

    Example::

        mind = NexusMind()
        response = await mind.chat("Hello, what can you do?")
        async for chunk in mind.chat_stream("Tell me a story"):
            print(chunk, end="")
        mind.ingest_documents(["/path/to/docs/"])
        models = await mind.list_models()
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the NexusMind engine.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or Config.from_env()
        self.config.ensure_data_dirs()

        # Core subsystems
        self.memory = MemoryManager(config=self.config.memory)
        self.skills = SkillEngine(persist_dir=self.config.memory.persist_dir)
        self.rag = RAGPipeline()
        self.scheduler = TaskScheduler(
            config=self.config.scheduler,
            executor=self._execute_scheduled_task,
            notification_config={
                "telegram_token": self.config.notification.telegram_token,
                "telegram_chat_id": self.config.notification.telegram_chat_id,
                "discord_webhook": self.config.notification.discord_webhook,
                "slack_token": self.config.notification.slack_token,
                "slack_channel": self.config.notification.slack_channel,
            },
        )

        # Provider management
        self._providers: dict[str, BaseProvider] = {}
        self._active_provider: BaseProvider | None = None
        self._active_model: str = self.config.model.default_model

        # Agent orchestrator (lazy-initialized)
        self._orchestrator: AgentOrchestrator | None = None

        # System prompt
        self._system_prompt: str = _DEFAULT_SYSTEM_PROMPT

        # Statistics
        self._stats = {
            "total_chats": 0,
            "total_tokens": 0,
            "start_time": time.time(),
        }

        # Initialize default provider
        self._init_provider(self.config.model.provider)

    def _init_provider(self, provider_name: str) -> None:
        """Initialize a provider by name.

        Args:
            provider_name: Name of the provider to initialize.
        """
        try:
            provider_config = self.config.get_provider_config(provider_name)
            self._active_provider = ProviderFactory.create(
                provider_name=provider_name,
                provider_config=provider_config,
                model_config=self.config.model,
            )
            self._providers[provider_name] = self._active_provider
            logger.info("Initialized provider: %s", provider_name)
        except Exception as e:
            logger.error("Failed to initialize provider '%s': %s", provider_name, e)

    def _get_provider(self, provider_name: str | None = None) -> BaseProvider:
        """Get a provider, creating it if necessary.

        Args:
            provider_name: Provider name. Defaults to active provider.

        Returns:
            The requested BaseProvider.

        Raises:
            RuntimeError: If no provider is available.
        """
        name = provider_name or self.config.model.provider
        if name not in self._providers:
            self._init_provider(name)
        provider = self._providers.get(name)
        if provider is None:
            raise RuntimeError(
                f"No provider available. Tried to initialize '{name}'. "
                f"Check your configuration and API keys."
            )
        return provider

    @property
    def orchestrator(self) -> AgentOrchestrator:
        """Get or create the agent orchestrator.

        Returns:
            AgentOrchestrator instance.
        """
        if self._orchestrator is None:
            self._orchestrator = AgentOrchestrator(
                memory=self.memory,
                provider=self._active_provider,
            )
        return self._orchestrator

    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom system prompt.

        Args:
            prompt: The new system prompt.
        """
        self._system_prompt = prompt

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        model: str | None = None,
        stream: bool = False,
        provider: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat message and get a response.

        This is the main chat method that combines memory, skills, and
        RAG context to provide an augmented response.

        Args:
            messages: Either a string message or a list of message dicts
                     with 'role' and 'content' keys.
            model: Model name override.
            stream: If True, returns the first chunk (use chat_stream for full streaming).
            provider: Provider name override.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ChatResponse with the model's reply.
        """
        # Normalize input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Store user messages in memory
        for msg in messages:
            if msg["role"] == "user":
                self.memory.add_message("user", msg["content"])

        # Build augmented messages with context
        augmented = self._build_augmented_messages(messages)

        # Get provider and send request
        provider_instance = self._get_provider(provider)
        response = await provider_instance.chat(
            messages=augmented,
            model=model or self._active_model,
            **kwargs,
        )

        # Store assistant response in memory
        self.memory.add_message("assistant", response.content)

        # Update stats
        self._stats["total_chats"] += 1
        self._stats["total_tokens"] += response.usage.get("total_tokens", 0)

        # Learn from interaction
        self.skills.learn_from_interaction(
            conversation=messages + [{"role": "assistant", "content": response.content}],
            outcome="success",
        )

        return response

    async def chat_stream(
        self,
        messages: list[dict[str, str]] | str,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat response chunk by chunk.

        Args:
            messages: Either a string message or a list of message dicts.
            model: Model name override.
            provider: Provider name override.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Chunks of the response text.
        """
        # Normalize input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Store user messages in memory
        for msg in messages:
            if msg["role"] == "user":
                self.memory.add_message("user", msg["content"])

        # Build augmented messages
        augmented = self._build_augmented_messages(messages)

        # Stream from provider
        provider_instance = self._get_provider(provider)
        full_response = ""
        async for chunk in provider_instance.chat_stream(
            messages=augmented,
            model=model or self._active_model,
            **kwargs,
        ):
            full_response += chunk
            yield chunk

        # Store full response in memory
        self.memory.add_message("assistant", full_response)

        # Update stats
        self._stats["total_chats"] += 1

        # Learn from interaction
        self.skills.learn_from_interaction(
            conversation=messages + [{"role": "assistant", "content": full_response}],
            outcome="success",
        )

    def _build_augmented_messages(
        self, messages: list[dict[str, str]]
    ) -> list[ChatMessage]:
        """Build augmented message list with system prompt, memory, and RAG context.

        Args:
            messages: Original user messages.

        Returns:
            Augmented list of ChatMessage objects.
        """
        result: list[ChatMessage] = []

        # System prompt with memory context
        system_content = self._system_prompt

        # Add relevant memory context
        if messages:
            last_user_msg = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if last_user_msg:
                memory_context = self.memory.get_context(last_user_msg, max_tokens=1000)
                if memory_context:
                    system_content += f"\n\n## Your Memory Context\n{memory_context}"

                # Add RAG context if available
                rag_context = self.rag.build_context(last_user_msg, max_tokens=1000)
                if rag_context:
                    system_content += f"\n\n## Relevant Documents\n{rag_context}"

                # Suggest relevant skills
                suggested_skills = self.skills.suggest_skills(last_user_msg)
                if suggested_skills:
                    skill_names = [s.name for s in suggested_skills]
                    system_content += (
                        f"\n\n## Available Skills\n"
                        f"You have these relevant skills: {', '.join(skill_names)}. "
                        f"Use them if appropriate."
                    )

        result.append(ChatMessage(role="system", content=system_content))

        # Add conversation messages
        for msg in messages:
            result.append(ChatMessage(role=msg["role"], content=msg["content"]))

        return result

    def ingest_documents(self, paths: list[str | Path]) -> int:
        """Ingest documents into the RAG pipeline.

        Args:
            paths: List of file paths or directories to ingest.

        Returns:
            Number of chunks created.
        """
        return self.rag.ingest(paths)

    async def switch_model(
        self,
        model: str,
        provider: str | None = None,
    ) -> None:
        """Switch the active model and/or provider.

        Args:
            model: Model name to switch to.
            provider: Provider name. If None, keeps current provider.
        """
        if provider:
            self._init_provider(provider)
            self.config.model.provider = provider

        self._active_model = model
        self.config.model.default_model = model
        logger.info("Switched to model: %s (provider: %s)", model, provider or "current")

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from all configured providers.

        Returns:
            List of model info dictionaries.
        """
        all_models: list[dict[str, Any]] = []

        for provider_name in self.config.providers:
            try:
                provider = self._get_provider(provider_name)
                models = await provider.list_models()
                for m in models:
                    all_models.append({
                        "id": m.id,
                        "name": m.name,
                        "provider": m.provider,
                        "context_length": m.context_length,
                        "description": m.description,
                        "active": m.id == self._active_model
                        and m.provider == self.config.model.provider,
                    })
            except Exception as e:
                logger.warning("Could not list models from '%s': %s", provider_name, e)

        return all_models

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics.

        Returns:
            Dictionary with all system statistics.
        """
        uptime = time.time() - self._stats["start_time"]
        return {
            "version": "0.1.0",
            "uptime_seconds": round(uptime, 1),
            "active_model": self._active_model,
            "active_provider": self.config.model.provider,
            "total_chats": self._stats["total_chats"],
            "total_tokens": self._stats["total_tokens"],
            "memory": self.memory.get_stats(),
            "skills": self.skills.get_stats(),
            "scheduler": self.scheduler.get_stats(),
            "rag": self.rag.get_stats(),
            "available_providers": list(self.config.providers.keys()),
        }

    async def _execute_scheduled_task(
        self, prompt: str, model: str | None = None
    ) -> str:
        """Execute a scheduled task prompt.

        Used as the executor callback for the task scheduler.

        Args:
            prompt: The task prompt.
            model: Model to use.

        Returns:
            The model's response text.
        """
        try:
            response = await self.chat(prompt, model=model)
            return response.content
        except Exception as e:
            logger.error("Scheduled task execution failed: %s", e)
            return f"[Error: {str(e)}]"

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        # Close providers
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()

        # Close subsystems
        self.memory.close()
        self.skills.close()
        self.scheduler.close()

        logger.info("NexusMind engine closed")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def create_nexusmind(
    config: Config | None = None,
    config_path: str | Path | None = None,
) -> NexusMind:
    """Create and initialize a NexusMind instance.

    Convenience function for creating a NexusMind engine with
    optional configuration.

    Args:
        config: Configuration object. Takes priority over config_path.
        config_path: Path to a configuration file.

    Returns:
        An initialized NexusMind instance.
    """
    if config is None and config_path is not None:
        config = Config.from_file(config_path)
    return NexusMind(config=config)
