"""LLM Provider Abstraction Layer for NexusMind.

Provides a unified interface for interacting with multiple LLM providers
including Ollama, OpenAI, Anthropic, and OpenRouter. All providers use
httpx.AsyncClient for HTTP communication.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from nexusmind.core.config import ModelConfig, ProviderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ChatResponse:
    """Response from a chat completion request."""

    content: str
    model: str
    provider: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: str
    context_length: int | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Base Provider
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.

    Subclasses must implement chat(), chat_stream(), and list_models().
    """

    def __init__(self, config: ProviderConfig, model_config: ModelConfig) -> None:
        self.config = config
        self.model_config = model_config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Provider name."""
        return self.config.name

    @property
    def base_url(self) -> str:
        """Provider API base URL."""
        return self.config.base_url

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create an httpx.AsyncClient instance.

        Returns:
            A configured httpx.AsyncClient.
        """
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of chat messages.
            model: Model name override.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ChatResponse with the model's reply.
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response.

        Args:
            messages: List of chat messages.
            model: Model name override.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Chunks of the response text.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models from this provider.

        Returns:
            List of ModelInfo objects.
        """
        ...

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get information about a specific model.

        Args:
            model_id: The model identifier.

        Returns:
            ModelInfo if found, None otherwise.
        """
        models = await self.list_models()
        for m in models:
            if m.id == model_id:
                return m
        return None

    def _format_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to API-compatible dicts.

        Args:
            messages: List of ChatMessage objects.

        Returns:
            List of message dictionaries.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                d["name"] = msg.name
            if msg.tool_calls:
                d["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            result.append(d)
        return result


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------


class OllamaProvider(BaseProvider):
    """Provider for local Ollama instances.

    Uses the Ollama REST API at http://localhost:11434/api/.
    """

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama.

        Args:
            messages: List of chat messages.
            model: Model name (e.g., 'llama3', 'mistral').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters (e.g., 'num_predict').

        Returns:
            ChatResponse with the model's reply.
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model or self.model_config.default_model,
            "messages": self._format_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature or self.model_config.temperature,
                "num_predict": max_tokens or self.model_config.max_tokens,
                "top_p": self.model_config.top_p,
            },
        }
        payload["options"].update(kwargs)

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return ChatResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", ""),
            provider=self.name,
            finish_reason=data.get("done_reason", "stop"),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            },
            raw=data,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response from Ollama.

        Yields:
            Chunks of the response text.
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model or self.model_config.default_model,
            "messages": self._format_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature or self.model_config.temperature,
                "num_predict": max_tokens or self.model_config.max_tokens,
                "top_p": self.model_config.top_p,
            },
        }
        payload["options"].update(kwargs)

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                import json

                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    async def list_models(self) -> list[ModelInfo]:
        """List locally available Ollama models.

        Returns:
            List of ModelInfo objects.
        """
        client = self._get_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models: list[ModelInfo] = []
            for m in data.get("models", []):
                name = m.get("name", "")
                models.append(
                    ModelInfo(
                        id=name,
                        name=name,
                        provider=self.name,
                        context_length=m.get("context_length"),
                        description=m.get("description", ""),
                    )
                )
            return models
        except httpx.HTTPError:
            logger.warning("Could not connect to Ollama at %s", self.config.base_url)
            return []


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API.

    Uses the OpenAI Chat Completions API.
    """

    def _get_client(self) -> httpx.AsyncClient:
        """Get httpx client with OpenAI auth headers."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI.

        Args:
            messages: List of chat messages.
            model: Model name (e.g., 'gpt-4o', 'gpt-3.5-turbo').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with the model's reply.
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model or self.model_config.default_model,
            "messages": self._format_messages(messages),
            "temperature": temperature or self.model_config.temperature,
            "max_tokens": max_tokens or self.model_config.max_tokens,
            "top_p": self.model_config.top_p,
            "frequency_penalty": self.model_config.frequency_penalty,
            "presence_penalty": self.model_config.presence_penalty,
        }
        if self.model_config.stop_sequences:
            payload["stop"] = self.model_config.stop_sequences
        payload.update(kwargs)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return ChatResponse(
            content=message.get("content", ""),
            model=data.get("model", ""),
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            raw=data,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response from OpenAI.

        Yields:
            Chunks of the response text.
        """
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model or self.model_config.default_model,
            "messages": self._format_messages(messages),
            "temperature": temperature or self.model_config.temperature,
            "max_tokens": max_tokens or self.model_config.max_tokens,
            "top_p": self.model_config.top_p,
            "stream": True,
        }
        payload.update(kwargs)

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                import json

                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    async def list_models(self) -> list[ModelInfo]:
        """List available OpenAI models.

        Returns:
            List of ModelInfo objects.
        """
        client = self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            models: list[ModelInfo] = []
            for m in data.get("data", []):
                models.append(
                    ModelInfo(
                        id=m.get("id", ""),
                        name=m.get("id", ""),
                        provider=self.name,
                        description=m.get("description", ""),
                    )
                )
            return sorted(models, key=lambda x: x.id)
        except httpx.HTTPError as e:
            logger.warning("Could not list OpenAI models: %s", e)
            return []


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude API.

    Uses the Anthropic Messages API with anthropic-version header.
    """

    def _get_client(self) -> httpx.AsyncClient:
        """Get httpx client with Anthropic auth headers."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            if self.config.api_key:
                headers["x-api-key"] = self.config.api_key
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    def _format_messages_anthropic(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Anthropic API format.

        Anthropic requires system prompt separate from messages.

        Args:
            messages: List of ChatMessage objects.

        Returns:
            Tuple of (system_prompt, messages_list).
        """
        system_prompt: str | None = None
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                formatted.append({"role": msg.role, "content": msg.content})

        return system_prompt, formatted

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to Anthropic.

        Args:
            messages: List of chat messages.
            model: Model name (e.g., 'claude-3-opus-20240229').
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with the model's reply.
        """
        client = self._get_client()
        system_prompt, formatted_messages = self._format_messages_anthropic(messages)

        payload: dict[str, Any] = {
            "model": model or self.model_config.default_model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.model_config.max_tokens,
            "temperature": temperature or self.model_config.temperature,
            "top_p": self.model_config.top_p,
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(kwargs)

        response = await client.post("/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        content_blocks = data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage = data.get("usage", {})

        return ChatResponse(
            content=text_content,
            model=data.get("model", ""),
            provider=self.name,
            finish_reason=data.get("stop_reason", "end_turn"),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0),
            },
            raw=data,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response from Anthropic.

        Yields:
            Chunks of the response text.
        """
        client = self._get_client()
        system_prompt, formatted_messages = self._format_messages_anthropic(messages)

        payload: dict[str, Any] = {
            "model": model or self.model_config.default_model,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.model_config.max_tokens,
            "temperature": temperature or self.model_config.temperature,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(kwargs)

        async with client.stream("POST", "/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                import json

                try:
                    data = json.loads(data_str)
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue

    async def list_models(self) -> list[ModelInfo]:
        """List available Anthropic Claude models.

        Returns:
            List of ModelInfo objects.
        """
        # Anthropic doesn't have a public models listing API,
        # so we return known models.
        known_models = [
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4",
                provider=self.name,
                context_length=200000,
                description="Latest Claude Sonnet model",
            ),
            ModelInfo(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=self.name,
                context_length=200000,
                description="Balanced performance and speed",
            ),
            ModelInfo(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                provider=self.name,
                context_length=200000,
                description="Fast and efficient",
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=self.name,
                context_length=200000,
                description="Most capable Claude model",
            ),
        ]
        return known_models


# ---------------------------------------------------------------------------
# OpenRouter Provider
# ---------------------------------------------------------------------------


class OpenRouterProvider(OpenAIProvider):
    """Provider for OpenRouter API.

    OpenRouter uses an OpenAI-compatible API format with additional
    provider routing capabilities.
    """

    def _get_client(self) -> httpx.AsyncClient:
        """Get httpx client with OpenRouter auth headers."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nexusmind.local",
                "X-Title": "NexusMind",
            }
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        """List available OpenRouter models.

        Returns:
            List of ModelInfo objects.
        """
        client = self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            models: list[ModelInfo] = []
            for m in data.get("data", []):
                models.append(
                    ModelInfo(
                        id=m.get("id", ""),
                        name=m.get("name", m.get("id", "")),
                        provider=self.name,
                        context_length=m.get("context_length"),
                        description=m.get("description", ""),
                    )
                )
            return sorted(models, key=lambda x: x.id)
        except httpx.HTTPError as e:
            logger.warning("Could not list OpenRouter models: %s", e)
            return []


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------


class ProviderFactory:
    """Factory for creating LLM provider instances.

    Example::

        provider = ProviderFactory.create("openai", provider_config, model_config)
        response = await provider.chat(messages)
    """

    _registry: dict[str, type[BaseProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        provider_config: ProviderConfig,
        model_config: ModelConfig,
    ) -> BaseProvider:
        """Create a provider instance.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'ollama').
            provider_config: Provider configuration.
            model_config: Model configuration.

        Returns:
            An instance of the requested provider.

        Raises:
            ValueError: If the provider name is not recognized.
        """
        provider_cls = cls._registry.get(provider_name.lower())
        if provider_cls is None:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available: {available}"
            )
        return provider_cls(provider_config, model_config)

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseProvider]) -> None:
        """Register a custom provider class.

        Args:
            name: Provider name.
            provider_cls: Provider class to register.
        """
        if not issubclass(provider_cls, BaseProvider):
            raise TypeError(
                f"{provider_cls} must be a subclass of BaseProvider"
            )
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def available_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider name strings.
        """
        return list(cls._registry.keys())
