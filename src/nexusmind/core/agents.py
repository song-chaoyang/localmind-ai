"""Multi-Agent Collaboration System for NexusMind.

Provides specialized agents for different tasks (coding, research, data analysis)
and an orchestrator for coordinating multi-agent workflows.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from nexusmind.core.memory import MemoryManager
from nexusmind.core.providers import BaseProvider, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base exception for agent operations."""


class AgentNotFoundError(AgentError):
    """Raised when an agent type is not found."""


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all specialized agents.

    Each agent has a specific role, system prompt, and can interact
    with LLM providers through a shared memory system.
    """

    def __init__(
        self,
        name: str,
        role: str,
        memory: MemoryManager,
        provider: BaseProvider | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.memory = memory
        self.provider = provider
        self._conversation_history: list[ChatMessage] = []

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            System prompt string.
        """
        return self._build_system_prompt()

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent type.

        Returns:
            System prompt string.
        """
        ...

    async def chat(
        self,
        message: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a message to this agent.

        Args:
            message: The user message.
            model: Model override.
            **kwargs: Additional provider parameters.

        Returns:
            ChatResponse from the agent.

        Raises:
            AgentError: If no provider is configured.
        """
        if self.provider is None:
            raise AgentError(f"No provider configured for agent '{self.name}'")

        # Add user message to history
        self._conversation_history.append(ChatMessage(role="user", content=message))

        # Build messages with system prompt
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        messages.extend(self._conversation_history)

        response = await self.provider.chat(messages, model=model, **kwargs)

        # Add assistant response to history
        self._conversation_history.append(
            ChatMessage(role="assistant", content=response.content)
        )

        # Store in memory
        self.memory.add_message("user", message)
        self.memory.add_message("assistant", response.content)

        return response

    async def chat_stream(
        self,
        message: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from this agent.

        Args:
            message: The user message.
            model: Model override.
            **kwargs: Additional provider parameters.

        Yields:
            Chunks of the response text.
        """
        if self.provider is None:
            raise AgentError(f"No provider configured for agent '{self.name}'")

        self._conversation_history.append(ChatMessage(role="user", content=message))
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        messages.extend(self._conversation_history)

        full_response = ""
        async for chunk in self.provider.chat_stream(messages, model=model, **kwargs):
            full_response += chunk
            yield chunk

        self._conversation_history.append(
            ChatMessage(role="assistant", content=full_response)
        )
        self.memory.add_message("user", message)
        self.memory.add_message("assistant", full_response)

    def clear_history(self) -> None:
        """Clear the conversation history for this agent."""
        self._conversation_history.clear()

    def get_history(self) -> list[ChatMessage]:
        """Get the conversation history.

        Returns:
            List of ChatMessage objects.
        """
        return list(self._conversation_history)


# ---------------------------------------------------------------------------
# Coder Agent
# ---------------------------------------------------------------------------


class CoderAgent(BaseAgent):
    """Specialized agent for code generation, review, and debugging.

    Has expertise in multiple programming languages, best practices,
    and can generate, review, refactor, and debug code.
    """

    def _build_system_prompt(self) -> str:
        """Build the coder agent system prompt."""
        return """You are an expert coding assistant with deep knowledge of software engineering.

Your expertise includes:
- Code generation in Python, JavaScript, TypeScript, Rust, Go, Java, C++, and more
- Code review with focus on correctness, performance, security, and maintainability
- Debugging complex issues with systematic approaches
- Refactoring and optimization
- Testing strategies (unit, integration, e2e)
- Architecture and design patterns
- DevOps and CI/CD best practices

Guidelines:
- Always write clean, well-documented code with type hints
- Follow language-specific conventions and best practices
- Consider edge cases and error handling
- Explain your reasoning when making important decisions
- Suggest improvements when reviewing code
- Use appropriate design patterns
- Consider performance implications

When generating code:
1. Understand the requirements fully
2. Plan the approach
3. Write clean, modular code
4. Add appropriate error handling
5. Include documentation
6. Consider testability"""


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent):
    """Specialized agent for research and documentation.

    Excels at gathering information, summarizing findings, and
    creating comprehensive documentation.
    """

    def _build_system_prompt(self) -> str:
        """Build the research agent system prompt."""
        return """You are an expert research assistant with strong analytical skills.

Your expertise includes:
- Information synthesis and analysis
- Technical documentation writing
- Research methodology and fact-checking
- Summarization of complex topics
- Comparative analysis
- Trend identification and forecasting

Guidelines:
- Provide well-structured, comprehensive responses
- Cite sources and evidence when available
- Present multiple perspectives on complex topics
- Use clear headings and organized formatting
- Distinguish between facts, opinions, and hypotheses
- Highlight key findings and insights
- Suggest further research directions when appropriate

When researching:
1. Clarify the research question
2. Identify key information sources
3. Analyze and synthesize findings
4. Present results in a clear, organized manner
5. Note limitations and gaps
6. Suggest next steps"""


# ---------------------------------------------------------------------------
# Data Agent
# ---------------------------------------------------------------------------


class DataAgent(BaseAgent):
    """Specialized agent for data analysis and visualization.

    Handles data processing, statistical analysis, and generating
    insights from data.
    """

    def _build_system_prompt(self) -> str:
        """Build the data agent system prompt."""
        return """You are an expert data analyst with strong statistical and visualization skills.

Your expertise includes:
- Statistical analysis (descriptive, inferential, predictive)
- Data cleaning and preprocessing
- Data visualization (charts, graphs, dashboards)
- Machine learning basics
- SQL and database querying
- Python data ecosystem (pandas, numpy, matplotlib, seaborn)
- Exploratory data analysis (EDA)
- Report generation and data storytelling

Guidelines:
- Always validate data quality before analysis
- Use appropriate statistical methods
- Visualize data effectively to communicate insights
- Explain statistical findings in accessible terms
- Consider biases and limitations in data
- Provide actionable recommendations
- Use reproducible analysis methods

When analyzing data:
1. Understand the data and its context
2. Clean and preprocess as needed
3. Explore and summarize key characteristics
4. Apply appropriate analytical methods
5. Visualize important findings
6. Draw conclusions with appropriate caveats
7. Recommend actions based on insights"""


# ---------------------------------------------------------------------------
# Agent Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class CollaborationResult:
    """Result from a multi-agent collaboration."""

    task: str
    agent_results: dict[str, str] = field(default_factory=dict)
    final_summary: str = ""
    success: bool = True
    errors: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """Orchestrates multiple specialized agents for complex tasks.

    Manages agent creation, delegation, and multi-agent collaboration
    with shared memory.

    Example::

        orchestrator = AgentOrchestrator(memory, provider)
        result = await orchestrator.delegate("Fix this bug", "coder")
        collab = await orchestrator.collaborate(
            "Build a REST API",
            ["coder", "research"]
        )
    """

    _AGENT_TYPES: dict[str, type[BaseAgent]] = {
        "coder": CoderAgent,
        "research": ResearchAgent,
        "data": DataAgent,
    }

    def __init__(
        self,
        memory: MemoryManager,
        provider: BaseProvider | None = None,
    ) -> None:
        self.memory = memory
        self.provider = provider
        self._agents: dict[str, BaseAgent] = {}

    def _get_or_create_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent of the specified type.

        Args:
            agent_type: The agent type key ('coder', 'research', 'data').

        Returns:
            An instance of the requested agent.

        Raises:
            AgentNotFoundError: If the agent type is not recognized.
        """
        if agent_type not in self._AGENT_TYPES:
            available = ", ".join(self._AGENT_TYPES.keys())
            raise AgentNotFoundError(
                f"Unknown agent type '{agent_type}'. Available: {available}"
            )

        if agent_type not in self._agents:
            agent_cls = self._AGENT_TYPES[agent_type]
            self._agents[agent_type] = agent_cls(
                name=agent_type,
                role=agent_type,
                memory=self.memory,
                provider=self.provider,
            )

        return self._agents[agent_type]

    async def delegate(
        self,
        task: str,
        agent_type: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Delegate a task to a specific agent.

        Args:
            task: The task description.
            agent_type: The agent type to delegate to.
            model: Model override.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse from the agent.
        """
        agent = self._get_or_create_agent(agent_type)
        return await agent.chat(task, model=model, **kwargs)

    async def collaborate(
        self,
        task: str,
        agent_types: list[str],
        model: str | None = None,
    ) -> CollaborationResult:
        """Have multiple agents collaborate on a task.

        Each agent receives the task and produces output. The results
        are then synthesized into a final summary.

        Args:
            task: The task description.
            agent_types: List of agent types to involve.
            model: Model override.

        Returns:
            CollaborationResult with all agent outputs.
        """
        result = CollaborationResult(task=task)

        for agent_type in agent_types:
            try:
                agent = self._get_or_create_agent(agent_type)
                response = await agent.chat(task, model=model)
                result.agent_results[agent_type] = response.content
            except Exception as e:
                result.errors.append(f"{agent_type}: {str(e)}")
                result.agent_results[agent_type] = f"[Error: {str(e)}]"

        # Build summary
        parts = [f"## Task: {task}\n"]
        for agent_type, output in result.agent_results.items():
            parts.append(f"### {agent_type.title()} Agent\n{output}\n")

        if result.errors:
            result.success = False
            parts.append("### Errors\n")
            for error in result.errors:
                parts.append(f"- {error}\n")

        result.final_summary = "\n".join(parts)

        # Store collaboration in memory
        self.memory.remember(
            key=f"collaboration_{hash(task) % 10000}",
            value=f"Task: {task[:200]} | Agents: {', '.join(agent_types)} | "
                  f"Success: {result.success}",
            category="collaboration",
        )

        return result

    def share_memory(self) -> MemoryManager:
        """Get the shared memory system.

        Returns:
            The shared MemoryManager instance.
        """
        return self.memory

    def list_agents(self) -> list[dict[str, str]]:
        """List all available agent types.

        Returns:
            List of agent info dictionaries.
        """
        return [
            {"type": key, "class": cls.__name__}
            for key, cls in self._AGENT_TYPES.items()
        ]

    def get_agent(self, agent_type: str) -> BaseAgent | None:
        """Get an existing agent instance.

        Args:
            agent_type: The agent type key.

        Returns:
            The agent instance, or None if not created yet.
        """
        return self._agents.get(agent_type)

    def register_agent(self, agent_type: str, agent_cls: type[BaseAgent]) -> None:
        """Register a custom agent type.

        Args:
            agent_type: The agent type key.
            agent_cls: The agent class to register.
        """
        if not issubclass(agent_cls, BaseAgent):
            raise TypeError(f"{agent_cls} must be a subclass of BaseAgent")
        self._AGENT_TYPES[agent_type] = agent_cls

    def clear_all_history(self) -> None:
        """Clear conversation history for all agents."""
        for agent in self._agents.values():
            agent.clear_history()
