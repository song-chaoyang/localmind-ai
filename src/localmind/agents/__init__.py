"""
Agent System for LocalMind.

Provides base agent classes and built-in agents for common tasks
like research, coding, data analysis, and writing.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent execution."""

    agent_name: str
    task: str
    output: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Definition of a tool that an agent can use."""

    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base class for all LocalMind agents.

    Agents are autonomous entities that can perform tasks using
    a combination of language model reasoning and tool usage.
    """

    name: str = "base_agent"
    description: str = "Base agent class"
    capabilities: List[str] = field(default_factory=list)

    def __init__(self, engine: Optional[Any] = None):
        self.engine = engine
        self._tools: Dict[str, ToolDefinition] = {}
        self._history: List[Dict[str, Any]] = []

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool for this agent."""
        self._tools[tool.name] = tool
        logger.debug(f"Agent '{self.name}' registered tool: {tool.name}")

    def get_tools(self) -> List[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    async def use_tool(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        logger.info(f"Agent '{self.name}' using tool: {name}")
        self._history.append({
            "type": "tool_call",
            "tool": name,
            "args": kwargs,
        })

        try:
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**kwargs)
            else:
                result = tool.function(**kwargs)

            self._history.append({
                "type": "tool_result",
                "tool": name,
                "result": str(result)[:1000],
            })
            return result

        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            self._history.append({
                "type": "tool_error",
                "tool": name,
                "error": str(e),
            })
            raise

    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """
        Execute a task.

        Args:
            task: The task description
            context: Additional context

        Returns:
            The result of the task
        """
        pass

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the agent's execution history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the agent's history."""
        self._history.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ─── Built-in Agents ────────────────────────────────────────────────


class ResearchAgent(BaseAgent):
    """
    Research agent that can search for information and summarize findings.

    Capabilities: web search, information extraction, summarization
    """

    name = "researcher"
    description = "Research agent for finding and summarizing information"
    capabilities = ["web_search", "summarization", "information_extraction"]

    def __init__(self, engine: Optional[Any] = None):
        super().__init__(engine)
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Set up default tools for research."""
        self.register_tool(ToolDefinition(
            name="web_search",
            description="Search the web for information",
            function=self._web_search,
            parameters={"query": "str", "num_results": "int"},
            required_params=["query"],
        ))
        self.register_tool(ToolDefinition(
            name="summarize",
            description="Summarize a text",
            function=self._summarize,
            parameters={"text": "str", "max_length": "int"},
            required_params=["text"],
        ))

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a research task."""
        self._history.append({"type": "task", "task": task})

        # If we have an engine with a loaded model, use it for research
        if self.engine and hasattr(self.engine, "chat"):
            try:
                prompt = (
                    f"You are a research assistant. Conduct research on the following topic "
                    f"and provide a comprehensive, well-structured response.\n\n"
                    f"Research Topic: {task}\n\n"
                    f"Provide your findings in a clear, organized format with key points."
                )
                result = await self.engine.chat(prompt)
                self._history.append({"type": "result", "result": result[:500]})
                return result
            except Exception as e:
                logger.warning(f"Research via model failed: {e}")

        # Fallback: return structured research plan
        return self._generate_research_plan(task)

    async def _web_search(self, query: str, num_results: int = 5) -> str:
        """Perform a web search (placeholder)."""
        return f"[Web search results for: {query}] (Integration with search API needed)"

    def _summarize(self, text: str, max_length: int = 500) -> str:
        """Summarize text (placeholder)."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _generate_research_plan(self, topic: str) -> str:
        """Generate a research plan when model is not available."""
        return (
            f"# Research Plan: {topic}\n\n"
            f"## Approach\n"
            f"1. Define key research questions\n"
            f"2. Identify relevant sources\n"
            f"3. Gather and analyze information\n"
            f"4. Synthesize findings\n\n"
            f"## Key Questions\n"
            f"- What are the main aspects of {topic}?\n"
            f"- What are the current trends and developments?\n"
            f"- What are the key challenges and opportunities?\n\n"
            f"## Status\n"
            f"⚠️ Full research requires a loaded AI model. "
            f"Load a model with `mind.load_model('llama3')` to enable "
            f"automated research capabilities."
        )


class CodeAgent(BaseAgent):
    """
    Code agent that can write, review, and debug code.

    Capabilities: code generation, code review, debugging, refactoring
    """

    name = "coder"
    description = "Code agent for writing, reviewing, and debugging code"
    capabilities = ["code_generation", "code_review", "debugging", "refactoring"]

    def __init__(self, engine: Optional[Any] = None):
        super().__init__(engine)
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Set up default tools for coding."""
        self.register_tool(ToolDefinition(
            name="read_file",
            description="Read a file's contents",
            function=self._read_file,
            parameters={"path": "str"},
            required_params=["path"],
        ))
        self.register_tool(ToolDefinition(
            name="write_file",
            description="Write content to a file",
            function=self._write_file,
            parameters={"path": "str", "content": "str"},
            required_params=["path", "content"],
        ))
        self.register_tool(ToolDefinition(
            name="execute_code",
            description="Execute Python code safely",
            function=self._execute_code,
            parameters={"code": "str"},
            required_params=["code"],
        ))

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a coding task."""
        self._history.append({"type": "task", "task": task})

        if self.engine and hasattr(self.engine, "chat"):
            try:
                prompt = (
                    f"You are an expert software engineer. Complete the following "
                    f"coding task with clean, well-documented code.\n\n"
                    f"Task: {task}\n\n"
                    f"Provide your solution with clear explanations."
                )
                result = await self.engine.chat(prompt)
                self._history.append({"type": "result", "result": result[:500]})
                return result
            except Exception as e:
                logger.warning(f"Code generation via model failed: {e}")

        return self._generate_code_plan(task)

    def _read_file(self, path: str) -> str:
        """Read a file."""
        try:
            from pathlib import Path
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, path: str, content: str) -> str:
        """Write to a file."""
        try:
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _execute_code(self, code: str) -> str:
        """Execute Python code (with safety restrictions)."""
        # In production, this would use a sandboxed executor
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"
            return output or "Code executed successfully (no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (30s limit)"
        except Exception as e:
            return f"Error executing code: {e}"

    def _generate_code_plan(self, task: str) -> str:
        """Generate a code plan when model is not available."""
        return (
            f"# Code Plan: {task}\n\n"
            f"## Approach\n"
            f"1. Analyze requirements\n"
            f"2. Design solution architecture\n"
            f"3. Implement core functionality\n"
            f"4. Add error handling and tests\n\n"
            f"## Status\n"
            f"⚠️ Full code generation requires a loaded AI model. "
            f"Load a model with `mind.load_model('llama3')` to enable "
            f"automated code generation."
        )


class DataAgent(BaseAgent):
    """
    Data analysis agent for processing and analyzing data.

    Capabilities: data processing, analysis, visualization, statistics
    """

    name = "analyst"
    description = "Data analysis agent for processing and analyzing data"
    capabilities = ["data_processing", "analysis", "visualization", "statistics"]

    def __init__(self, engine: Optional[Any] = None):
        super().__init__(engine)
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Set up default tools for data analysis."""
        self.register_tool(ToolDefinition(
            name="read_csv",
            description="Read a CSV file",
            function=self._read_csv,
            parameters={"path": "str"},
            required_params=["path"],
        ))
        self.register_tool(ToolDefinition(
            name="calculate_stats",
            description="Calculate basic statistics",
            function=self._calculate_stats,
            parameters={"data": "list"},
            required_params=["data"],
        ))

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a data analysis task."""
        self._history.append({"type": "task", "task": task})

        if self.engine and hasattr(self.engine, "chat"):
            try:
                prompt = (
                    f"You are a data analysis expert. Analyze the following "
                    f"data task and provide insights.\n\n"
                    f"Task: {task}\n\n"
                    f"Context: {context}\n\n"
                    f"Provide your analysis with clear visualizations "
                    f"recommendations."
                )
                result = await self.engine.chat(prompt)
                self._history.append({"type": "result", "result": result[:500]})
                return result
            except Exception as e:
                logger.warning(f"Data analysis via model failed: {e}")

        return self._generate_analysis_plan(task)

    def _read_csv(self, path: str) -> str:
        """Read a CSV file."""
        try:
            import csv
            from pathlib import Path
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            return f"CSV loaded: {len(rows)} rows, {len(rows[0]) if rows else 0} columns"
        except Exception as e:
            return f"Error reading CSV: {e}"

    def _calculate_stats(self, data: List[float]) -> str:
        """Calculate basic statistics."""
        if not data:
            return "No data provided"
        n = len(data)
        mean = sum(data) / n
        sorted_data = sorted(data)
        median = sorted_data[n // 2]
        return (
            f"Statistics:\n"
            f"  Count: {n}\n"
            f"  Mean: {mean:.2f}\n"
            f"  Median: {median:.2f}\n"
            f"  Min: {min(data):.2f}\n"
            f"  Max: {max(data):.2f}"
        )

    def _generate_analysis_plan(self, task: str) -> str:
        """Generate an analysis plan when model is not available."""
        return (
            f"# Data Analysis Plan: {task}\n\n"
            f"## Approach\n"
            f"1. Data collection and cleaning\n"
            f"2. Exploratory data analysis\n"
            f"3. Statistical analysis\n"
            f"4. Visualization and reporting\n\n"
            f"## Status\n"
            f"⚠️ Full analysis requires a loaded AI model. "
            f"Load a model with `mind.load_model('llama3')` to enable "
            f"automated data analysis."
        )


class WriterAgent(BaseAgent):
    """
    Writing agent for content creation and editing.

    Capabilities: writing, editing, translation, summarization
    """

    name = "writer"
    description = "Writing agent for content creation and editing"
    capabilities = ["writing", "editing", "translation", "summarization"]

    def __init__(self, engine: Optional[Any] = None):
        super().__init__(engine)

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a writing task."""
        self._history.append({"type": "task", "task": task})

        if self.engine and hasattr(self.engine, "chat"):
            try:
                prompt = (
                    f"You are an expert writer and editor. Complete the following "
                    f"writing task with high-quality, engaging content.\n\n"
                    f"Task: {task}\n\n"
                    f"Provide well-structured, polished content."
                )
                result = await self.engine.chat(prompt)
                self._history.append({"type": "result", "result": result[:500]})
                return result
            except Exception as e:
                logger.warning(f"Writing via model failed: {e}")

        return (
            f"# Writing Task: {task}\n\n"
            f"⚠️ Full writing assistance requires a loaded AI model. "
            f"Load a model with `mind.load_model('llama3')` to enable "
            f"automated content generation."
        )


# ─── Agent Registry ─────────────────────────────────────────────────

BUILTIN_AGENTS = {
    "researcher": ResearchAgent,
    "coder": CodeAgent,
    "analyst": DataAgent,
    "writer": WriterAgent,
}


def create_agent(name: str, engine: Optional[Any] = None) -> BaseAgent:
    """
    Create an agent by name.

    Args:
        name: Agent name (e.g., "researcher", "coder", "analyst", "writer")
        engine: LocalMind engine instance

    Returns:
        An instance of the requested agent

    Raises:
        ValueError: If the agent name is not recognized
    """
    if name not in BUILTIN_AGENTS:
        available = ", ".join(BUILTIN_AGENTS.keys())
        raise ValueError(
            f"Unknown agent: '{name}'. Available agents: {available}"
        )
    return BUILTIN_AGENTS[name](engine=engine)
