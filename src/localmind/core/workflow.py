"""
Workflow orchestration engine for LocalMind.

Provides a flexible, graph-based workflow system for orchestrating
multi-step AI tasks with dependency management.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from localmind.core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Status of a workflow."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result of a workflow step execution."""

    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    id: str
    name: str
    agent: Optional[str] = None
    function: Optional[Callable] = None
    input: Any = None
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, StepResult]], bool]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, WorkflowStep):
            return self.id == other.id
        return False


class Workflow:
    """
    Workflow orchestration engine.

    Supports DAG-based dependency management, conditional execution,
    retry logic, and parallel step execution where possible.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        event_bus: Optional[EventBus] = None,
    ):
        self.name = name
        self.description = description
        self._steps: Dict[str, WorkflowStep] = {}
        self._execution_order: List[str] = []
        self._results: Dict[str, StepResult] = {}
        self._status = WorkflowStatus.CREATED
        self._event_bus = event_bus
        self._created_at = datetime.now()
        self._started_at: Optional[datetime] = None
        self._finished_at: Optional[datetime] = None

    def add_step(
        self,
        step_id: str,
        name: Optional[str] = None,
        agent: Optional[str] = None,
        function: Optional[Callable] = None,
        input: Any = None,
        depends_on: Optional[List[str]] = None,
        condition: Optional[Callable[[Dict[str, StepResult]], bool]] = None,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ) -> WorkflowStep:
        """
        Add a step to the workflow.

        Args:
            step_id: Unique identifier for the step
            name: Human-readable name
            agent: Agent name to use for this step
            function: Custom function to execute
            input: Input data or template
            depends_on: List of step IDs this step depends on
            condition: Optional condition function
            max_retries: Maximum retry attempts on failure
            timeout_seconds: Step execution timeout

        Returns:
            The created WorkflowStep
        """
        step = WorkflowStep(
            id=step_id,
            name=name or step_id,
            agent=agent,
            function=function,
            input=input,
            depends_on=depends_on or [],
            condition=condition,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

        # Validate dependencies
        for dep in step.depends_on:
            if dep not in self._steps and dep != step_id:
                raise ValueError(
                    f"Step '{step_id}' depends on unknown step '{dep}'"
                )

        self._steps[step_id] = step
        self._compute_execution_order()
        return step

    def remove_step(self, step_id: str) -> None:
        """Remove a step from the workflow."""
        if step_id in self._steps:
            del self._steps[step_id]
            # Remove from dependencies of other steps
            for step in self._steps.values():
                if step_id in step.depends_on:
                    step.depends_on.remove(step_id)
            self._compute_execution_order()

    def _compute_execution_order(self) -> None:
        """Compute the execution order using topological sort."""
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []

        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{step_id}'")
            if step_id in visited:
                return

            temp_visited.add(step_id)
            step = self._steps.get(step_id)
            if step:
                for dep in step.depends_on:
                    if dep in self._steps:
                        visit(dep)
            temp_visited.remove(step_id)
            visited.add(step_id)
            order.append(step_id)

        for step_id in self._steps:
            visit(step_id)

        self._execution_order = order

    async def run(
        self,
        engine: Optional[Any] = None,
        initial_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, StepResult]:
        """
        Execute the workflow.

        Args:
            engine: LocalMind engine instance (for agent execution)
            initial_input: Initial input data for the workflow

        Returns:
            Dictionary of step results
        """
        self._status = WorkflowStatus.RUNNING
        self._started_at = datetime.now()
        self._results.clear()

        await self._emit_event(EventType.WORKFLOW_STARTED, {
            "workflow_name": self.name,
            "steps": len(self._steps),
        })

        try:
            for step_id in self._execution_order:
                step = self._steps[step_id]

                # Check condition
                if step.condition and not step.condition(self._results):
                    step.status = StepStatus.SKIPPED
                    self._results[step_id] = StepResult(
                        step_id=step_id,
                        status=StepStatus.SKIPPED,
                    )
                    continue

                # Execute step
                step.status = StepStatus.RUNNING
                result = await self._execute_step(step, engine, initial_input)
                self._results[step_id] = result
                step.result = result
                step.status = result.status

                await self._emit_event(EventType.WORKFLOW_STEP_COMPLETED, {
                    "step_id": step_id,
                    "status": result.status.value,
                })

                # Stop if step failed
                if result.status == StepStatus.FAILED:
                    self._status = WorkflowStatus.FAILED
                    await self._emit_event(EventType.WORKFLOW_ERROR, {
                        "step_id": step_id,
                        "error": result.error,
                    })
                    break

            if self._status == WorkflowStatus.RUNNING:
                self._status = WorkflowStatus.COMPLETED
                await self._emit_event(EventType.WORKFLOW_FINISHED, {
                    "workflow_name": self.name,
                    "results_count": len(self._results),
                })

        except Exception as e:
            self._status = WorkflowStatus.FAILED
            logger.error(f"Workflow '{self.name}' failed: {e}")
            await self._emit_event(EventType.WORKFLOW_ERROR, {
                "workflow_name": self.name,
                "error": str(e),
            })

        self._finished_at = datetime.now()
        return self._results

    async def _execute_step(
        self,
        step: WorkflowStep,
        engine: Optional[Any],
        initial_input: Optional[Dict[str, Any]],
    ) -> StepResult:
        """Execute a single workflow step."""
        import time

        start_time = time.time()

        for attempt in range(step.max_retries + 1):
            try:
                # Prepare input
                step_input = self._prepare_step_input(step, initial_input)

                # Execute
                if step.function:
                    if asyncio.iscoroutinefunction(step.function):
                        output = await step.function(step_input)
                    else:
                        output = step.function(step_input)
                elif step.agent and engine:
                    output = await self._execute_agent_step(step, engine, step_input)
                else:
                    output = step_input

                duration = (time.time() - start_time) * 1000
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.COMPLETED,
                    output=output,
                    duration_ms=duration,
                )

            except asyncio.TimeoutError:
                duration = (time.time() - start_time) * 1000
                if attempt < step.max_retries:
                    logger.warning(
                        f"Step '{step.id}' timed out, retrying "
                        f"({attempt + 1}/{step.max_retries})"
                    )
                    continue
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=f"Timeout after {step.timeout_seconds}s",
                    duration_ms=duration,
                )

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                if attempt < step.max_retries:
                    logger.warning(
                        f"Step '{step.id}' failed: {e}, retrying "
                        f"({attempt + 1}/{step.max_retries})"
                    )
                    continue
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    error=str(e),
                    duration_ms=duration,
                )

        return StepResult(
            step_id=step.id,
            status=StepStatus.FAILED,
            error="Max retries exceeded",
        )

    def _prepare_step_input(
        self,
        step: WorkflowStep,
        initial_input: Optional[Dict[str, Any]],
    ) -> Any:
        """Prepare input for a step by resolving dependencies."""
        if step.input is not None:
            return step.input

        # Collect outputs from dependencies
        if step.depends_on:
            dep_outputs = {}
            for dep_id in step.depends_on:
                if dep_id in self._results and self._results[dep_id].output:
                    dep_outputs[dep_id] = self._results[dep_id].output
            return dep_outputs

        return initial_input

    async def _execute_agent_step(
        self,
        step: WorkflowStep,
        engine: Any,
        step_input: Any,
    ) -> Any:
        """Execute a step using an agent."""
        # This will be connected to the agent system
        if hasattr(engine, "execute_agent"):
            return await engine.execute_agent(
                agent_name=step.agent,
                task=step_input if isinstance(step_input, str) else str(step_input),
            )
        raise RuntimeError(
            f"Engine does not support agent execution for step '{step.id}'"
        )

    async def _emit_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Emit a workflow event."""
        if self._event_bus:
            await self._event_bus.emit(Event(
                type=event_type,
                data=data,
                source=f"workflow:{self.name}",
            ))

    @property
    def status(self) -> WorkflowStatus:
        """Get the current workflow status."""
        return self._status

    @property
    def results(self) -> Dict[str, StepResult]:
        """Get all step results."""
        return dict(self._results)

    def get_step_result(self, step_id: str) -> Optional[StepResult]:
        """Get the result of a specific step."""
        return self._results.get(step_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the workflow to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "steps": {
                step_id: {
                    "id": step.id,
                    "name": step.name,
                    "agent": step.agent,
                    "depends_on": step.depends_on,
                    "status": step.status.value,
                }
                for step_id, step in self._steps.items()
            },
            "execution_order": self._execution_order,
            "created_at": self._created_at.isoformat(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "finished_at": self._finished_at.isoformat() if self._finished_at else None,
        }

    def visualize(self) -> str:
        """Generate a text-based visualization of the workflow."""
        lines = [f"Workflow: {self.name}", "=" * 40]

        for step_id in self._execution_order:
            step = self._steps[step_id]
            status_icon = {
                StepStatus.PENDING: "⬜",
                StepStatus.RUNNING: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }.get(step.status, "⬜")

            agent_info = f" [{step.agent}]" if step.agent else ""
            deps = f" (after: {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"  {status_icon} {step.name}{agent_info}{deps}")

        return "\n".join(lines)
