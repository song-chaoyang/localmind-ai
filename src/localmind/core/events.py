"""
Event Bus system for LocalMind.

Provides a publish-subscribe mechanism for decoupled communication
between components.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be emitted in LocalMind."""

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"

    # Model events
    MODEL_LOADING = "model.loading"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    MODEL_ERROR = "model.error"

    # Chat events
    CHAT_MESSAGE = "chat.message"
    CHAT_RESPONSE = "chat.response"
    CHAT_STREAM_CHUNK = "chat.stream_chunk"
    CHAT_ERROR = "chat.error"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_STARTED = "agent.started"
    AGENT_FINISHED = "agent.finished"
    AGENT_ERROR = "agent.error"
    AGENT_TOOL_CALL = "agent.tool_call"

    # Plugin events
    PLUGIN_LOADED = "plugin.loaded"
    PLUGIN_UNLOADED = "plugin.unloaded"
    PLUGIN_ERROR = "plugin.error"

    # Memory events
    MEMORY_STORE = "memory.store"
    MEMORY_RETRIEVE = "memory.retrieve"
    MEMORY_CLEAR = "memory.clear"

    # RAG events
    RAG_DOCUMENT_INGESTED = "rag.document_ingested"
    RAG_QUERY = "rag.query"
    RAG_ERROR = "rag.error"

    # Workflow events
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step_completed"
    WORKFLOW_FINISHED = "workflow.finished"
    WORKFLOW_ERROR = "workflow.error"


@dataclass
class Event:
    """Represents an event in the system."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())


# Event handler type
EventHandler = Callable[[Event], Awaitable[None]] | Callable[[Event], None]


class EventBus:
    """
    Central event bus for publish-subscribe communication.

    Supports both sync and async event handlers, with filtering
    and wildcard subscriptions.
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: List[EventHandler] = []
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._running: bool = True

    def subscribe(
        self,
        event_type: EventType | str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to a specific event type."""
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler {handler.__name__} subscribed to {event_type.value}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events (wildcard)."""
        self._wildcard_handlers.append(handler)
        logger.debug(f"Handler {handler.__name__} subscribed to all events")

    def unsubscribe(
        self,
        event_type: EventType | str,
        handler: EventHandler,
    ) -> None:
        """Unsubscribe a handler from an event type."""
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribed handlers.

        Handlers are called in order of subscription.
        Exceptions in handlers are logged but don't stop propagation.
        """
        if not self._running:
            return

        # Record event in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Get all handlers for this event type
        handlers = list(self._handlers.get(event.type, []))
        handlers.extend(self._wildcard_handlers)

        # Call all handlers
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(
                    f"Error in event handler {handler.__name__} "
                    f"for event {event.type.value}: {e}"
                )

    def emit_sync(self, event: Event) -> None:
        """Emit an event synchronously (non-async handlers only)."""
        if not self._running:
            return

        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        handlers = list(self._handlers.get(event.type, []))
        handlers.extend(self._wildcard_handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler {handler.__name__} "
                    f"for event {event.type.value}: {e}"
                )

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def shutdown(self) -> None:
        """Shut down the event bus."""
        self._running = False
        self._handlers.clear()
        self._wildcard_handlers.clear()
        self._event_history.clear()
