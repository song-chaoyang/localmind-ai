"""
Tests for LocalMind Core - Event Bus
"""

import asyncio

import pytest

from localmind.core.events import EventBus, Event, EventType


class TestEvent:
    """Tests for Event."""

    def test_event_creation(self):
        event = Event(type=EventType.CHAT_MESSAGE, data={"message": "hello"})
        assert event.type == EventType.CHAT_MESSAGE
        assert event.data["message"] == "hello"
        assert event.id  # Auto-generated

    def test_event_with_source(self):
        event = Event(
            type=EventType.MODEL_LOADED,
            data={"model": "llama3"},
            source="engine",
        )
        assert event.source == "engine"


class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.CHAT_MESSAGE, handler)
        await bus.emit(Event(type=EventType.CHAT_MESSAGE, data={"msg": "hi"}))

        assert len(received) == 1
        assert received[0].data["msg"] == "hi"

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = EventBus()
        results = []

        async def handler1(event):
            results.append("handler1")

        async def handler2(event):
            results.append("handler2")

        bus.subscribe(EventType.CHAT_MESSAGE, handler1)
        bus.subscribe(EventType.CHAT_MESSAGE, handler2)
        await bus.emit(Event(type=EventType.CHAT_MESSAGE))

        assert len(results) == 2
        assert "handler1" in results
        assert "handler2" in results

    @pytest.mark.asyncio
    async def test_wildcard_handler(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event.type)

        bus.subscribe_all(handler)
        await bus.emit(Event(type=EventType.CHAT_MESSAGE))
        await bus.emit(Event(type=EventType.MODEL_LOADED))
        await bus.emit(Event(type=EventType.AGENT_STARTED))

        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.CHAT_MESSAGE, handler)
        bus.unsubscribe(EventType.CHAT_MESSAGE, handler)
        await bus.emit(Event(type=EventType.CHAT_MESSAGE))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_stop_propagation(self):
        bus = EventBus()
        received = []

        async def failing_handler(event):
            raise RuntimeError("Intentional error")

        async def good_handler(event):
            received.append("ok")

        bus.subscribe(EventType.CHAT_MESSAGE, failing_handler)
        bus.subscribe(EventType.CHAT_MESSAGE, good_handler)
        await bus.emit(Event(type=EventType.CHAT_MESSAGE))

        assert len(received) == 1

    def test_sync_emit(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event.type)

        bus.subscribe(EventType.CHAT_MESSAGE, handler)
        bus.emit_sync(Event(type=EventType.CHAT_MESSAGE))

        assert len(received) == 1

    def test_event_history(self):
        bus = EventBus()
        bus.emit_sync(Event(type=EventType.CHAT_MESSAGE))
        bus.emit_sync(Event(type=EventType.MODEL_LOADED))
        bus.emit_sync(Event(type=EventType.AGENT_STARTED))

        history = bus.get_history()
        assert len(history) == 3

    def test_event_history_with_filter(self):
        bus = EventBus()
        bus.emit_sync(Event(type=EventType.CHAT_MESSAGE))
        bus.emit_sync(Event(type=EventType.MODEL_LOADED))
        bus.emit_sync(Event(type=EventType.CHAT_MESSAGE))

        chat_events = bus.get_history(event_type=EventType.CHAT_MESSAGE)
        assert len(chat_events) == 2

    def test_clear_history(self):
        bus = EventBus()
        bus.emit_sync(Event(type=EventType.CHAT_MESSAGE))
        bus.clear_history()

        assert len(bus.get_history()) == 0

    def test_shutdown(self):
        bus = EventBus()

        async def handler(event):
            pass

        bus.subscribe(EventType.CHAT_MESSAGE, handler)
        bus.shutdown()

        assert len(bus._handlers) == 0
        assert len(bus._wildcard_handlers) == 0
