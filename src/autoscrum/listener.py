"""CrewAI event listener and litellm stream capture for the Rich display."""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm
from crewai.events import BaseEventListener
from crewai.events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

if TYPE_CHECKING:
    from .display import ScrumDisplay


def install_stream_capture(display: ScrumDisplay) -> None:
    """Wrap ``litellm.completion`` and ``litellm.acompletion`` to capture
    streaming chunks.

    CrewAI replaces ``litellm.callbacks`` during crew setup, so the standard
    CustomLogger approach does not work.  Instead we wrap the completion
    functions themselves and intercept the generator/async-generator when
    ``stream=True``.  Both ``delta.content`` and ``delta.reasoning_content``
    are forwarded to the display's rolling buffer.

    The sync path is used by ``Crew.kickoff()`` and the async path by
    ``Crew.akickoff()`` (sprint execution).
    """

    def _extract(chunk):
        try:
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None) or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if reasoning:
                display.append_llm_chunk(reasoning)
            if text:
                display.append_llm_chunk(text)
        except Exception:
            pass

    _orig_sync = litellm.completion

    def _patched_sync(*args, **kwargs):
        result = _orig_sync(*args, **kwargs)
        if not kwargs.get("stream"):
            return result

        def _tap(gen):
            for chunk in gen:
                _extract(chunk)
                yield chunk

        return _tap(result)

    litellm.completion = _patched_sync

    _orig_async = litellm.acompletion

    async def _patched_async(*args, **kwargs):
        result = await _orig_async(*args, **kwargs)
        if not kwargs.get("stream"):
            return result

        async def _atap(agen):
            async for chunk in agen:
                _extract(chunk)
                yield chunk

        return _atap(result)

    litellm.acompletion = _patched_async


class ScrumEventListener(BaseEventListener):
    """Subscribes to CrewAI's event bus and pushes updates to a ScrumDisplay."""

    _display: ScrumDisplay | None = None

    def __init__(self, display: ScrumDisplay | None = None) -> None:
        self._display = display
        super().__init__()

    @classmethod
    def set_display(cls, display: ScrumDisplay) -> None:
        cls._display = display

    def setup_listeners(self, crewai_event_bus) -> None:  # type: ignore[override]
        display = self

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def _on_crew_start(source, event):
            d = display._display
            if d:
                d.log_activity("Crew", f"[cyan]{event.crew_name}[/cyan] kicked off")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def _on_crew_done(source, event):
            d = display._display
            if d:
                d.set_working(None)
                d.log_activity("Crew", f"[green]{event.crew_name}[/green] completed")

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def _on_agent_start(source, event):
            d = display._display
            if d and hasattr(event, "agent"):
                role = getattr(event.agent, "role", "Agent")
                d.set_working(role)
                d.log_activity(role, "started working")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def _on_agent_done(source, event):
            d = display._display
            if d and hasattr(event, "agent"):
                role = getattr(event.agent, "role", "Agent")
                d.set_working(None)
                d.log_activity(role, "[green]finished[/green]")

        @crewai_event_bus.on(TaskStartedEvent)
        def _on_task_start(source, event):
            d = display._display
            if d and hasattr(event, "task"):
                raw = getattr(event.task, "description", "")
                first_line = raw.split("\n", 1)[0][:80]
                d.log_activity("Task", f"started: {first_line}")

        @crewai_event_bus.on(TaskCompletedEvent)
        def _on_task_done(source, event):
            d = display._display
            if d and hasattr(event, "task"):
                raw = getattr(event.task, "description", "")
                first_line = raw.split("\n", 1)[0][:80]
                d.log_activity("Task", f"[green]done[/green]: {first_line}")

        @crewai_event_bus.on(LLMCallStartedEvent)
        def _on_llm_start(source, event):
            d = display._display
            if d:
                d.bump_llm_calls()
                role = getattr(event, "agent_role", None) or "Agent"
                model = getattr(event, "model", None) or ""
                d.begin_llm_call(role, model)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_done(source, event):
            d = display._display
            if d:
                d.finish_llm_call()

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_tool_start(source, event):
            d = display._display
            if d:
                name = getattr(event, "tool_name", "tool")
                if "writer" not in name.lower():
                    d.log_activity("Tool", f"using [bold]{name}[/bold]")

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_done(source, event):
            d = display._display
            if not d:
                return
            name = getattr(event, "tool_name", "tool")
            output = getattr(event, "output", "") or ""
            if "writer" in name.lower() and "successfully" in output.lower():
                d.log_activity("File", f"[green]{output}[/green]")
            else:
                d.log_activity("Tool", f"[green]{name} done[/green]")
