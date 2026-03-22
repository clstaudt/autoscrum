"""Tests for ScrumDisplay state management (no live terminal needed)."""

from __future__ import annotations

from autoscrum.display import ScrumDisplay
from autoscrum.models import UserStory


class TestScrumDisplayState:
    """Verify state mutations without starting the live display."""

    def test_set_ceremony(self):
        d = ScrumDisplay()
        d.set_ceremony("Sprint Planning")
        assert d._ceremony == "Sprint Planning"
        assert any("Sprint Planning" in line for line in d._activity)

    def test_set_sprint_info(self):
        d = ScrumDisplay()
        d.set_sprint_info(number=2, max_sprints=5, goal="Build API", velocity=8)
        assert d._sprint_number == 2
        assert d._max_sprints == 5
        assert d._sprint_goal == "Build API"
        assert d._velocity == 8

    def test_sync_stories_tracks_completed_points(self):
        d = ScrumDisplay()
        stories = [
            UserStory(id="A", title="T", description="D", story_points=5, status="done"),
            UserStory(id="B", title="T", description="D", story_points=3, status="in_progress"),
            UserStory(id="C", title="T", description="D", story_points=2, status="done"),
        ]
        d.sync_stories(stories)
        assert len(d._stories) == 3
        assert d._completed_points == 7

    def test_log_activity(self):
        d = ScrumDisplay()
        d.log_activity("Developer", "Writing code")
        assert any("Developer" in line and "Writing code" in line for line in d._activity)

    def test_bump_llm_calls(self):
        d = ScrumDisplay()
        assert d._llm_calls == 0
        d.bump_llm_calls()
        d.bump_llm_calls()
        assert d._llm_calls == 2

    def test_set_working(self):
        d = ScrumDisplay()
        d.set_working("QA Engineer")
        assert d._working_role == "QA Engineer"
        assert d._working_since is not None

        d.set_working(None)
        assert d._working_role is None
        assert d._working_since is None

    def test_activity_capped_at_max(self):
        d = ScrumDisplay()
        for i in range(30):
            d.log_activity("Bot", f"msg {i}")
        from autoscrum.display import MAX_ACTIVITY_LINES
        assert len(d._activity) == MAX_ACTIVITY_LINES

    def test_begin_llm_call(self):
        d = ScrumDisplay()
        d.begin_llm_call("Product Owner", "openai/gpt-4o")
        assert d._llm_role == "Product Owner"
        assert d._llm_model == "openai/gpt-4o"
        assert any("Product Owner" in line for line in d._llm_lines)

    def test_append_llm_chunk_and_finish(self):
        d = ScrumDisplay()
        d.begin_llm_call("Dev", "openai/gpt-4o")
        d.append_llm_chunk("Hello ")
        d.append_llm_chunk("world\nSecond line")
        assert "Hello world" in list(d._llm_lines)
        assert d._llm_buf == "Second line"
        d.finish_llm_call()
        assert d._llm_buf == ""
        assert "Second line" in list(d._llm_lines)

    def test_llm_rolling_window(self):
        d = ScrumDisplay()
        from autoscrum.display import MAX_LLM_OUTPUT_LINES
        for i in range(MAX_LLM_OUTPUT_LINES + 10):
            d.append_llm_chunk(f"line {i}\n")
        assert len(d._llm_lines) == MAX_LLM_OUTPUT_LINES

    def test_llm_output_panel_renders(self):
        d = ScrumDisplay()
        d.begin_llm_call("QA", "openai/model")
        d.append_llm_chunk("All tests pass.")
        d.finish_llm_call()
        panel = d._render_llm_output()
        assert panel.title is not None


class TestStreamCaptureIntegration:
    """Verify that install_stream_capture patches litellm.completion
    and tokens arrive in the display's rolling buffer."""

    def test_streaming_chunks_reach_display(self, monkeypatch):
        """Simulate a streaming litellm.completion and check that
        both content and reasoning tokens end up in the display."""
        import litellm
        from unittest.mock import MagicMock

        d = ScrumDisplay()

        # Build fake streaming chunks
        def _make_chunk(content=None, reasoning=None):
            delta = MagicMock()
            delta.content = content
            delta.reasoning_content = reasoning
            choice = MagicMock()
            choice.delta = delta
            chunk = MagicMock()
            chunk.choices = [choice]
            return chunk

        fake_chunks = [
            _make_chunk(reasoning="Let me think"),
            _make_chunk(reasoning="...\n"),
            _make_chunk(content="Hello "),
            _make_chunk(content="world"),
        ]

        def fake_completion(*args, **kwargs):
            return iter(fake_chunks)

        monkeypatch.setattr(litellm, "completion", fake_completion)

        from autoscrum.listener import install_stream_capture
        install_stream_capture(d)

        # Simulate what CrewAI does: call litellm.completion(stream=True)
        gen = litellm.completion(
            model="openai/test", messages=[], stream=True
        )
        for _ in gen:
            pass

        d.finish_llm_call()
        all_text = "\n".join(d._llm_lines) + d._llm_buf
        assert "Let me think" in all_text
        assert "Hello world" in all_text

    def test_non_streaming_calls_pass_through(self, monkeypatch):
        """Non-streaming calls should not be intercepted."""
        import litellm

        d = ScrumDisplay()
        sentinel = object()

        def fake_completion(*args, **kwargs):
            return sentinel

        monkeypatch.setattr(litellm, "completion", fake_completion)

        from autoscrum.listener import install_stream_capture
        install_stream_capture(d)

        result = litellm.completion(model="openai/test", messages=[])
        assert result is sentinel
        assert len(d._llm_lines) == 0

    def test_async_streaming_chunks_reach_display(self, monkeypatch):
        """Verify acompletion (async) streaming also reaches the display."""
        import asyncio
        import litellm
        from unittest.mock import MagicMock

        d = ScrumDisplay()

        def _make_chunk(content=None, reasoning=None):
            delta = MagicMock()
            delta.content = content
            delta.reasoning_content = reasoning
            choice = MagicMock()
            choice.delta = delta
            chunk = MagicMock()
            chunk.choices = [choice]
            return chunk

        fake_chunks = [
            _make_chunk(reasoning="Thinking…\n"),
            _make_chunk(content="Done"),
        ]

        async def fake_acompletion(*args, **kwargs):
            async def _gen():
                for c in fake_chunks:
                    yield c
            return _gen()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        from autoscrum.listener import install_stream_capture
        install_stream_capture(d)

        async def _run():
            gen = await litellm.acompletion(model="openai/test", messages=[], stream=True)
            async for _ in gen:
                pass

        asyncio.run(_run())
        d.finish_llm_call()
        all_text = "\n".join(d._llm_lines) + d._llm_buf
        assert "Thinking" in all_text
        assert "Done" in all_text
