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
