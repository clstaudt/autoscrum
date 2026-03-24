"""Tests for the sprint diary memory system."""

from __future__ import annotations

import pytest

from autoscrum.models import (
    RetroOutput,
    ScrumState,
    Sprint,
    TeamConfig,
    UserStory,
)


def _story(id: str, title: str = "T", status: str = "done", rejection_reason: str = "") -> UserStory:
    return UserStory(
        id=id, title=title, description="D",
        story_points=3, status=status, rejection_reason=rejection_reason,
    )


class TestScrumStateDiary:
    def test_diary_defaults_empty(self):
        state = ScrumState()
        assert state.diary == ""

    def test_rejection_reason_defaults_empty(self):
        s = UserStory(id="X", title="T", description="D")
        assert s.rejection_reason == ""

    def test_rejection_reason_round_trips(self):
        s = UserStory(id="X", title="T", description="D", rejection_reason="No files written")
        rebuilt = UserStory.model_validate(s.model_dump())
        assert rebuilt.rejection_reason == "No files written"


class _FakeFlow:
    """Lightweight stand-in for ScrumFlow — avoids heavy CrewAI Memory init."""

    def __init__(self):
        self.state = ScrumState()

    _append_review_to_diary = None  # patched below
    _append_retro_to_diary = None


# Attach the real methods so we test production logic without the Flow base class.
from autoscrum.flow import ScrumFlow as _SF

_FakeFlow._append_review_to_diary = _SF._append_review_to_diary
_FakeFlow._append_retro_to_diary = _SF._append_retro_to_diary


class TestDiaryHelpers:
    """Test the _append_review_to_diary and _append_retro_to_diary helpers."""

    def test_review_diary_records_accepted_and_rejected(self):
        flow = _FakeFlow()
        sprint = Sprint(
            number=1,
            stories=[
                _story("US-001", "Login", status="done"),
                _story("US-002", "Signup", status="backlog", rejection_reason="No files"),
            ],
        )

        flow._append_review_to_diary(sprint)

        assert "Sprint 1" in flow.state.diary
        assert "US-001 (Login): accepted" in flow.state.diary
        assert "US-002 (Signup): rejected — No files" in flow.state.diary

    def test_retro_diary_records_findings(self):
        flow = _FakeFlow()
        sprint = Sprint(number=1)
        retro = RetroOutput(
            went_well=["Good communication"],
            needs_improvement=["Tool usage"],
            action_items=["Use file writer tool", "Test before submit"],
        )

        flow._append_retro_to_diary(sprint, retro)

        assert "Sprint 1" in flow.state.diary
        assert "Good communication" in flow.state.diary
        assert "Tool usage" in flow.state.diary
        assert "Use file writer tool" in flow.state.diary

    def test_diary_accumulates_across_sprints(self):
        flow = _FakeFlow()

        sprint1 = Sprint(number=1, stories=[_story("US-001", status="done")])
        flow._append_review_to_diary(sprint1)
        retro1 = RetroOutput(action_items=["Improve testing"])
        flow._append_retro_to_diary(sprint1, retro1)

        sprint2 = Sprint(number=2, stories=[_story("US-002", status="backlog", rejection_reason="Bug")])
        flow._append_review_to_diary(sprint2)

        assert "Sprint 1 — Review" in flow.state.diary
        assert "Sprint 1 — Retrospective" in flow.state.diary
        assert "Sprint 2 — Review" in flow.state.diary
        assert "Improve testing" in flow.state.diary
        assert "rejected — Bug" in flow.state.diary


@pytest.fixture()
def _patch_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")


class TestCrewDiaryInjection:
    """Verify that diary contents appear in crew task descriptions."""

    SAMPLE_DIARY = (
        "## Sprint 1 — Review\n"
        "- US-001 (Login): rejected — No files\n"
        "## Sprint 1 — Retrospective\n"
        "Action items: Use file writer tool\n"
    )

    @pytest.mark.usefixtures("_patch_llm")
    def test_planning_crew_includes_diary(self):
        from autoscrum.crews.planning import build_planning_crew

        story = _story("US-001", status="backlog", rejection_reason="No files")
        crew = build_planning_crew(
            backlog=[story],
            velocity=10,
            sprint_number=2,
            team=TeamConfig(),
            diary=self.SAMPLE_DIARY,
        )
        desc = crew.tasks[0].description
        assert "Sprint diary" in desc
        assert "No files" in desc

    @pytest.mark.usefixtures("_patch_llm")
    def test_planning_crew_omits_diary_when_empty(self):
        from autoscrum.crews.planning import build_planning_crew

        crew = build_planning_crew(
            backlog=[_story("US-001", status="backlog")],
            velocity=10,
            sprint_number=1,
            team=TeamConfig(),
        )
        desc = crew.tasks[0].description
        assert "Sprint diary" not in desc

    @pytest.mark.usefixtures("_patch_llm")
    def test_execution_crew_includes_diary(self):
        from autoscrum.crews.execution import build_execution_crew

        crew = build_execution_crew(
            stories=[_story("US-001")],
            sprint_number=2,
            team=TeamConfig(),
            output_dir="/tmp/test",
            enable_code_execution=False,
            diary=self.SAMPLE_DIARY,
        )
        desc = crew.tasks[0].description
        assert "Sprint diary" in desc
        assert "Use file writer tool" in desc

    @pytest.mark.usefixtures("_patch_llm")
    def test_execution_crew_includes_rejection_reason(self):
        from autoscrum.crews.execution import build_execution_crew

        story = _story("US-001", status="planned", rejection_reason="Code not written")
        crew = build_execution_crew(
            stories=[story],
            sprint_number=2,
            team=TeamConfig(),
            output_dir="/tmp/test",
            enable_code_execution=False,
        )
        desc = crew.tasks[0].description
        assert "Code not written" in desc

    @pytest.mark.usefixtures("_patch_llm")
    def test_review_crew_includes_diary(self):
        from autoscrum.crews.review import build_review_crew

        crew = build_review_crew(
            stories=[_story("US-001")],
            sprint_number=2,
            team=TeamConfig(),
            output_dir="/tmp/test",
            enable_code_execution=False,
            diary=self.SAMPLE_DIARY,
        )
        desc = crew.tasks[0].description
        assert "Sprint diary" in desc

    @pytest.mark.usefixtures("_patch_llm")
    def test_retrospective_crew_includes_diary(self):
        from autoscrum.crews.retrospective import build_retrospective_crew

        crew = build_retrospective_crew(
            sprint_number=2,
            planned_points=10,
            completed_points=5,
            team=TeamConfig(),
            diary=self.SAMPLE_DIARY,
        )
        desc = crew.tasks[0].description
        assert "Sprint diary" in desc
        assert "Use file writer tool" in desc
