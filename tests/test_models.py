"""Tests for Pydantic domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoscrum.models import (
    AgentConfig,
    BacklogOutput,
    RetroOutput,
    ReviewOutput,
    ReviewVerdict,
    ScrumState,
    Sprint,
    TeamConfig,
    UserStory,
)


class TestUserStory:
    def test_defaults(self):
        s = UserStory(id="US-001", title="T", description="D")
        assert s.status == "backlog"
        assert s.story_points == 0
        assert s.acceptance_criteria == []
        assert s.deliverables == []

    def test_valid_statuses(self):
        for status in ("backlog", "planned", "in_progress", "in_review", "done", "rejected"):
            s = UserStory(id="X", title="T", description="D", status=status)
            assert s.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            UserStory(id="X", title="T", description="D", status="invalid")

    def test_round_trip_serialization(self):
        s = UserStory(id="US-001", title="T", description="D", story_points=5, status="done")
        rebuilt = UserStory.model_validate(s.model_dump())
        assert rebuilt == s


class TestSprint:
    def test_defaults(self):
        sp = Sprint(number=1)
        assert sp.goal == ""
        assert sp.stories == []
        assert sp.planned_points == 0
        assert sp.completed_points == 0
        assert sp.duration_seconds == 300

    def test_with_stories(self, sample_stories):
        sp = Sprint(number=1, stories=sample_stories[:2], planned_points=7)
        assert len(sp.stories) == 2
        assert sp.planned_points == 7


class TestScrumState:
    def test_defaults(self):
        state = ScrumState()
        assert state.project_goal == ""
        assert state.velocity == 13
        assert state.max_sprints == 3
        assert state.output_dir == "product"
        assert state.enable_code_execution is True

    def test_custom_values(self):
        state = ScrumState(
            project_goal="Calculator",
            velocity=8,
            max_sprints=5,
            enable_code_execution=False,
        )
        assert state.project_goal == "Calculator"
        assert state.velocity == 8
        assert state.enable_code_execution is False


class TestTeamConfig:
    def test_all_defaults(self):
        tc = TeamConfig()
        assert tc.product_owner.llm == "openai/gpt-4o"
        assert tc.developer.count == 1

    def test_partial_override(self):
        tc = TeamConfig(developer=AgentConfig(llm="ollama/llama3", count=2))
        assert tc.developer.llm == "ollama/llama3"
        assert tc.developer.count == 2
        assert tc.product_owner.llm == "openai/gpt-4o"


class TestStructuredOutputs:
    def test_backlog_output(self, sample_stories):
        out = BacklogOutput(stories=sample_stories)
        assert len(out.stories) == 5

    def test_review_output(self):
        out = ReviewOutput(
            verdicts=[
                ReviewVerdict(story_id="US-001", status="done", reason="ok"),
                ReviewVerdict(story_id="US-002", status="rejected", reason="missing tests"),
            ]
        )
        assert len(out.verdicts) == 2
        assert out.verdicts[1].status == "rejected"

    def test_review_verdict_invalid_status(self):
        with pytest.raises(ValidationError):
            ReviewVerdict(story_id="X", status="maybe")

    def test_retro_output_defaults(self):
        out = RetroOutput()
        assert out.went_well == []
        assert out.needs_improvement == []
        assert out.action_items == []
