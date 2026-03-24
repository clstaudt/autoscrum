"""Tests for pure helper functions in flow.py (no LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from litellm.exceptions import APIConnectionError, AuthenticationError

from autoscrum.flow import (
    LLMConnectionError,
    _assign_ids,
    _extract_slug,
    _fallback_stories,
    _generate_project_slug,
    _safe_kickoff,
    _sanitize_slug,
    _select_by_velocity,
)
from autoscrum.models import AgentConfig, TeamConfig, UserStory
from autoscrum.preflight import preflight_check


class TestAssignIds:
    def test_assigns_sequential_ids(self, sample_stories):
        _assign_ids(sample_stories)
        assert [s.id for s in sample_stories] == [
            "US-001", "US-002", "US-003", "US-004", "US-005",
        ]

    def test_resets_status_to_backlog(self):
        stories = [
            UserStory(id="X", title="T", description="D", status="done", story_points=3),
        ]
        _assign_ids(stories)
        assert stories[0].status == "backlog"

    def test_minimum_story_points(self):
        stories = [
            UserStory(id="X", title="T", description="D", story_points=0),
            UserStory(id="Y", title="T", description="D", story_points=1),
        ]
        _assign_ids(stories)
        assert stories[0].story_points == 2
        assert stories[1].story_points == 1

    def test_empty_list(self):
        stories: list[UserStory] = []
        _assign_ids(stories)
        assert stories == []


class TestSelectByVelocity:
    def test_selects_within_budget(self, sample_stories):
        selected = _select_by_velocity(sample_stories, velocity=10)
        total = sum(s.story_points for s in selected)
        assert total <= 10
        assert len(selected) >= 1

    def test_greedy_front_order(self):
        stories = [
            UserStory(id="A", title="A", description="D", story_points=3),
            UserStory(id="B", title="B", description="D", story_points=3),
            UserStory(id="C", title="C", description="D", story_points=3),
        ]
        selected = _select_by_velocity(stories, velocity=7)
        assert [s.id for s in selected] == ["A", "B"]

    def test_exact_fit(self):
        stories = [
            UserStory(id="A", title="A", description="D", story_points=5),
            UserStory(id="B", title="B", description="D", story_points=5),
        ]
        selected = _select_by_velocity(stories, velocity=10)
        assert len(selected) == 2

    def test_nothing_fits(self):
        stories = [
            UserStory(id="A", title="A", description="D", story_points=10),
        ]
        selected = _select_by_velocity(stories, velocity=5)
        assert selected == []

    def test_empty_backlog(self):
        assert _select_by_velocity([], velocity=13) == []

    def test_zero_velocity(self, sample_stories):
        assert _select_by_velocity(sample_stories, velocity=0) == []

    def test_skips_oversized_picks_smaller(self):
        stories = [
            UserStory(id="A", title="Big", description="D", story_points=10),
            UserStory(id="B", title="Small", description="D", story_points=3),
        ]
        selected = _select_by_velocity(stories, velocity=5)
        assert [s.id for s in selected] == ["B"]


class TestSafeKickoff:
    def test_auth_error_raises_llm_connection_error(self):
        crew = MagicMock()
        crew.kickoff.side_effect = AuthenticationError(
            "bad key", llm_provider="openai", model="openai/gpt-4o"
        )
        with pytest.raises(LLMConnectionError):
            _safe_kickoff(crew)

    def test_connection_error_raises_llm_connection_error(self):
        crew = MagicMock()
        crew.kickoff.side_effect = APIConnectionError(
            "unreachable", llm_provider="openai", model="openai/gpt-4o"
        )
        with pytest.raises(LLMConnectionError):
            _safe_kickoff(crew)

    def test_wrapped_auth_error_raises_llm_connection_error(self):
        """CrewAI wraps litellm errors in retry envelopes — detect by message."""
        crew = MagicMock()
        crew.kickoff.side_effect = RuntimeError(
            "<failed_attempts>\n"
            "litellm.AuthenticationError: Incorrect API key provided: test\n"
            "</failed_attempts>"
        )
        with pytest.raises(LLMConnectionError):
            _safe_kickoff(crew)

    def test_wrapped_connection_error_raises_llm_connection_error(self):
        crew = MagicMock()
        crew.kickoff.side_effect = RuntimeError(
            "litellm.InternalServerError: Connection error."
        )
        with pytest.raises(LLMConnectionError):
            _safe_kickoff(crew)

    def test_other_errors_return_none(self):
        crew = MagicMock()
        crew.kickoff.side_effect = ValueError("parse error")
        assert _safe_kickoff(crew) is None

    def test_success_returns_result(self):
        crew = MagicMock()
        crew.kickoff.return_value = "result"
        assert _safe_kickoff(crew) == "result"


class TestPreflightCheck:
    def test_auth_error_raises_llm_connection_error(self, monkeypatch):
        team = TeamConfig(
            product_owner=AgentConfig(llm="openai/gpt-4o"),
            scrum_master=AgentConfig(llm="openai/gpt-4o"),
            developer=AgentConfig(llm="openai/gpt-4o"),
            qa_engineer=AgentConfig(llm="openai/gpt-4o"),
        )
        monkeypatch.setattr(
            "autoscrum.preflight.completion",
            MagicMock(side_effect=AuthenticationError(
                "bad key", llm_provider="openai", model="openai/gpt-4o"
            )),
        )
        with pytest.raises(LLMConnectionError, match="unreachable"):
            preflight_check(team)

    def test_success_passes(self, monkeypatch):
        team = TeamConfig()
        monkeypatch.setattr(
            "autoscrum.preflight.completion",
            MagicMock(return_value="ok"),
        )
        preflight_check(team)

    def test_deduplicates_identical_configs(self, monkeypatch):
        team = TeamConfig()
        mock_completion = MagicMock(return_value="ok")
        monkeypatch.setattr("autoscrum.preflight.completion", mock_completion)
        preflight_check(team)
        assert mock_completion.call_count == 1

    def test_non_fatal_error_passes(self, monkeypatch):
        team = TeamConfig()
        monkeypatch.setattr(
            "autoscrum.preflight.completion",
            MagicMock(side_effect=ValueError("parse error")),
        )
        preflight_check(team)


class TestFallbackStories:
    def test_returns_three_stories(self):
        stories = _fallback_stories("Build a calculator")
        assert len(stories) == 3

    def test_goal_embedded_in_descriptions(self):
        goal = "Todo app"
        stories = _fallback_stories(goal)
        assert any(goal in s.description for s in stories)

    def test_all_have_ids_and_points(self):
        stories = _fallback_stories("X")
        for s in stories:
            assert s.id.startswith("US-")
            assert s.story_points > 0

    def test_total_points(self):
        stories = _fallback_stories("X")
        assert sum(s.story_points for s in stories) == 10


class TestSanitizeSlug:
    def test_basic(self):
        assert _sanitize_slug("flet-calculator") == "flet-calculator"

    def test_strips_quotes_and_backticks(self):
        assert _sanitize_slug('`"my-project"` ') == "my-project"

    def test_replaces_spaces_and_special_chars(self):
        assert _sanitize_slug("My Cool App!") == "my-cool-app"

    def test_truncates_to_60(self):
        assert len(_sanitize_slug("a" * 100)) == 60

    def test_empty_returns_project(self):
        assert _sanitize_slug("!!!") == "project"


class TestExtractSlug:
    def test_clean_slug(self):
        assert _extract_slug("flet-calculator") == "flet-calculator"

    def test_slug_from_verbose_reasoning(self):
        text = (
            "Thinking Process:\n"
            "1. Analyze the request\n"
            "2. Consider keywords\n\n"
            "flet-calculator-gui"
        )
        assert _extract_slug(text) == "flet-calculator-gui"

    def test_slug_from_backtick_wrapped(self):
        assert _extract_slug("`simple-calc-app`") == "simple-calc-app"

    def test_returns_none_for_garbage(self):
        assert _extract_slug("!!!") is None

    def test_requires_hyphen(self):
        assert _extract_slug("singleword") is None


class TestGenerateProjectSlug:
    def test_uses_llm_response(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "flet-calc-gui"
        monkeypatch.setattr(
            "autoscrum.flow.litellm_completion", MagicMock(return_value=mock_resp)
        )
        stories = [
            UserStory(id="US-001", title="Calculator UI", description="Build the UI"),
        ]
        slug = _generate_project_slug("Build a calculator", stories, AgentConfig())
        assert slug == "flet-calc-gui"

    def test_falls_back_on_llm_failure(self, monkeypatch):
        monkeypatch.setattr(
            "autoscrum.flow.litellm_completion",
            MagicMock(side_effect=RuntimeError("nope")),
        )
        stories = [
            UserStory(id="US-001", title="X", description="Y"),
        ]
        slug = _generate_project_slug("Build a calculator", stories, AgentConfig())
        assert slug == "build-a-calculator"
