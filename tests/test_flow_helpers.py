"""Tests for pure helper functions in flow.py (no LLM calls)."""

from __future__ import annotations

from autoscrum.flow import _assign_ids, _fallback_stories, _select_by_velocity
from autoscrum.models import UserStory


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
