"""Shared fixtures for the AutoScrum test suite."""

from __future__ import annotations

import pytest

from autoscrum.models import AgentConfig, Sprint, TeamConfig, UserStory


@pytest.fixture()
def sample_stories() -> list[UserStory]:
    """A handful of stories with varied statuses and point values."""
    return [
        UserStory(id="US-001", title="Setup", description="Init project", story_points=2, status="backlog"),
        UserStory(id="US-002", title="Auth", description="Login flow", story_points=5, status="backlog"),
        UserStory(id="US-003", title="Dashboard", description="Main view", story_points=8, status="backlog"),
        UserStory(id="US-004", title="Settings", description="User prefs", story_points=3, status="backlog"),
        UserStory(id="US-005", title="Polish", description="Edge cases", story_points=1, status="done"),
    ]


@pytest.fixture()
def team_config() -> TeamConfig:
    return TeamConfig(
        product_owner=AgentConfig(llm="fake/model"),
        scrum_master=AgentConfig(llm="fake/model"),
        developer=AgentConfig(llm="fake/model"),
        qa_engineer=AgentConfig(llm="fake/model"),
    )


@pytest.fixture()
def completed_sprints() -> list[Sprint]:
    return [
        Sprint(number=1, completed_points=10),
        Sprint(number=2, completed_points=8),
        Sprint(number=3, completed_points=12),
    ]
