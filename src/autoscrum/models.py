"""Pydantic models for AutoScrum state, configuration, and domain objects."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """LLM and behavioural configuration for a single Scrum role."""

    llm: str = "openai/gpt-4o"
    base_url: Optional[str] = None
    backstory: Optional[str] = None
    count: int = 1


class TeamConfig(BaseModel):
    """Per-role agent configuration loaded from team.yaml."""

    product_owner: AgentConfig = Field(default_factory=AgentConfig)
    scrum_master: AgentConfig = Field(default_factory=AgentConfig)
    developer: AgentConfig = Field(default_factory=AgentConfig)
    qa_engineer: AgentConfig = Field(default_factory=AgentConfig)


class UserStory(BaseModel):
    """A single product-backlog item."""

    id: str
    title: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    story_points: int = 0
    status: Literal[
        "backlog", "planned", "in_progress", "in_review", "done", "rejected"
    ] = "backlog"
    deliverables: list[str] = Field(default_factory=list)


class Sprint(BaseModel):
    """Tracks a single sprint's scope and progress."""

    number: int
    goal: str = ""
    stories: list[UserStory] = Field(default_factory=list)
    planned_points: int = 0
    completed_points: int = 0
    duration_seconds: int = 300
    time_remaining: int = 300


class ScrumState(BaseModel):
    """Top-level state threaded through the ScrumFlow."""

    project_goal: str = ""
    product_backlog: list[UserStory] = Field(default_factory=list)
    current_sprint: Optional[Sprint] = None
    completed_sprints: list[Sprint] = Field(default_factory=list)
    sprint_number: int = 0
    velocity: int = 13
    max_sprints: int = 3
    sprint_duration_seconds: int = 300
    output_dir: str = "product"
    retro_action_items: list[str] = Field(default_factory=list)


# -- Structured output wrappers for crew tasks ------------------------------

class BacklogOutput(BaseModel):
    """Structured output from backlog refinement and planning crews."""

    stories: list[UserStory]


class ReviewVerdict(BaseModel):
    """Per-story accept/reject decision from the review crew."""

    story_id: str
    status: Literal["done", "rejected"]
    reason: str = ""


class ReviewOutput(BaseModel):
    """Structured output from the sprint review crew."""

    verdicts: list[ReviewVerdict]


class RetroOutput(BaseModel):
    """Structured output from the retrospective crew."""

    went_well: list[str] = Field(default_factory=list)
    needs_improvement: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
