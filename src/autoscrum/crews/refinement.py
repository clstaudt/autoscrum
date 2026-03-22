"""Backlog Refinement Crews: decompose + estimate as two separate steps."""

from __future__ import annotations

import json

from crewai import Agent, Crew, Process, Task

from ..models import BacklogOutput, TeamConfig, UserStory


def build_decompose_crew(
    project_goal: str,
    team: TeamConfig,
) -> Crew:
    """PO decomposes the project goal into user stories (what & why)."""
    po_cfg = team.product_owner

    product_owner = Agent(
        role="Product Owner",
        goal="Decompose the project goal into clear, small user stories.",
        backstory=(
            po_cfg.backstory
            or "Experienced Product Owner who writes clear, small user stories."
        ),
        llm=po_cfg.create_llm(),
        verbose=False,
    )

    decompose = Task(
        description=(
            f"Project goal: {project_goal}\n\n"
            "Create exactly 3 user stories. For each provide:\n"
            "- title: short name\n"
            "- description: one sentence\n"
            "- acceptance_criteria: list of 2 short criteria\n"
            "Set id to empty string, story_points to 0, status to 'backlog'."
        ),
        expected_output="3 user stories with title, description, and acceptance_criteria.",
        agent=product_owner,
        output_pydantic=BacklogOutput,
    )

    return Crew(
        agents=[product_owner],
        tasks=[decompose],
        process=Process.sequential,
        verbose=False,
    )


def build_estimate_crew(
    stories: list[UserStory],
    team: TeamConfig,
) -> Crew:
    """Developers estimate story points (how hard)."""
    dev_cfg = team.developer

    developer = Agent(
        role="Developer",
        goal="Estimate the technical effort of each user story.",
        backstory=(
            dev_cfg.backstory
            or "Pragmatic Developer who sizes work based on complexity."
        ),
        llm=dev_cfg.create_llm(),
        verbose=False,
    )

    stories_json = json.dumps(
        [s.model_dump(include={"id", "title", "description", "acceptance_criteria"})
         for s in stories],
        indent=2,
    )

    estimate = Task(
        description=(
            f"Stories to estimate:\n{stories_json}\n\n"
            "For each story assign story_points using Fibonacci sizing "
            "(1, 2, 3, or 5) based on technical complexity."
        ),
        expected_output="The stories with story_points estimates.",
        agent=developer,
        output_pydantic=BacklogOutput,
    )

    return Crew(
        agents=[developer],
        tasks=[estimate],
        process=Process.sequential,
        verbose=False,
    )
