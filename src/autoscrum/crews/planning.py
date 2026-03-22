"""Sprint Planning Crew: prioritize backlog by business value."""

from __future__ import annotations

import json

from crewai import Agent, Crew, Process, Task

from ..models import BacklogOutput, TeamConfig, UserStory


def build_planning_crew(
    backlog: list[UserStory],
    velocity: int,
    sprint_number: int,
    team: TeamConfig,
) -> Crew:
    """Build a crew that prioritizes the backlog.

    Roles per Scrum Guide:
      - Product Owner orders the backlog by value.
    Velocity-based selection is done programmatically by the flow.
    """
    po_cfg = team.product_owner

    product_owner = Agent(
        role="Product Owner",
        goal="Order the backlog by business value — most important first.",
        backstory=po_cfg.backstory or "Product Owner focused on value delivery.",
        llm=po_cfg.llm,
        verbose=False,
    )

    backlog_json = json.dumps(
        [s.model_dump(include={"id", "title", "story_points"}) for s in backlog],
        indent=2,
    )

    prioritize = Task(
        description=(
            f"Sprint {sprint_number} backlog:\n{backlog_json}\n\n"
            "Return these stories ordered by business value (most valuable first). "
            "Do not change any fields — only reorder."
        ),
        expected_output="The same stories reordered by priority.",
        agent=product_owner,
        output_pydantic=BacklogOutput,
    )

    return Crew(
        agents=[product_owner],
        tasks=[prioritize],
        process=Process.sequential,
        verbose=False,
    )
