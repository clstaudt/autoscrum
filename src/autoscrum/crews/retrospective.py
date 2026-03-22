"""Sprint Retrospective Crew: reflect on the sprint and identify improvements."""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..models import RetroOutput, TeamConfig


def build_retrospective_crew(
    sprint_number: int,
    planned_points: int,
    completed_points: int,
    team: TeamConfig,
) -> Crew:
    """Build a crew that reflects on the sprint and proposes improvements."""
    sm_cfg = team.scrum_master
    dev_cfg = team.developer

    scrum_master = Agent(
        role="Scrum Master",
        goal="Facilitate the retrospective and capture action items.",
        backstory=sm_cfg.backstory or "Reflective Scrum Master.",
        llm=sm_cfg.create_llm(),
        verbose=False,
    )

    developer = Agent(
        role="Developer",
        goal="Give honest feedback on the sprint.",
        backstory=dev_cfg.backstory or "Candid Developer.",
        llm=dev_cfg.create_llm(),
        verbose=False,
    )

    reflect = Task(
        description=(
            f"Sprint {sprint_number}: {completed_points}/{planned_points} points done.\n"
            "List 1-2 things that went well and 1-2 things to improve."
        ),
        expected_output="What went well and what to improve.",
        agent=developer,
    )

    summarise = Task(
        description=(
            "Summarise the retrospective. List what went well, what needs improvement, "
            "and 1-2 concrete action items for the next sprint."
        ),
        expected_output="Retrospective summary with action items.",
        agent=scrum_master,
        context=[reflect],
        output_pydantic=RetroOutput,
    )

    return Crew(
        agents=[scrum_master, developer],
        tasks=[reflect, summarise],
        process=Process.sequential,
        verbose=False,
    )
