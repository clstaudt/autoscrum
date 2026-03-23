"""Sprint Retrospective Crew: reflect on the sprint and identify improvements."""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task

from ..models import RetroOutput, TeamConfig


def build_retrospective_crew(
    sprint_number: int,
    planned_points: int,
    completed_points: int,
    team: TeamConfig,
    diary: str = "",
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

    reflect_desc = (
        f"Sprint {sprint_number}: {completed_points}/{planned_points} points done.\n"
        "Reflect on the sprint. What went well? What should improve?"
    )
    if diary:
        reflect_desc += f"\n\n--- Sprint diary (previous events) ---\n{diary}"

    reflect = Task(
        description=reflect_desc,
        expected_output="What went well and what to improve.",
        agent=developer,
    )

    summarise = Task(
        description=(
            "Summarise the retrospective. Capture what went well, what needs "
            "improvement, and concrete action items for the next sprint."
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
