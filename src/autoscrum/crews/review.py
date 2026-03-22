"""Sprint Review Crew: evaluate deliverables against acceptance criteria."""

from __future__ import annotations

import json

from crewai import Agent, Crew, Process, Task

from ..models import ReviewOutput, TeamConfig, UserStory


def build_review_crew(
    stories: list[UserStory],
    sprint_number: int,
    team: TeamConfig,
) -> Crew:
    """Build a crew that reviews sprint work and accepts or rejects stories."""
    po_cfg = team.product_owner
    qa_cfg = team.qa_engineer

    product_owner = Agent(
        role="Product Owner",
        goal="Accept or reject stories based on acceptance criteria.",
        backstory=po_cfg.backstory or "Product Owner with high quality standards.",
        llm=po_cfg.llm,
        verbose=False,
    )

    qa_engineer = Agent(
        role="QA Engineer",
        goal="Check whether acceptance criteria are satisfied.",
        backstory=qa_cfg.backstory or "Detail-oriented QA Engineer.",
        llm=qa_cfg.llm,
        verbose=False,
    )

    stories_json = json.dumps(
        [s.model_dump(include={"id", "title", "status", "acceptance_criteria", "deliverables"})
         for s in stories],
        indent=2,
    )

    review = Task(
        description=(
            f"Sprint {sprint_number} review:\n{stories_json}\n\n"
            "Check each story against its acceptance criteria."
        ),
        expected_output="Per-story pass/fail assessment.",
        agent=qa_engineer,
    )

    accept_reject = Task(
        description=(
            "For each story, decide: 'done' (accept) or 'rejected' (needs rework). "
            "Provide the story_id, status, and a brief reason."
        ),
        expected_output="Verdict for each story.",
        agent=product_owner,
        context=[review],
        output_pydantic=ReviewOutput,
    )

    return Crew(
        agents=[product_owner, qa_engineer],
        tasks=[review, accept_reject],
        process=Process.sequential,
        verbose=False,
    )
