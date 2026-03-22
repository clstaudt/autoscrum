"""Sprint Review Crew: evaluate deliverables against acceptance criteria."""

from __future__ import annotations

import json

from crewai import Agent, Crew, Process, Task
from crewai_tools import DirectoryReadTool, FileReadTool

from ..models import ReviewOutput, TeamConfig, UserStory


def build_review_crew(
    stories: list[UserStory],
    sprint_number: int,
    team: TeamConfig,
    output_dir: str = "product",
    enable_code_execution: bool = True,
) -> Crew:
    """Build a crew that reviews sprint work and accepts or rejects stories."""
    po_cfg = team.product_owner
    qa_cfg = team.qa_engineer

    reader = FileReadTool()
    explorer = DirectoryReadTool(directory=output_dir)
    qa_tools: list = [reader, explorer]

    if enable_code_execution:
        from crewai_tools import CodeInterpreterTool

        qa_tools.append(CodeInterpreterTool())

    product_owner = Agent(
        role="Product Owner",
        goal="Accept or reject stories based on acceptance criteria.",
        backstory=po_cfg.backstory or "Product Owner with high quality standards.",
        llm=po_cfg.llm,
        verbose=False,
    )

    qa_engineer = Agent(
        role="QA Engineer",
        goal=(
            "Validate deliverables against acceptance criteria"
            + (" by reading and executing the code" if enable_code_execution else "")
            + "."
        ),
        backstory=qa_cfg.backstory or "Detail-oriented QA Engineer who validates through testing.",
        llm=qa_cfg.llm,
        tools=qa_tools,
        verbose=False,
    )

    stories_json = json.dumps(
        [s.model_dump(include={"id", "title", "status", "acceptance_criteria", "deliverables"})
         for s in stories],
        indent=2,
    )

    review_description = f"Sprint {sprint_number} review:\n{stories_json}\n\n"
    if enable_code_execution:
        review_description += (
            "For each story:\n"
            "1. Read the deliverable files with the file read tool.\n"
            "2. Execute the code with the Code Interpreter tool to verify it "
            "runs without errors.\n"
            "3. Check that the runtime output and behaviour satisfy the "
            "acceptance criteria.\n"
            "4. Report pass/fail per story with evidence (output, errors, or "
            "missing behaviour)."
        )
    else:
        review_description += "Check each story against its acceptance criteria."

    review = Task(
        description=review_description,
        expected_output="Per-story pass/fail assessment with evidence.",
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
