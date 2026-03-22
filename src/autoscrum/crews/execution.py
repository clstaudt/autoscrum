"""Sprint Execution Crew: implement stories within the time box."""

from __future__ import annotations

import json

from crewai import Agent, Crew, Process, Task
from crewai_tools import DirectoryReadTool, FileReadTool, FileWriterTool

from ..models import BacklogOutput, TeamConfig, UserStory


def build_execution_crew(
    stories: list[UserStory],
    sprint_number: int,
    team: TeamConfig,
    output_dir: str = "output",
) -> Crew:
    """Build a crew that implements *stories* for the sprint."""
    dev_cfg = team.developer
    qa_cfg = team.qa_engineer

    writer = FileWriterTool()
    reader = FileReadTool()
    explorer = DirectoryReadTool(directory=output_dir)

    developer = Agent(
        role="Developer",
        goal="Implement user stories by writing code to disk.",
        backstory=dev_cfg.backstory or "Pragmatic Developer who writes clean code.",
        llm=dev_cfg.llm,
        tools=[writer, reader, explorer],
        verbose=False,
    )

    qa_engineer = Agent(
        role="QA Engineer",
        goal="Verify code meets acceptance criteria by reading the written files.",
        backstory=qa_cfg.backstory or "Thorough QA Engineer.",
        llm=qa_cfg.llm,
        tools=[reader, explorer],
        verbose=False,
    )

    stories_json = json.dumps(
        [s.model_dump(include={"id", "title", "description", "acceptance_criteria"})
         for s in stories],
        indent=2,
    )

    implement = Task(
        description=(
            f"Sprint {sprint_number} — implement these stories:\n{stories_json}\n\n"
            "Use the directory read tool to see what already exists in the codebase.\n"
            "Then use the file writer tool to create or update files for each story.\n"
            f"All files MUST be written inside the '{output_dir}/' directory.\n\n"
            "IMPORTANT: You MUST use the file writer tool for each story. "
            "Do NOT just describe code — actually write it.\n\n"
            f"Example: File Writer Tool(filename='todo.py', directory='{output_dir}', "
            "content='print(\"hello\")', overwrite='true')\n\n"
            "Write one file per story at minimum."
        ),
        expected_output="A list of files written (one per story minimum).",
        agent=developer,
    )

    verify = Task(
        description=(
            "Use the directory read tool to see what the Developer produced.\n"
            "Then read each file to review it against the acceptance criteria.\n"
            "For each story set status to 'done' if criteria are met, "
            "or 'in_progress' if not."
        ),
        expected_output="Stories with updated statuses and deliverables.",
        agent=qa_engineer,
        context=[implement],
        output_pydantic=BacklogOutput,
    )

    return Crew(
        agents=[developer, qa_engineer],
        tasks=[implement, verify],
        process=Process.sequential,
        verbose=False,
    )
