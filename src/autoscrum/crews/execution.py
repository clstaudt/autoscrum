"""Sprint Execution Crew: implement, test, and verify stories within the time box."""

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
    enable_code_execution: bool = True,
    diary: str = "",
) -> Crew:
    """Build a crew that implements, tests, and verifies *stories* for the sprint."""
    dev_cfg = team.developer
    qa_cfg = team.qa_engineer

    writer = FileWriterTool()
    reader = FileReadTool()
    explorer = DirectoryReadTool(directory=output_dir)

    dev_tools = [writer, reader, explorer]
    qa_tools = [reader, explorer]

    if enable_code_execution:
        from crewai_tools import CodeInterpreterTool

        code_runner = CodeInterpreterTool()
        dev_tools.append(code_runner)
        qa_tools.append(code_runner)

    developer = Agent(
        role="Developer",
        goal="Implement user stories by writing working, tested code to disk.",
        backstory=dev_cfg.backstory or "Pragmatic Developer who writes clean, tested code.",
        llm=dev_cfg.create_llm(),
        tools=dev_tools,
        verbose=False,
    )

    qa_engineer = Agent(
        role="QA Engineer",
        goal=(
            "Verify code meets acceptance criteria by reading the written files"
            + (" and executing them" if enable_code_execution else "")
            + "."
        ),
        backstory=qa_cfg.backstory or "Thorough QA Engineer who validates through testing.",
        llm=qa_cfg.create_llm(),
        tools=qa_tools,
        verbose=False,
    )

    fields = {"id", "title", "description", "acceptance_criteria", "rejection_reason"}
    stories_json = json.dumps(
        [s.model_dump(include=fields) for s in stories],
        indent=2,
    )

    # -- Task 1: Implement -------------------------------------------------------
    impl_desc = (
        f"Sprint {sprint_number} — implement these stories:\n{stories_json}\n\n"
        "Use the directory read tool to see what already exists in the codebase.\n"
        "Then use the file writer tool to create or update files for each story.\n"
        f"All files MUST be written inside the '{output_dir}/' directory.\n\n"
        "IMPORTANT: You MUST use the file writer tool to produce working code. "
        "Do NOT just describe code — actually write it.\n\n"
        f"Example: File Writer Tool(filename='todo.py', directory='{output_dir}', "
        "content='print(\"hello\")', overwrite='true')"
    )

    if diary:
        impl_desc += f"\n\n--- Sprint diary (previous events) ---\n{diary}"

    implement = Task(
        description=impl_desc,
        expected_output="A list of files written (one per story minimum).",
        agent=developer,
    )

    tasks = [implement]

    # -- Task 2: Test (when code execution is enabled) ---------------------------
    if enable_code_execution:
        test = Task(
            description=(
                "Run every file produced in the implementation step to verify it "
                "executes without errors.\n\n"
                "For each file:\n"
                "1. Read its content with the file read tool.\n"
                "2. Execute it with the Code Interpreter tool. If the code needs "
                "external libraries, pass them via libraries_used.\n"
                "3. If execution fails, use the file writer tool to fix the code, "
                "then re-run it until it passes.\n\n"
                "Continue until every file runs successfully. Report the execution "
                "result (pass/fail and output) for each file."
            ),
            expected_output="Per-file test results showing pass/fail and output.",
            agent=developer,
            context=[implement],
        )
        tasks.append(test)

    # -- Task 3: QA Verify -------------------------------------------------------
    verify_description = (
        "Use the directory read tool to see what the Developer produced.\n"
        "Then read each file to review it against the acceptance criteria.\n"
    )
    if enable_code_execution:
        verify_description += (
            "Execute each file with the Code Interpreter tool to confirm it "
            "runs correctly and its output matches the acceptance criteria. "
            "Report any runtime errors or incorrect output.\n"
        )
    verify_description += (
        "\nFor each story set status to 'done' if criteria are met, "
        "or 'in_progress' if not."
    )

    verify = Task(
        description=verify_description,
        expected_output="Stories with updated statuses and deliverables.",
        agent=qa_engineer,
        context=tasks[:],
        output_pydantic=BacklogOutput,
    )
    tasks.append(verify)

    return Crew(
        agents=[developer, qa_engineer],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
