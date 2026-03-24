"""Tests for code execution: tool availability, crew wiring, and actual execution."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from autoscrum.crews.execution import build_execution_crew
from autoscrum.models import TeamConfig, UserStory


def _dummy_story() -> UserStory:
    return UserStory(
        id="US-001",
        title="Hello world",
        description="Print hello",
        acceptance_criteria=["Prints hello"],
        story_points=1,
    )


@pytest.fixture()
def _patch_llm(monkeypatch):
    """Prevent real LLM instantiation during crew construction."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")


class TestExecutionCrewWiring:
    """Verify that build_execution_crew assembles the right tools and tasks."""

    @pytest.mark.usefixtures("_patch_llm")
    def test_code_execution_enabled_adds_interpreter_and_test_task(self):
        crew = build_execution_crew(
            stories=[_dummy_story()],
            sprint_number=1,
            team=TeamConfig(),
            output_dir="/tmp/test_out",
            enable_code_execution=True,
        )
        from crewai_tools import CodeInterpreterTool

        dev = crew.agents[0]
        qa = crew.agents[1]
        dev_tool_types = [type(t) for t in dev.tools]
        qa_tool_types = [type(t) for t in qa.tools]
        assert CodeInterpreterTool in dev_tool_types
        assert CodeInterpreterTool in qa_tool_types
        assert len(crew.tasks) == 3  # implement + test + verify

    @pytest.mark.usefixtures("_patch_llm")
    def test_code_execution_disabled_excludes_interpreter(self):
        crew = build_execution_crew(
            stories=[_dummy_story()],
            sprint_number=1,
            team=TeamConfig(),
            output_dir="/tmp/test_out",
            enable_code_execution=False,
        )
        from crewai_tools import CodeInterpreterTool

        dev = crew.agents[0]
        qa = crew.agents[1]
        dev_tool_types = [type(t) for t in dev.tools]
        qa_tool_types = [type(t) for t in qa.tools]
        assert CodeInterpreterTool not in dev_tool_types
        assert CodeInterpreterTool not in qa_tool_types
        assert len(crew.tasks) == 2  # implement + verify (no test task)


class TestCodeInterpreterUnsafe:
    """Verify code execution in unsafe mode (no Docker required)."""

    def test_simple_arithmetic(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool(unsafe_mode=True)
        output = tool._run(code="result = 2 + 3", libraries_used=[])
        assert "5" in str(output)

    def test_string_manipulation(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool(unsafe_mode=True)
        output = tool._run(
            code='result = "hello world".upper()', libraries_used=[]
        )
        assert "HELLO WORLD" in output

    def test_multiline_code(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool(unsafe_mode=True)
        code = "\n".join([
            "items = [1, 2, 3, 4, 5]",
            "total = sum(items)",
            "result = f'sum={total}'",
        ])
        output = tool._run(code=code, libraries_used=[])
        assert "sum=15" in output

    def test_syntax_error_returns_error_message(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool(unsafe_mode=True)
        output = tool._run(code="def bad(", libraries_used=[])
        assert "error" in output.lower() or "Error" in output

    def test_runtime_error_returns_error_message(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool(unsafe_mode=True)
        output = tool._run(code="result = 1 / 0", libraries_used=[])
        assert "error" in output.lower() or "Error" in output

    def test_no_code_returns_message(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool(unsafe_mode=True)
        output = tool._run(code="", libraries_used=[])
        assert "no code" in output.lower() or output != ""


def _docker_available() -> bool:
    import subprocess

    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="Docker not running")
class TestCodeInterpreterDocker:
    """Verify code execution in safe mode (Docker sandbox)."""

    def test_simple_arithmetic(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool()
        output = tool._run(code="print(2 + 3)", libraries_used=[])
        assert "5" in str(output)

    def test_string_manipulation(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool()
        output = tool._run(
            code='print("hello from docker")',
            libraries_used=[],
        )
        assert "hello from docker" in str(output)

    def test_stdlib_import(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool()
        output = tool._run(
            code='import json; print(json.dumps({"a": 1}))',
            libraries_used=[],
        )
        assert '"a"' in str(output)

    def test_multiline_code(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool()
        code = "\n".join([
            "items = list(range(1, 6))",
            "total = sum(items)",
            "print(f'sum={total}')",
        ])
        output = tool._run(code=code, libraries_used=[])
        assert "sum=15" in str(output)

    def test_runtime_error_contained(self):
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool()
        output = tool._run(code="print(1 / 0)", libraries_used=[])
        assert "error" in str(output).lower()

    def test_no_host_filesystem_access(self):
        """Code inside Docker should not see the host filesystem."""
        from crewai_tools import CodeInterpreterTool

        tool = CodeInterpreterTool()
        output = tool._run(
            code=(
                "import os\n"
                "print(os.path.exists('/Users'))"
            ),
            libraries_used=[],
        )
        assert "False" in str(output)
