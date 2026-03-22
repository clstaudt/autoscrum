"""Custom CrewAI tools for reading/writing the product backlog as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

BACKLOG_PATH = Path("output/backlog.json")


class _ReadInput(BaseModel):
    status_filter: str = Field(
        default="",
        description="Optional status to filter by (backlog, planned, in_progress, in_review, done, rejected). Empty returns all.",
    )


class ReadBacklogTool(BaseTool):
    name: str = "read_backlog"
    description: str = (
        "Read the current product backlog. Returns a JSON list of user stories. "
        "Optionally filter by status."
    )
    args_schema: Type[BaseModel] = _ReadInput

    def _run(self, status_filter: str = "") -> str:
        if not BACKLOG_PATH.exists():
            return "[]"
        data = json.loads(BACKLOG_PATH.read_text())
        if status_filter:
            data = [s for s in data if s.get("status") == status_filter]
        return json.dumps(data, indent=2)


class _WriteInput(BaseModel):
    stories_json: str = Field(
        description="JSON string — a list of user-story objects to persist."
    )


class WriteBacklogTool(BaseTool):
    name: str = "write_backlog"
    description: str = (
        "Write (overwrite) the product backlog with a JSON list of user stories."
    )
    args_schema: Type[BaseModel] = _WriteInput

    def _run(self, stories_json: str) -> str:
        BACKLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        stories = json.loads(stories_json)
        BACKLOG_PATH.write_text(json.dumps(stories, indent=2))
        return f"Saved {len(stories)} stories to {BACKLOG_PATH}"
