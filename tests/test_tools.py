"""Tests for custom backlog tools (filesystem-based, no LLM)."""

from __future__ import annotations

import json

import autoscrum.tools.backlog as backlog_mod
from autoscrum.tools.backlog import ReadBacklogTool, WriteBacklogTool


class TestReadBacklogTool:
    def test_missing_file_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backlog_mod, "BACKLOG_PATH", tmp_path / "missing.json")
        tool = ReadBacklogTool()
        assert json.loads(tool._run()) == []

    def test_reads_all_stories(self, tmp_path, monkeypatch):
        path = tmp_path / "backlog.json"
        stories = [{"id": "US-001", "status": "backlog"}, {"id": "US-002", "status": "done"}]
        path.write_text(json.dumps(stories))
        monkeypatch.setattr(backlog_mod, "BACKLOG_PATH", path)

        tool = ReadBacklogTool()
        result = json.loads(tool._run())
        assert len(result) == 2

    def test_filters_by_status(self, tmp_path, monkeypatch):
        path = tmp_path / "backlog.json"
        stories = [{"id": "US-001", "status": "backlog"}, {"id": "US-002", "status": "done"}]
        path.write_text(json.dumps(stories))
        monkeypatch.setattr(backlog_mod, "BACKLOG_PATH", path)

        tool = ReadBacklogTool()
        result = json.loads(tool._run(status_filter="done"))
        assert len(result) == 1
        assert result[0]["id"] == "US-002"


class TestWriteBacklogTool:
    def test_writes_and_creates_directory(self, tmp_path, monkeypatch):
        path = tmp_path / "sub" / "backlog.json"
        monkeypatch.setattr(backlog_mod, "BACKLOG_PATH", path)

        stories = [{"id": "US-001", "title": "Test"}]
        tool = WriteBacklogTool()
        result = tool._run(stories_json=json.dumps(stories))

        assert "1 stories" in result
        assert path.exists()
        assert json.loads(path.read_text()) == stories

    def test_overwrites_existing(self, tmp_path, monkeypatch):
        path = tmp_path / "backlog.json"
        path.write_text("[]")
        monkeypatch.setattr(backlog_mod, "BACKLOG_PATH", path)

        new_stories = [{"id": "US-099", "title": "New"}]
        WriteBacklogTool()._run(stories_json=json.dumps(new_stories))

        assert json.loads(path.read_text()) == new_stories
