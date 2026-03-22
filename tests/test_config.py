"""Tests for team configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from autoscrum.config import load_team_config


class TestLoadTeamConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_team_config(tmp_path / "nonexistent.yaml")
        assert cfg.developer.llm == "openai/gpt-4o"
        assert cfg.product_owner.llm == "openai/gpt-4o"

    def test_empty_file_returns_defaults(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_team_config(empty)
        assert cfg.developer.llm == "openai/gpt-4o"

    def test_partial_override(self, tmp_path):
        config = {"developer": {"llm": "ollama/llama3", "count": 2}}
        path = tmp_path / "team.yaml"
        path.write_text(yaml.dump(config))

        cfg = load_team_config(path)
        assert cfg.developer.llm == "ollama/llama3"
        assert cfg.developer.count == 2
        assert cfg.product_owner.llm == "openai/gpt-4o"

    def test_all_roles_overridden(self, tmp_path):
        config = {
            "product_owner": {"llm": "a/a"},
            "scrum_master": {"llm": "b/b"},
            "developer": {"llm": "c/c"},
            "qa_engineer": {"llm": "d/d"},
        }
        path = tmp_path / "team.yaml"
        path.write_text(yaml.dump(config))

        cfg = load_team_config(path)
        assert cfg.product_owner.llm == "a/a"
        assert cfg.scrum_master.llm == "b/b"
        assert cfg.developer.llm == "c/c"
        assert cfg.qa_engineer.llm == "d/d"

    def test_backstory_preserved(self, tmp_path):
        config = {"developer": {"backstory": "Veteran coder"}}
        path = tmp_path / "team.yaml"
        path.write_text(yaml.dump(config))

        cfg = load_team_config(path)
        assert cfg.developer.backstory == "Veteran coder"

    def test_accepts_string_path(self, tmp_path):
        path = tmp_path / "team.yaml"
        path.write_text(yaml.dump({"developer": {"llm": "test/model"}}))
        cfg = load_team_config(str(path))
        assert cfg.developer.llm == "test/model"
