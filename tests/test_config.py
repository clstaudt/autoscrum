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


class TestEnvVarDefaults:
    """AUTOSCRUM_LLM / AUTOSCRUM_BASE_URL environment variable integration."""

    def test_env_llm_applies_to_all_roles(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOSCRUM_LLM", "openai/custom-model")
        cfg = load_team_config(tmp_path / "nonexistent.yaml")
        for role in ("product_owner", "scrum_master", "developer", "qa_engineer"):
            assert getattr(cfg, role).llm == "openai/custom-model"

    def test_env_base_url_applies_to_all_roles(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOSCRUM_BASE_URL", "http://myhost:8000/v1")
        cfg = load_team_config(tmp_path / "nonexistent.yaml")
        for role in ("product_owner", "scrum_master", "developer", "qa_engineer"):
            assert getattr(cfg, role).base_url == "http://myhost:8000/v1"

    def test_yaml_role_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOSCRUM_LLM", "openai/from-env")
        config = {"product_owner": {"llm": "anthropic/from-yaml"}}
        path = tmp_path / "team.yaml"
        path.write_text(yaml.dump(config))

        cfg = load_team_config(path)
        assert cfg.product_owner.llm == "anthropic/from-yaml"
        assert cfg.developer.llm == "openai/from-env"

    def test_no_env_uses_hardcoded_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AUTOSCRUM_LLM", raising=False)
        monkeypatch.delenv("AUTOSCRUM_BASE_URL", raising=False)
        cfg = load_team_config(tmp_path / "nonexistent.yaml")
        assert cfg.developer.llm == "openai/gpt-4o"
        assert cfg.developer.base_url is None

    def test_env_llm_and_base_url_together(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTOSCRUM_LLM", "openai/local-model")
        monkeypatch.setenv("AUTOSCRUM_BASE_URL", "http://lab.local:8000/v1")
        cfg = load_team_config(tmp_path / "nonexistent.yaml")
        assert cfg.developer.llm == "openai/local-model"
        assert cfg.developer.base_url == "http://lab.local:8000/v1"

    def test_base_url_propagates_to_openai_api_base(self, tmp_path, monkeypatch):
        """AUTOSCRUM_BASE_URL must also set OPENAI_API_BASE so that
        instructor / raw OpenAI clients route to the custom endpoint."""
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.setenv("AUTOSCRUM_BASE_URL", "http://lab.local:9000/v1")
        load_team_config(tmp_path / "nonexistent.yaml")
        import os
        assert os.environ.get("OPENAI_API_BASE") == "http://lab.local:9000/v1"

    def test_openai_api_base_not_overwritten_if_already_set(self, tmp_path, monkeypatch):
        """If the user explicitly sets OPENAI_API_BASE, don't clobber it."""
        monkeypatch.setenv("OPENAI_API_BASE", "http://explicit:1234/v1")
        monkeypatch.setenv("AUTOSCRUM_BASE_URL", "http://lab.local:9000/v1")
        load_team_config(tmp_path / "nonexistent.yaml")
        import os
        assert os.environ.get("OPENAI_API_BASE") == "http://explicit:1234/v1"
