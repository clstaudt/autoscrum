"""Load and merge team configuration from YAML and environment variables.

Merge order (later wins):
    hard-coded defaults  →  env vars (AUTOSCRUM_LLM, AUTOSCRUM_BASE_URL)  →  per-role YAML
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import AgentConfig, TeamConfig

_ROLES = ("product_owner", "scrum_master", "developer", "qa_engineer")
_HARD_DEFAULTS = AgentConfig().model_dump()


def _env_defaults() -> dict:
    """Read global LLM settings from environment variables.

    Also propagates ``AUTOSCRUM_BASE_URL`` to ``OPENAI_API_BASE`` so that
    libraries that create their own OpenAI client (e.g. *instructor*, used by
    CrewAI for structured output) route through the same custom endpoint.
    """
    env: dict = {}
    if llm := os.environ.get("AUTOSCRUM_LLM"):
        env["llm"] = llm
    if base_url := os.environ.get("AUTOSCRUM_BASE_URL"):
        env["base_url"] = base_url
        os.environ.setdefault("OPENAI_API_BASE", base_url)
    return env


def load_team_config(path: str | Path = "team.yaml") -> TeamConfig:
    """Read *path* and return a validated :class:`TeamConfig`.

    LLM defaults come from ``AUTOSCRUM_LLM`` / ``AUTOSCRUM_BASE_URL``
    environment variables.  Per-role sections in the YAML override those
    defaults for individual roles.
    """
    p = Path(path)
    raw: dict = {}
    if p.exists():
        with p.open() as f:
            raw = yaml.safe_load(f) or {}

    env = _env_defaults()

    roles: dict[str, AgentConfig] = {}
    for role in _ROLES:
        merged = {**_HARD_DEFAULTS, **env, **raw.get(role, {})}
        roles[role] = AgentConfig(**merged)

    return TeamConfig(**roles)
