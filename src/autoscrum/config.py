"""Load and merge team configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import AgentConfig, TeamConfig

_DEFAULTS = TeamConfig()


def load_team_config(path: str | Path = "team.yaml") -> TeamConfig:
    """Read *path* and return a validated TeamConfig, falling back to defaults
    for any missing fields."""
    p = Path(path)
    if not p.exists():
        return TeamConfig()

    with p.open() as f:
        raw: dict = yaml.safe_load(f) or {}

    roles: dict[str, AgentConfig] = {}
    for role in ("product_owner", "scrum_master", "developer", "qa_engineer"):
        default = getattr(_DEFAULTS, role)
        overrides = raw.get(role, {})
        roles[role] = AgentConfig(**{**default.model_dump(), **overrides})

    return TeamConfig(**roles)
