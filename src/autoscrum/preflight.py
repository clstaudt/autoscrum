"""Pre-flight LLM connectivity check.

Sends a tiny completion request to each unique LLM endpoint before the
Scrum flow starts, so configuration errors (bad key, wrong URL, server
down) surface immediately instead of silently advancing cards.
"""

from __future__ import annotations

from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

from .flow import LLMConnectionError
from .models import TeamConfig

_FATAL_LLM_ERRORS = (
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)


def preflight_check(team: TeamConfig) -> None:
    """Verify every unique LLM endpoint in *team* is reachable.

    Raises :class:`LLMConnectionError` on the first failure.
    """
    seen: set[tuple[str, str | None]] = set()

    for role in ("product_owner", "scrum_master", "developer", "qa_engineer"):
        cfg = getattr(team, role)
        key = (cfg.llm, cfg.base_url)
        if key in seen:
            continue
        seen.add(key)

        kwargs: dict = {
            "model": cfg.llm,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 3,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url

        try:
            completion(**kwargs)
        except _FATAL_LLM_ERRORS as exc:
            raise LLMConnectionError(
                f"LLM unreachable for role '{role}' "
                f"(model={cfg.llm}, base_url={cfg.base_url}): {exc}"
            ) from exc
        except Exception:
            pass
