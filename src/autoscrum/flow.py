"""ScrumFlow: CrewAI Flow that orchestrates the full Scrum lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

from crewai import Crew
from crewai.flow.flow import Flow, listen, router, start
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)

from .config import load_team_config
from .crews.execution import build_execution_crew
from .crews.planning import build_planning_crew
from .crews.refinement import build_decompose_crew, build_estimate_crew
from .crews.retrospective import build_retrospective_crew
from .crews.review import build_review_crew
from .display import ScrumDisplay
from .listener import ScrumEventListener
from .models import (
    BacklogOutput,
    RetroOutput,
    ReviewOutput,
    ScrumState,
    Sprint,
    TeamConfig,
    UserStory,
)

_log = logging.getLogger(__name__)

T = TypeVar("T")

_FATAL_LLM_ERRORS = (
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)

_FATAL_ERROR_PATTERNS = (
    "AuthenticationError",
    "APIConnectionError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Incorrect API key",
    "Connection error",
)


class LLMConnectionError(RuntimeError):
    """Raised when the LLM is unreachable or rejects credentials."""


def _is_fatal(exc: Exception) -> bool:
    """Check if *exc* is a fatal LLM error, even when wrapped by CrewAI retries."""
    if isinstance(exc, _FATAL_LLM_ERRORS):
        return True
    msg = str(exc)
    return any(pattern in msg for pattern in _FATAL_ERROR_PATTERNS)


def _safe_kickoff(crew: Crew):
    """Run crew.kickoff(), returning the result or ``None`` on non-fatal failure.

    Fatal LLM errors (auth, connection, rate-limit) are re-raised as
    :class:`LLMConnectionError` so the flow can abort immediately — even
    when CrewAI wraps the original litellm exception in retry envelopes.
    """
    try:
        return crew.kickoff()
    except Exception as exc:
        if _is_fatal(exc):
            raise LLMConnectionError(str(exc)) from exc
        _log.warning("Crew kickoff failed (non-fatal): %s", exc)
        return None


def _assign_ids(stories: list[UserStory]) -> None:
    """Assign sequential IDs and ensure status is 'backlog'."""
    for i, story in enumerate(stories, start=1):
        story.id = f"US-{i:03d}"
        story.status = "backlog"
        if story.story_points < 1:
            story.story_points = 2


def _select_by_velocity(ordered: list[UserStory], velocity: int) -> list[UserStory]:
    """Greedily pick stories from the front of *ordered* that fit the velocity."""
    selected: list[UserStory] = []
    pts = 0
    for s in ordered:
        if pts + s.story_points <= velocity:
            selected.append(s)
            pts += s.story_points
    return selected


def _fallback_stories(project_goal: str) -> list[UserStory]:
    """Generate minimal placeholder stories when the LLM output can't be parsed."""
    return [
        UserStory(
            id="US-001",
            title="Project setup",
            description=f"Set up the initial project structure for: {project_goal}",
            acceptance_criteria=["Project directory exists", "Entry point runs"],
            story_points=2,
        ),
        UserStory(
            id="US-002",
            title="Core implementation",
            description=f"Implement the core functionality: {project_goal}",
            acceptance_criteria=["Core feature works", "Basic error handling"],
            story_points=5,
        ),
        UserStory(
            id="US-003",
            title="Tests and polish",
            description="Add tests and handle edge cases",
            acceptance_criteria=["Tests pass", "Clean output"],
            story_points=3,
        ),
    ]


class ScrumFlow(Flow[ScrumState]):
    """Orchestrates Refinement -> Planning -> Execution -> Review -> Retro in a loop."""

    display: ScrumDisplay
    team: TeamConfig

    def __init__(self, display: ScrumDisplay | None = None, **kwargs):
        super().__init__(**kwargs)
        self.display = display or ScrumDisplay()
        self.team = TeamConfig()

    # -- helpers -------------------------------------------------------------

    def _backlog_stories(self, *statuses: str) -> list[UserStory]:
        """Return stories from the product backlog matching any of *statuses*."""
        if not statuses:
            return list(self.state.product_backlog)
        return [s for s in self.state.product_backlog if s.status in statuses]

    def _compute_velocity(self) -> int:
        """Derive velocity from the average completed points of past sprints."""
        sprints = self.state.completed_sprints
        if not sprints:
            return self.state.velocity
        total = sum(s.completed_points for s in sprints)
        return max(1, round(total / len(sprints)))

    def _sync_display(self) -> None:
        """Push the current story list to the live display."""
        all_stories = list(self.state.product_backlog)
        if self.state.current_sprint:
            all_stories.extend(self.state.current_sprint.stories)
        self.display.sync_stories(all_stories)

    # -- flow steps ----------------------------------------------------------

    @start()
    def refine_backlog(self):
        """Ceremony 1: Backlog Refinement — decompose then estimate."""
        team_path = getattr(self, "_team_config_path", "team.yaml")
        self.team = load_team_config(team_path)

        self.display.set_ceremony("Backlog Refinement")
        self.display.log_activity("Scrum Master", "Starting backlog refinement")

        # Step 1: PO creates stories
        result = _safe_kickoff(build_decompose_crew(self.state.project_goal, self.team))

        stories = None
        if result and result.pydantic and isinstance(result.pydantic, BacklogOutput):
            stories = result.pydantic.stories

        if not stories:
            self.display.log_activity(
                "Scrum Master", "[yellow]Structured output failed — using fallback[/yellow]"
            )
            stories = _fallback_stories(self.state.project_goal)

        _assign_ids(stories)
        self.state.product_backlog = stories
        self._sync_display()
        self.display.log_activity(
            "Product Owner", f"Created {len(stories)} stories"
        )

        # Step 2: Developer estimates
        result = _safe_kickoff(build_estimate_crew(stories, self.team))
        if result and result.pydantic and isinstance(result.pydantic, BacklogOutput):
            estimated = result.pydantic.stories
            pts_map = {s.id: s.story_points for s in estimated if s.story_points > 0}
            for s in self.state.product_backlog:
                if s.id in pts_map:
                    s.story_points = pts_map[s.id]

        for s in self.state.product_backlog:
            if s.story_points < 1:
                s.story_points = 2

        self._sync_display()
        self.display.log_activity(
            "Developer", "Estimation complete"
        )
        return self.state.product_backlog

    @listen(refine_backlog)
    def plan_sprint(self, _=None):
        """Ceremony 2: Sprint Planning — select stories for the sprint."""
        self.state.sprint_number += 1
        self.display.set_ceremony("Sprint Planning")

        backlog = self._backlog_stories("backlog", "rejected")
        if not backlog:
            self.display.log_activity("Scrum Master", "Backlog empty — nothing to plan")
            return []

        self.display.set_sprint_info(
            number=self.state.sprint_number,
            max_sprints=self.state.max_sprints,
            goal=self.state.project_goal[:60],
            velocity=self.state.velocity,
        )

        crew = build_planning_crew(
            backlog=backlog,
            velocity=self.state.velocity,
            sprint_number=self.state.sprint_number,
            team=self.team,
        )
        result = _safe_kickoff(crew)

        ordered = backlog
        if result and result.pydantic and isinstance(result.pydantic, BacklogOutput):
            prioritised = result.pydantic.stories
            known_ids = {s.id for s in backlog}
            if prioritised and all(s.id in known_ids for s in prioritised):
                ordered = prioritised

        planned = _select_by_velocity(ordered, self.state.velocity)

        for s in planned:
            s.status = "planned"

        planned_points = sum(s.story_points for s in planned)
        self.state.current_sprint = Sprint(
            number=self.state.sprint_number,
            goal=self.state.project_goal[:80],
            stories=planned,
            planned_points=planned_points,
            duration_seconds=self.state.sprint_duration_seconds,
        )

        remaining_ids = {s.id for s in planned}
        self.state.product_backlog = [
            s for s in self.state.product_backlog if s.id not in remaining_ids
        ]

        self._sync_display()
        self.display.log_activity(
            "Scrum Master",
            f"Planned {len(planned)} stories ({planned_points} pts)",
        )
        return planned

    @listen(plan_sprint)
    def execute_sprint(self, _=None):
        """Ceremony 3: Sprint Execution — implement stories within the time box."""
        sprint = self.state.current_sprint
        if not sprint or not sprint.stories:
            return []

        self.display.set_ceremony("Sprint Execution")
        if self.state.enable_code_execution:
            self.display.log_activity(
                "Scrum Master",
                "Code execution enabled — agents will run and test code via Docker",
            )
        self.display.start_timer(self.state.sprint_duration_seconds)

        for s in sprint.stories:
            s.status = "in_progress"
        self._sync_display()

        crew = build_execution_crew(
            stories=sprint.stories,
            sprint_number=sprint.number,
            team=self.team,
            output_dir=self.state.output_dir,
            enable_code_execution=self.state.enable_code_execution,
        )

        try:
            result = asyncio.run(
                asyncio.wait_for(
                    crew.akickoff(),
                    timeout=self.state.sprint_duration_seconds,
                )
            )
        except RuntimeError:
            result = _safe_kickoff(crew)
        except asyncio.TimeoutError:
            self.display.log_activity(
                "Scrum Master",
                "[yellow]Sprint time box expired — moving incomplete work to review[/yellow]",
            )
            result = None
        except LLMConnectionError:
            raise
        except Exception as exc:
            if _is_fatal(exc):
                raise LLMConnectionError(str(exc)) from exc
            self.display.log_activity(
                "Scrum Master",
                f"[red]Execution error: {type(exc).__name__}: {exc}[/red]",
            )
            result = None

        self.display.stop_timer()

        if result and result.pydantic and isinstance(result.pydantic, BacklogOutput):
            by_id = {s.id: s for s in result.pydantic.stories}
            for s in sprint.stories:
                if s.id in by_id:
                    s.status = by_id[s.id].status
                    s.deliverables = by_id[s.id].deliverables

        for s in sprint.stories:
            if s.status == "in_progress":
                s.status = "in_review"

        self._sync_display()
        return sprint.stories

    @listen(execute_sprint)
    def review_sprint(self, _=None):
        """Ceremony 4: Sprint Review — accept or reject completed work."""
        sprint = self.state.current_sprint
        if not sprint:
            return []

        self.display.set_ceremony("Sprint Review")

        review_stories = [
            s for s in sprint.stories if s.status in ("done", "in_review")
        ]
        if not review_stories:
            self.display.log_activity("Product Owner", "No stories to review")
            return []

        crew = build_review_crew(
            stories=review_stories,
            sprint_number=sprint.number,
            team=self.team,
            output_dir=self.state.output_dir,
            enable_code_execution=self.state.enable_code_execution,
        )
        result = _safe_kickoff(crew)

        if result and result.pydantic and isinstance(result.pydantic, ReviewOutput):
            verdict_map = {v.story_id: v.status for v in result.pydantic.verdicts}
            for s in sprint.stories:
                if s.id in verdict_map:
                    s.status = verdict_map[s.id]
        else:
            for s in sprint.stories:
                if s.status == "in_review":
                    s.status = "done"

        completed_pts = sum(
            s.story_points for s in sprint.stories if s.status == "done"
        )
        sprint.completed_points = completed_pts

        rejected = [s for s in sprint.stories if s.status == "rejected"]
        for s in rejected:
            s.status = "backlog"
            self.state.product_backlog.append(s)

        self._sync_display()
        self.display.log_activity(
            "Product Owner",
            f"Accepted {completed_pts} pts, rejected {len(rejected)} stories",
        )
        return sprint.stories

    @listen(review_sprint)
    def run_retrospective(self, _=None):
        """Ceremony 5: Sprint Retrospective — reflect and adjust velocity."""
        sprint = self.state.current_sprint
        if not sprint:
            return "done"

        self.display.set_ceremony("Sprint Retrospective")

        crew = build_retrospective_crew(
            sprint_number=sprint.number,
            planned_points=sprint.planned_points,
            completed_points=sprint.completed_points,
            team=self.team,
        )
        result = _safe_kickoff(crew)

        if result and result.pydantic and isinstance(result.pydantic, RetroOutput):
            retro: RetroOutput = result.pydantic
            self.state.retro_action_items = retro.action_items

        self.state.completed_sprints.append(sprint)
        self.state.current_sprint = None

        self.state.velocity = self._compute_velocity()
        self._sync_display()
        self.display.log_activity(
            "Scrum Master",
            f"Velocity updated to {self.state.velocity} (empirical)",
        )

    @router(run_retrospective)
    def decide_next(self):
        """Route: continue to the next sprint or wrap up."""
        has_backlog = any(
            s.status in ("backlog", "rejected") for s in self.state.product_backlog
        )
        sprints_left = self.state.sprint_number < self.state.max_sprints
        if has_backlog and sprints_left:
            return "continue"
        return "done"

    @listen("continue")
    def next_sprint(self):
        """Loop back to planning for another sprint."""
        self.display.log_activity("Scrum Master", "Preparing next sprint...")
        return self.plan_sprint()

    @listen("done")
    def wrap_up(self):
        """Final summary after all sprints are complete."""
        from pathlib import Path

        self.display.set_ceremony("Project Complete")
        total_pts = sum(
            sp.completed_points for sp in self.state.completed_sprints
        )
        total_stories = sum(
            len([s for s in sp.stories if s.status == "done"])
            for sp in self.state.completed_sprints
        )
        self.display.log_activity(
            "Scrum Master",
            f"Finished {self.state.sprint_number} sprint(s) — "
            f"{total_stories} stories, {total_pts} points delivered",
        )
        remaining = self._backlog_stories("backlog", "rejected")
        if remaining:
            self.display.log_activity(
                "Product Owner",
                f"{len(remaining)} stories remain in the backlog",
            )

        output_dir = Path(self.state.output_dir)
        if output_dir.exists():
            files = sorted(p for p in output_dir.rglob("*") if p.is_file())
            if files:
                self.display.log_activity(
                    "Scrum Master",
                    f"[bold]Product output ({len(files)} files in {output_dir}/):[/bold]",
                )
                for f in files[:20]:
                    self.display.log_activity("  ", f"[dim]{f}[/dim]")
            else:
                self.display.log_activity(
                    "Scrum Master", "[yellow]No output files were produced[/yellow]"
                )
        else:
            self.display.log_activity(
                "Scrum Master", "[yellow]No output directory found[/yellow]"
            )
