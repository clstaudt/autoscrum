"""CLI entry point for AutoScrum."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .config import load_team_config
from .display import ScrumDisplay
from .flow import LLMConnectionError, ScrumFlow
from .listener import ScrumEventListener, install_stream_capture
from .preflight import preflight_check

app = typer.Typer(
    name="autoscrum",
    help="Autonomous Scrum powered by CrewAI agents.",
    add_completion=False,
)


def _silence_crewai_console() -> None:
    """Suppress CrewAI's built-in Rich console output so our Live display is the sole UI."""
    from crewai.events.event_listener import event_listener
    from crewai.events.utils.console_formatter import set_suppress_console_output

    set_suppress_console_output(True)
    event_listener.formatter.console = Console(file=io.StringIO())


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="What to build, in plain English.")],
    output_dir: Annotated[
        str,
        typer.Option(help="Directory for product deliverables. Auto-generated under products/ if omitted."),
    ] = "",
    team_config: Annotated[Path, typer.Option(help="Path to team YAML config file.")] = Path("team.yaml"),
    sprint_duration: Annotated[int, typer.Option(help="Sprint time-box in minutes.")] = 5,
    max_sprints: Annotated[int, typer.Option(help="Number of sprints to run.")] = 3,
    initial_velocity: Annotated[int, typer.Option(help="Story points for sprint 1 (adjusts empirically).")] = 13,
    no_code_execution: Annotated[bool, typer.Option("--no-code-execution", help="Disable running code via Docker during testing and review.")] = False,
    verbose: Annotated[bool, typer.Option(help="Show CrewAI's internal output alongside the board.")] = False,
) -> None:
    """Give the team a goal and watch them build it."""
    if not verbose:
        _silence_crewai_console()
        logging.getLogger("crewai").setLevel(logging.WARNING)

    team = load_team_config(team_config)
    console = Console(stderr=True)
    console.print("[dim]Pre-flight: checking LLM connectivity…[/dim]")
    try:
        preflight_check(team)
    except LLMConnectionError as exc:
        console.print(f"[red bold]Pre-flight failed:[/red bold] {exc}")
        raise typer.Exit(code=1)
    console.print("[green]Pre-flight passed.[/green]\n")

    display = ScrumDisplay()
    _listener = ScrumEventListener(display=display)
    install_stream_capture(display)

    flow = ScrumFlow(display=display)
    flow._team_config_path = str(team_config)

    display.start()
    try:
        flow.kickoff(
            inputs={
                "project_goal": goal,
                "sprint_duration_seconds": sprint_duration * 60,
                "max_sprints": max_sprints,
                "velocity": initial_velocity,
                "output_dir": output_dir,
                "enable_code_execution": not no_code_execution,
            }
        )
    except KeyboardInterrupt:
        display.log_activity("System", "Interrupted by user")
    except LLMConnectionError as exc:
        display.log_activity("System", f"[red]Cannot reach LLM — aborting: {exc}[/red]")
    except Exception as exc:
        display.log_activity("System", f"[red]Error: {exc}[/red]")
    finally:
        display.stop()


def main() -> None:
    """Entry point for the `autoscrum` console script."""
    app()


if __name__ == "__main__":
    main()
