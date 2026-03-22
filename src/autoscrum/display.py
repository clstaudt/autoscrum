"""Rich-based live terminal UI: Kanban board, sprint timer, agent activity feed."""

from __future__ import annotations

import threading
import time
from collections import deque

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .models import UserStory

_STATUS_STYLE = {
    "backlog": ("dim white", "⬜"),
    "planned": ("cyan", "📋"),
    "in_progress": ("yellow", "🔨"),
    "in_review": ("magenta", "🔍"),
    "done": ("green", "✅"),
    "rejected": ("red", "❌"),
}

_BOARD_COLUMNS = ["backlog", "planned", "in_progress", "in_review", "done"]

_COLUMN_HEADER = {
    "backlog": "BACKLOG",
    "planned": "PLANNED",
    "in_progress": "IN PROGRESS",
    "in_review": "IN REVIEW",
    "done": "DONE",
}

MAX_ACTIVITY_LINES = 14
MAX_LLM_OUTPUT_LINES = 20


class ScrumDisplay:
    """Thread-safe live display driven by state mutations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._console = Console()
        self._live: Live | None = None
        self._ceremony: str = ""
        self._activity: deque[str] = deque(maxlen=MAX_ACTIVITY_LINES)
        self._stories: list[UserStory] = []
        self._sprint_number: int = 0
        self._max_sprints: int = 3
        self._sprint_goal: str = ""
        self._timer_end: float | None = None
        self._timer_duration: int = 0
        self._velocity: int = 0
        self._completed_points: int = 0
        self._working_role: str | None = None
        self._working_since: float | None = None
        self._llm_calls: int = 0
        self._llm_role: str = ""
        self._llm_model: str = ""
        self._llm_buf: str = ""
        self._llm_lines: deque[str] = deque(maxlen=MAX_LLM_OUTPUT_LINES)
        self._refresh_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Begin the live-refreshing terminal display."""
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()
        self._stop_event.clear()
        self._refresh_thread = threading.Thread(
            target=self._auto_refresh_loop, daemon=True
        )
        self._refresh_thread.start()

    def stop(self) -> None:
        """Tear down the live display."""
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2)
            self._refresh_thread = None
        if self._live:
            self._live.stop()
            self._live = None

    # -- state setters (thread-safe) -----------------------------------------

    def set_ceremony(self, name: str) -> None:
        """Update the current ceremony label shown on the board."""
        with self._lock:
            self._ceremony = name
            self._log(f"[bold dim]── {name} ──[/bold dim]")
        self._refresh()

    def set_sprint_info(
        self,
        number: int,
        max_sprints: int,
        goal: str,
        velocity: int,
    ) -> None:
        """Set header metadata for the active sprint."""
        with self._lock:
            self._sprint_number = number
            self._max_sprints = max_sprints
            self._sprint_goal = goal
            self._velocity = velocity
        self._refresh()

    def sync_stories(self, stories: list[UserStory]) -> None:
        """Replace the displayed story list and recalculate points."""
        with self._lock:
            self._stories = list(stories)
            self._completed_points = sum(
                s.story_points for s in stories if s.status == "done"
            )
        self._refresh()

    def start_timer(self, duration_seconds: int) -> None:
        """Start the countdown timer displayed in the header."""
        with self._lock:
            self._timer_duration = duration_seconds
            self._timer_end = time.monotonic() + duration_seconds
        self._refresh()

    def stop_timer(self) -> None:
        """Clear the countdown timer."""
        with self._lock:
            self._timer_end = None
        self._refresh()

    def log_activity(self, role: str, message: str) -> None:
        """Append a line to the scrolling activity feed."""
        with self._lock:
            self._log(f"[bold]{role}[/bold]: {message}")
        self._refresh()

    def set_working(self, role: str | None) -> None:
        """Track which agent is currently thinking."""
        with self._lock:
            self._working_role = role
            self._working_since = time.monotonic() if role else None

    def bump_llm_calls(self) -> None:
        """Increment the LLM call counter."""
        with self._lock:
            self._llm_calls += 1

    def begin_llm_call(self, role: str, model: str) -> None:
        """Signal that a new LLM call has started; flush current buffer line."""
        with self._lock:
            if self._llm_buf:
                self._llm_lines.append(self._llm_buf)
                self._llm_buf = ""
            self._llm_role = role
            self._llm_model = model
            short = model.rsplit("/", 1)[-1] if model else "…"
            self._llm_lines.append(f"[dim]── {role} → {short} ──[/dim]")

    def append_llm_chunk(self, text: str) -> None:
        """Append a streaming token chunk to the rolling LLM output."""
        with self._lock:
            self._llm_buf += text
            while "\n" in self._llm_buf:
                line, self._llm_buf = self._llm_buf.split("\n", 1)
                self._llm_lines.append(line)

    def finish_llm_call(self) -> None:
        """Flush remaining buffer after an LLM call completes."""
        with self._lock:
            if self._llm_buf:
                self._llm_lines.append(self._llm_buf)
                self._llm_buf = ""

    # -- internals -----------------------------------------------------------

    def _log(self, line: str) -> None:
        self._activity.append(line)

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _auto_refresh_loop(self) -> None:
        """Background thread that refreshes every second for live indicators."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1.0)
            if not self._stop_event.is_set():
                self._refresh()

    def _time_remaining(self) -> str:
        if self._timer_end is None:
            return ""
        remaining = max(0, self._timer_end - time.monotonic())
        mins, secs = divmod(int(remaining), 60)
        return f"{mins}:{secs:02d}"

    # -- rendering -----------------------------------------------------------

    def _render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="board", ratio=2),
            Layout(name="footer", ratio=3, minimum_size=8),
        )

        layout["footer"].split_row(
            Layout(name="activity", ratio=3),
            Layout(name="llm_output", ratio=2),
        )

        layout["header"].update(self._render_header())
        layout["board"].update(self._render_board())
        layout["footer"]["activity"].update(self._render_activity())
        layout["footer"]["llm_output"].update(self._render_llm_output())

        return Panel(
            layout,
            title="[bold blue]AUTOSCRUM[/bold blue]",
            border_style="blue",
        )

    def _render_header(self) -> Panel:
        parts: list[str] = []
        if self._sprint_number:
            parts.append(
                f"[bold]Sprint {self._sprint_number}/{self._max_sprints}[/bold]"
            )
        if self._sprint_goal:
            parts.append(f"Goal: [italic]{self._sprint_goal}[/italic]")
        timer = self._time_remaining()
        if timer:
            parts.append(f"[bold yellow]⏱  {timer}[/bold yellow]")
        if self._velocity:
            parts.append(
                f"[dim]{self._completed_points}/{self._velocity} pts[/dim]"
            )

        header_text = "  ·  ".join(parts) if parts else "[dim]Initializing...[/dim]"
        return Panel(
            Align.center(Text.from_markup(header_text)),
            style="bold",
            border_style="blue",
        )

    def _render_card(self, story: UserStory) -> Panel:
        """Render a single story as a compact one-line card."""
        style, _ = _STATUS_STYLE.get(story.status, ("white", "·"))
        line = Text.from_markup(
            f"[bold]{story.story_points}[/bold] {story.title}  [dim]{story.id}[/dim]"
        )
        return Panel(line, border_style=style, padding=(0, 1), expand=True)

    def _render_board(self) -> Panel:
        """Render the Kanban board as a table with one column per status."""
        board = Table(
            show_header=True,
            show_edge=False,
            pad_edge=True,
            expand=True,
            show_lines=False,
        )

        for col_key in _BOARD_COLUMNS:
            style, _ = _STATUS_STYLE[col_key]
            board.add_column(
                _COLUMN_HEADER[col_key],
                style=style,
                justify="center",
                ratio=1,
                no_wrap=False,
            )

        with self._lock:
            grouped: dict[str, list[UserStory]] = {k: [] for k in _BOARD_COLUMNS}
            for story in self._stories:
                if story.status in grouped:
                    grouped[story.status].append(story)

        max_rows = max((len(v) for v in grouped.values()), default=0)
        if max_rows == 0:
            board.add_row(
                *[Text("—", style="dim", justify="center") for _ in _BOARD_COLUMNS]
            )
        else:
            for i in range(max_rows):
                cells: list[RenderableType] = []
                for col_key in _BOARD_COLUMNS:
                    stories_in_col = grouped[col_key]
                    if i < len(stories_in_col):
                        cells.append(self._render_card(stories_in_col[i]))
                    else:
                        cells.append(Text(""))
                board.add_row(*cells)

        return Panel(
            board,
            title=f"[bold]{self._ceremony}[/bold]" if self._ceremony else "[bold]Board[/bold]",
            border_style="dim",
        )

    def _render_activity(self) -> Panel:
        with self._lock:
            lines = list(self._activity)

        working = self._working_status_renderable()
        if working:
            lines_text = "\n".join(lines) + "\n" if lines else ""
            content = Group(Text.from_markup(lines_text), working)
        elif lines:
            content = Text.from_markup("\n".join(lines))
        else:
            content = Text("Waiting for agents...", style="dim")

        return Panel(
            content,
            title="[bold]Activity[/bold]",
            border_style="dim",
        )

    def _render_llm_output(self) -> Panel:
        """Render a rolling-window panel of streamed LLM tokens."""
        with self._lock:
            lines = list(self._llm_lines)
            partial = self._llm_buf

        if not lines and not partial:
            content: RenderableType = Text("Waiting for LLM…", style="dim")
        else:
            visible = "\n".join(lines)
            if partial:
                visible = f"{visible}\n{partial}" if visible else partial
            content = Text.from_markup(visible)

        return Panel(
            content,
            title="[bold]LLM Output[/bold]",
            border_style="dim",
        )

    def _working_status_renderable(self) -> RenderableType | None:
        """Render a Rich Spinner with elapsed time for the active agent."""
        if not self._working_role or not self._working_since:
            return None
        elapsed = int(time.monotonic() - self._working_since)
        label = Text.from_markup(
            f" [yellow]{self._working_role}[/yellow] thinking… "
            f"[dim]({elapsed}s · {self._llm_calls} LLM calls)[/dim]"
        )
        return Spinner("dots", text=label, style="yellow")
