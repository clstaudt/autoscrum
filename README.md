<p align="center">
  <h1 align="center">AutoScrum</h1>
  <p align="center">
    <strong>A self-organizing AI team that builds software using Scrum.</strong>
  </p>
  <p align="center">
    Give it a goal. Watch the sprints. Get the code.
  </p>
</p>

---

> [!WARNING]
> AutoScrum is in early development and should be considered a proof of concept.
> Expect rough edges, breaking changes, and LLM-dependent output quality.

AutoScrum turns a one-line project description into working software. It spins up
an AI-powered Scrum team — Product Owner, Scrum Master, Developers, QA — that
refines a backlog, plans sprints, writes code, reviews it, and retrospects. You
watch the whole process unfold on a live Kanban board in your terminal.

## Quick Start

```bash
uv sync

uv run autoscrum "Build a CLI calculator in Python"
```

That's it. AutoScrum will:

1. **Refine** the goal into user stories with acceptance criteria
2. **Plan** a sprint by prioritizing and selecting stories
3. **Execute** — developers write real code files to `product/`
4. **Review** — QA reads the code and verifies acceptance criteria
5. **Retrospect** — the team reflects and adapts

Repeat for as many sprints as configured.

## The Live Board

<p align="center">
  <img src="docs/board.png" alt="AutoScrum live Kanban board" width="720">
</p>

Stories move across columns in real time. A spinner shows who's thinking, for how
long, and how many LLM calls have been made.

## Configuration

### Team (`team.yaml`)

Each role maps to any LLM supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

```yaml
product_owner:
  llm: anthropic/claude-sonnet-4-20250514

scrum_master:
  llm: openai/gpt-4o

developer:
  llm: ollama/qwen3:8b

qa_engineer:
  llm: ollama/codellama
```

Mix cloud APIs and local models freely. Run different models per role — give your
developer a coding-tuned model while the PO uses a reasoning model.

### CLI

```
uv run autoscrum [OPTIONS] GOAL
```

**GOAL** — what to build, in plain English. This is the only required argument.

```bash
# Minimal — just a goal
uv run autoscrum "Build a markdown blog engine"

# Full control
uv run autoscrum "Build a REST API for a bookstore" \
  --output-dir bookstore \
  --team-config team.yaml \
  --sprint-duration 600 \
  --max-sprints 3 \
  --initial-velocity 13
```

| Option | Default | Description |
|---|---|---|
| `--output-dir DIR` | `product` | Directory where the product code is written. Created automatically. Gitignored by default. |
| `--team-config FILE` | `team.yaml` | Team configuration file (LLM assignments per role). |
| `--sprint-duration SEC` | `300` | Wall-clock time-box per sprint in seconds. Incomplete work returns to the backlog when time runs out. |
| `--max-sprints N` | `3` | How many sprints to run before finishing. |
| `--initial-velocity N` | `13` | Story points for sprint 1. Adjusts empirically after each sprint based on completed work. |
| `--verbose` | off | Show CrewAI's internal output alongside the live board. Useful for debugging. |

### Local LLMs

AutoScrum works with any OpenAI-compatible server. For [Ollama](https://ollama.com):

```bash
ollama pull qwen3:8b
uv run autoscrum "Build a todo app" --team-config team.yaml
```

Point to a custom server by adding `base_url` in the team config:

```yaml
developer:
  llm: openai/my-model
  base_url: http://localhost:8080/v1
```

## What Gets Built

The product is real files on disk — not summaries, not descriptions. Developers use
file tools to read, write, and explore the codebase across sprints. Each sprint
builds on the last. At the end, `--output-dir` contains the project.

## Why

Most agent demos are single-shot: one prompt, one response. Real software
development is iterative. AutoScrum explores what happens when you give AI agents
a proven process framework instead of a blank canvas:

- **Structure reduces hallucination** — ceremonies constrain what each agent focuses on
- **Roles create accountability** — the PO doesn't write code, the developer doesn't prioritize
- **Time-boxing forces shipping** — incomplete work goes back to the backlog, not into an infinite loop
- **Retrospectives enable adaptation** — the team learns from each sprint

## Requirements

- Python 3.10 – 3.13
- [uv](https://docs.astral.sh/uv/)
- At least one LLM (cloud API key or local server)

## License

GPL-3.0
