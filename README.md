# dreaming

Dreaming reads the Claude Code and Codex sessions you ran in a project and writes what they established into a
folder of Markdown memory files. Future sessions of either tool can then read that folder instead of
rediscovering the same facts.

Each run has two steps. An analyzer agent reads each new session and adds its durable facts to the memory
folder. Then an organizer agent merges duplicates, marks conflicts and rebuilds the index.

This is a local version of "Dreaming with a dynamic workflow" from Anthropic's
[claude-cookbooks](https://github.com/anthropics/claude-cookbooks) (MIT License). The agents' prompts are
adapted from that cookbook's, and `src/dreaming/render.py` is copied from it unchanged.

## Requirements

- Linux or macOS. On Windows, use WSL.
- `git` and [`uv`](https://docs.astral.sh/uv/). uv downloads Python 3.14 for you.
- An Anthropic or OpenAI API key.
- A project folder where you have already used Claude Code or Codex. Dreaming reads the session logs those
  tools keep under `~/.claude` and `~/.codex`.
- For the daily run only: cron. See [Run every day](#run-every-day).

To install uv on a new machine:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal, so that the `uv` command is found.

## Set up in a project

Dreaming lives in a `dreaming/` folder inside each project that uses it, and keeps that project's memory there.
Run these from the project's root folder:

```bash
git clone https://github.com/HanranSong/dreaming.git dreaming
cp dreaming/.env.example dreaming/.env
uv sync --project dreaming
```

Then:

1. Open `dreaming/.env` and fill in your key. The comments in that file explain each setting.
   - With an Anthropic key: fill in `ANTHROPIC_API_KEY`. The other settings can stay empty.
   - With an OpenAI key: fill in `OPENAI_API_KEY`, and set both `DREAMING_ANALYZER_MODEL` and
     `DREAMING_ORGANIZER_MODEL` to an OpenAI model such as `gpt-6-luna`. Left empty, they use a Claude model,
     which needs an Anthropic key.
2. If the project is a git repository, add `dreaming/` to its `.gitignore`. The folder is a git repository of
   its own and holds your API key.

## Run

Always run from the project's root folder, not from inside `dreaming/`:

```bash
uv run --project dreaming dreaming
```

Dreaming finds sessions by the folder you run it from. It picks up only the sessions that were started in
exactly that folder, so a session started in a subfolder is not included.

A run analyzes every session it has not analyzed yet, and any session whose conversation changed since its last
analysis. Run it again whenever you like: sessions that are already done cost nothing.

| What you see | What it means |
| --- | --- |
| `Nothing to do.` | No new or changed sessions were found for this folder. |
| `No Claude Code or Codex session was started in <folder>` | Run it from the folder you start those tools in, the project's root folder. |
| `ANTHROPIC_API_KEY is not set` (or `OPENAI_API_KEY`) | The models in `dreaming/.env` need that key. Fill it in there. |
| `Not run <id>: waiting` | The session's log changed in the last 30 minutes, so it may still be in use. Run again later. |
| `failed` | The session is retried on the next run, up to 3 attempts. |

A run that analyzes a session calls the model and costs money. The cost of each agent is in the run's report.

### What a run writes

| Path | Contents |
| --- | --- |
| `dreaming/memory/` | The memory files, and `_index.md`, which lists them. |
| `dreaming/.state/runs/<run>/report.md` | What the run analyzed, what changed in memory, and what it cost. |
| `dreaming/.state/runs/<run>/memory_before/` and `sessions_before.json` | Memory and session state from before the run, which `undo` puts back. The latest five are kept. |
| `dreaming/.state/sessions.json` | Which sessions are done, failed or skipped. |

## Undo a run

To put memory back to how it was before the latest run:

```bash
uv run --project dreaming dreaming undo
```

Run it again to go back one more run. To go back before a particular run in one step, name it. A run's name
is its folder's name under `dreaming/.state/runs/`:

```bash
uv run --project dreaming dreaming undo 20261112-160548
```

- **The sessions are analyzed again.** Undo also forgets that the sessions analyzed since then were analyzed,
  so the next run takes them again. Without that, their facts would be gone for good.
- **An undo can be undone.** Each undo first saves what it replaces and prints the command that reverses it.
- **Five backups are kept.** You can go back before any of the latest five runs. An undo counts as a run.

## Run every day

The daily run uses cron, as the official cookbook does. macOS has it. Many Linux machines do not: check with
`command -v crontab`, and if nothing is printed, install it. On Debian or Ubuntu:

```bash
sudo apt install cron
```

Then, to turn a daily run on, from the project's root folder:

```bash
uv run --project dreaming dreaming schedule on
```

It runs at 01:00, this machine's time. To choose another time, add `--at`, for example `--at 04:30`. Turning
it on again with a new time replaces the old one.

```bash
uv run --project dreaming dreaming schedule status   # is it on, and for which folder
uv run --project dreaming dreaming schedule off
```

- **It does what a run by hand does**, over the folder you turned it on from. You can still run by hand at
  any time, and the two never run at once.
- **It spends money without asking.** A day with no new sessions costs nothing.
- **Each run's output** is added to `dreaming/.state/schedule.log`, and its report is under
  `dreaming/.state/runs/` as usual.
- **The machine has to be on** at that time, but you do not have to be logged in. A run missed while the
  machine was off or asleep is skipped, and the next run takes its sessions.
- **The schedule is one line in your crontab**, which `crontab -l` shows. Your other lines are left as they
  are.
- **If something it needs is missing**, `schedule on` says what and how to add it: cron to install or start,
  or an API key to fill in. `schedule status` repeats any warning.

## Let Claude Code and Codex read the memory

Neither tool reads `dreaming/memory/` on its own. You have to point them to it from the project's instruction
file.

The memory folder holds one Markdown file per topic and an index, `_index.md`, with one line per file. The
index refers to files as `[[slug]]`, which means the file named `<slug>.md` somewhere under `dreaming/memory/`.

### Both tools: add a note to `AGENTS.md`

Create `AGENTS.md` in the project's root folder, or add to the one you have:

```markdown
## Project memory

Notes from earlier sessions in this project are in `dreaming/memory/`. Before you start a task, read
`dreaming/memory/_index.md`, then open the files whose descriptions match the task. A `[[slug]]` in the
index is the file `<slug>.md` under `dreaming/memory/`. These are notes, not verified truth: check a fact
against the code before you rely on it. Do not edit this folder.
```

- **Codex** reads `AGENTS.md` at the start of every session.
- **Claude Code** reads `AGENTS.md` only when the project has no `CLAUDE.md`, and only from version 2.1.277.
  What works in every case is a `CLAUDE.md` in the project's root folder with this line in it, outside any
  code block:

  ```markdown
  @AGENTS.md
  ```

### Claude Code only: load the index into every session

Claude Code can import a file into `CLAUDE.md` with `@path`. Codex's documentation describes no such feature
for `AGENTS.md`, so Codex has to open the index itself, as the note above tells it to.

To have Claude Code start every session with the index already in context, add this line to the project's
`CLAUDE.md`, outside any code block:

```markdown
@dreaming/memory/_index.md
```

The path is relative to the `CLAUDE.md` that holds the line. Only the index is loaded this way. Claude Code
still opens the topic files when it needs them, so keep the `AGENTS.md` note as well. `_index.md` exists only
after the first run that analyzed a session.

## Update

```bash
git -C dreaming pull
uv sync --project dreaming
```

Your memory, state and `.env` are not touched by an update.
