"""Read Claude Code session logs into the export shape render.py expects."""

import json
import re
from pathlib import Path

# The preview Claude Code leaves in the log when a tool result is too large to keep inline.
PERSISTED_OUTPUT = re.compile(
    r"<persisted-output>\s*[^\n]*Full output saved to: [^\n]*/tool-results/([^\s/]+)"
)


def _records(path: Path):
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a session still being written can end in a partial line


def _started_at(path: Path) -> str:
    """The log's first message time. These are UTC ISO timestamps, so they sort as strings."""
    for record in _records(path):
        if record.get("type") in ("user", "assistant"):
            return record.get("timestamp") or ""
    return ""


def session_files(project_root: Path) -> list[Path]:
    """The session logs Claude Code keeps for this project, earliest session first."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project_root))
    return sorted((Path.home() / ".claude" / "projects" / slug).glob("*.jsonl"), key=_started_at)


def _full_result(content, session_dir: Path):
    """A tool result's complete output: the saved file replaces Claude Code's preview, and images are dropped."""
    if isinstance(content, str) and (match := PERSISTED_OUTPUT.match(content)):
        saved = session_dir / "tool-results" / match.group(1)
        if saved.is_file():
            return saved.read_text(encoding="utf-8", errors="replace")
    if isinstance(content, list):
        # Base64 images would otherwise be rendered into the transcript as text.
        return [block for block in content if block.get("type") != "image"]
    return content


def _events(record: dict, session_dir: Path):
    """Turn one user or assistant record into transcript events. Thinking is left out, as in the official export."""
    content = record["message"]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for block in content:
        kind = block.get("type")
        if kind == "text":
            event = "user.message" if record["type"] == "user" else "agent.message"
            yield {"type": event, "content": block["text"]}
        elif kind == "tool_use":
            yield {
                "type": "agent.tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": block["input"],
            }
        elif kind == "tool_result":
            yield {
                "type": "agent.tool_result",
                "tool_use_id": block["tool_use_id"],
                "content": _full_result(block.get("content"), session_dir),
                "is_error": block.get("is_error"),
            }


def _log_events(path: Path, session_dir: Path) -> list[dict]:
    events = []
    for record in _records(path):
        # isMeta marks text the harness injected, such as skill bodies, rather than what the user wrote.
        if record.get("type") in ("user", "assistant") and not record.get("isMeta"):
            events.extend(_events(record, session_dir))
    return events


def _subagent_threads(session_dir: Path) -> list[dict]:
    """Subagents log to their own files, under subagents/ directly or in a workflow's folder."""
    threads = []
    for log in sorted(session_dir.glob("subagents/**/agent-*.jsonl"), key=_started_at):
        meta_file = log.with_suffix(".meta.json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        threads.append(
            {
                "id": log.stem.removeprefix("agent-"),
                "parent_thread_id": "primary",
                "agent_name": meta.get("description") or meta.get("agentType") or "subagent",
                "events": _log_events(log, session_dir),
            }
        )
    return threads


def load_session(path: Path) -> dict:
    """One session log and its subagents as the dict render_text takes."""
    session_dir = path.with_suffix("")
    title = ""
    for record in _records(path):
        if record.get("type") == "ai-title":
            title = record.get("aiTitle", title)
    primary = {
        "id": "primary",
        "parent_thread_id": None,
        "agent_name": "claude-code",
        "events": _log_events(path, session_dir),
    }
    return {
        "session": {"id": path.stem, "title": title, "created_at": _started_at(path)},
        "agent": {"name": "claude-code"},
        "threads": [primary, *_subagent_threads(session_dir)],
    }
