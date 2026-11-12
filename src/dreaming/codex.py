"""Read Codex session logs into the export shape render.py expects."""

import json
import os
import re
import sys
from compression import zstd
from functools import cache
from pathlib import Path

from dreaming.sessions import TIME_EVENT, files_stat

SOURCE = "codex"

# Threads Codex starts for itself rather than for the project's work: the reviews behind its automatic
# approvals, its compaction and its own memory. Each has a log like any other thread.
INTERNAL_THREADS = {"guardian", "guardian_review", "compact", "memory_consolidation"}
# Records with no conversation in them: settings, token counts, the history a compaction left the model with,
# and event_msg, which repeats the response_item records for Codex's own interface.
SKIPPED_RECORDS = {
    "session_meta",
    "turn_context",
    "world_state",
    "event_msg",
    "token_usage_record",
    "compacted",
    "dynamic_tools_updated",
    "security_risk_score",
    "inter_agent_communication_metadata",
}
# Response items left out: reasoning is encrypted, and additional_tools lists the tools the model was given.
SKIPPED_ITEMS = {"reasoning", "additional_tools"}
# Fields of a response item that say nothing about what happened.
BOOKKEEPING = {
    "type",
    "id",
    "call_id",
    "name",
    "status",
    "encrypted_content",
    "internal_chat_message_metadata_passthrough",
}
# Text Codex puts in the user's turn itself: one of its own tags around the whole block. Only the tags named
# here count, so a request the user wrote inside a tag of their own is kept. A tag Codex adds later shows up
# in transcripts as a user message until it is named here.
INJECTED = re.compile(
    r"<(environment_context|user_instructions|user_shell_command|turn_aborted|subagent_notification|skill"
    r"|external_\w+)>.*</\1>",
    re.DOTALL,
)


def _home() -> Path:
    return Path(os.getenv("CODEX_HOME") or Path.home() / ".codex")


def _records(path: Path):
    # Codex can keep a log compressed, as rollout-....jsonl.zst.
    opened = zstd.open(path, "rt", encoding="utf-8") if path.suffix == ".zst" else path.open(encoding="utf-8")
    with opened as lines:
        for line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a session still being written can end in a partial line


def _subagent(meta: dict):
    """What kind of sub-agent a session_meta is for, if any: a name such as "compact", {"other": "guardian"},
    or {"thread_spawn": {...}} for one the model started."""
    source = meta.get("source")
    return source.get("subagent") if isinstance(source, dict) else None


def _spawn(meta: dict) -> dict:
    """What a sub-agent's session_meta says of its spawning: its parent thread, name and role."""
    kind = _subagent(meta)
    spawn = kind.get("thread_spawn") if isinstance(kind, dict) else None
    return spawn if isinstance(spawn, dict) else {}


def _internal(meta: dict) -> bool:
    kind = _subagent(meta)
    if isinstance(kind, dict):
        kind = kind.get("other")
    return bool({kind, meta.get("thread_source")} & INTERNAL_THREADS)


@cache
def _threads() -> dict[Path, dict]:
    """Every thread Codex keeps a log for, internal ones aside: its log and session_meta, earliest thread first.

    Codex files its logs by date, not by project, and gives a sub-agent a log of its own beside its parent's,
    so finding a project's sessions and a session's sub-agents both take every log's first record. They are
    read once per run.
    """
    home = _home()
    logs = [*home.glob("sessions/**/rollout-*"), *home.glob("archived_sessions/**/rollout-*")]
    threads = {}
    for log in sorted(logs, key=lambda log: log.stat().st_mtime):
        if log.suffix not in (".jsonl", ".zst"):
            continue
        try:
            first = next(_records(log), None)
        except Exception as error:  # one broken log, of any project, must not hide every other session
            print(f"Left out a Codex log that could not be read: {log}: {error!r}", file=sys.stderr)
            continue
        meta = first.get("payload") if isinstance(first, dict) and first.get("type") == "session_meta" else None
        # The id names the transcript file, so only a plain one is taken.
        if isinstance(meta, dict) and re.fullmatch(r"[\w-]+", str(meta.get("id"))) and not _internal(meta):
            # A thread with two logs, such as a live one and an archived one, is read from the newer.
            threads = {other: has for other, has in threads.items() if has["id"] != meta["id"]}
            threads[log] = meta
    return dict(sorted(threads.items(), key=lambda item: item[1].get("timestamp") or ""))


def _parent(meta: dict) -> str | None:
    return meta.get("parent_thread_id") or _spawn(meta).get("parent_thread_id")


def _sub_threads(path: Path) -> dict[Path, dict]:
    """The sub-agents under a session, at any depth, earliest first."""
    ids = {_threads()[path]["id"]}
    while True:  # each pass reaches one level further down, until a pass finds no new thread
        found = {log: meta for log, meta in _threads().items() if _parent(meta) in ids}
        reached = ids | {meta["id"] for meta in found.values()}
        if reached == ids:
            return found
        ids = reached


def session_files(project_root: Path) -> list[Path]:
    """The logs of the sessions Codex ran in this project. A sub-agent's log belongs to its parent's session."""
    return [
        log for log, meta in _threads().items() if not _parent(meta) and meta.get("cwd") == str(project_root)
    ]


def session_id(path: Path) -> str:
    return _threads()[path]["id"]


def started_at(path: Path) -> str:
    """When the session was created: a UTC ISO timestamp, as in Claude Code's logs."""
    return _threads()[path].get("timestamp") or ""


def log_stat(path: Path) -> tuple[str, float]:
    """A signature of the session's log and its sub-agents' logs, and when the newest of them changed."""
    return files_stat([path, *_sub_threads(path)], _home())


def _blocks(content, *, from_user: bool = False):
    """A message's or tool result's content with its text as the text blocks render_text reads. Images are
    dropped, and from the user's turn so is the text Codex injected. Codex encrypts the task one agent gives
    another, so all that can be said of such a block is that it was there."""
    if not isinstance(content, list):
        return content
    blocks = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind in ("input_text", "output_text", "text") and isinstance(block.get("text"), str):
            if not (from_user and INJECTED.fullmatch(block["text"].strip())):
                blocks.append({"type": "text", "text": block["text"]})
        elif kind == "encrypted_content":
            blocks.append({"type": "text", "text": "[encrypted by Codex]"})
        elif "image" not in str(kind):
            blocks.append(block)
    return blocks


def _call_input(item: dict):
    """What a tool was called with: its input, its JSON arguments, or whatever else the call holds."""
    if "input" in item:
        return item["input"]
    if isinstance(item.get("arguments"), str):
        try:
            return json.loads(item["arguments"])
        except json.JSONDecodeError:
            return item["arguments"]
    return {key: value for key, value in item.items() if key not in BOOKKEEPING}


def _events(item: dict, agent_path: str):
    """Turn one response item into transcript events. Developer messages are Codex's own instructions."""
    kind = str(item.get("type"))
    if kind == "message":
        role = item.get("role")
        content = _blocks(item.get("content"), from_user=role == "user")
        if role in ("user", "assistant") and content:
            yield {"type": "user.message" if role == "user" else "agent.message", "content": content}
    elif kind == "agent_message":  # between two agents of the session, each named by its path
        content = _blocks(item.get("content"))
        if item.get("author") == agent_path:
            yield {"type": "agent.thread_message_sent", "to_agent_name": item.get("recipient"), "content": content}
        else:
            yield {"type": "agent.thread_message_received", "from_agent_name": item.get("author"), "content": content}
    elif kind.endswith("_call_output"):
        yield {"type": "agent.tool_result", "tool_use_id": item.get("call_id"), "content": _blocks(item.get("output"))}
    elif kind.endswith("_call"):
        yield {
            "type": "agent.tool_use",
            "id": item.get("call_id"),
            "name": item.get("name") or kind.removesuffix("_call"),
            "input": _call_input(item),
        }
    elif kind not in SKIPPED_ITEMS:  # an item this reader does not know is kept, as an EVENT line
        yield {"type": f"codex.{kind}", **{key: value for key, value in item.items() if key not in BOOKKEEPING}}


def _agent_path(meta: dict) -> str | None:
    """The name a sub-agent goes by in the session, such as /root/read_constants. The primary agent is /root."""
    return meta.get("agent_path") or _spawn(meta).get("agent_path")


def _item_ids(path: Path) -> set[str]:
    return {
        record["payload"].get("id")
        for record in _records(path)
        if record.get("type") == "response_item" and isinstance(record.get("payload"), dict)
    }


def _log_events(path: Path) -> list[dict]:
    meta = _threads()[path]
    agent_path = _agent_path(meta) or "/root"
    # A sub-agent's log, like any forked thread's, opens with a copy of the thread it was forked from, each
    # item under the id it has there. Those items are that thread's, so they are left out of this one.
    forked_from = next((log for log, other in _threads().items() if other["id"] == meta.get("forked_from_id")), None)
    inherited = _item_ids(forked_from) - {None} if forked_from else set()
    events, timed = [], None
    for record in _records(path):
        kind, payload = record.get("type"), record.get("payload")
        if kind in SKIPPED_RECORDS or not isinstance(payload, dict) or payload.get("id") in inherited:
            continue
        if kind == "response_item":
            new = list(_events(payload, agent_path))
        else:  # a kind of record this reader does not know is kept, as an EVENT line
            new = [{"type": f"codex.{kind}", "payload": payload}]
        # As in Claude Code's logs: each user message gets its own time, to the second.
        time = (record.get("timestamp") or "")[:19]
        if time and time != timed and any(event["type"] == "user.message" for event in new):
            events.append({"type": TIME_EVENT, "utc": f"{time}Z"})
            timed = time
        events.extend(new)
    return events


def _title(session_id: str) -> str:
    """The name Codex gave the session. Its index gets a new line each time the name changes."""
    title, index = "", _home() / "session_index.jsonl"
    if index.is_file():
        for record in _records(index):
            if record.get("id") == session_id:
                title = record.get("thread_name") or title
    return title


def load_session(path: Path) -> dict:
    """One session log and its sub-agents' as the dict render_text takes, with a time event above user messages."""
    meta = _threads()[path]
    primary = {"id": "primary", "parent_thread_id": None, "agent_name": SOURCE, "events": _log_events(path)}
    sub_threads = [
        {
            "id": sub["id"],
            "parent_thread_id": "primary",
            "agent_name": _agent_path(sub) or "subagent",
            "events": _log_events(log),
        }
        for log, sub in _sub_threads(path).items()
    ]
    return {
        "session": {"id": meta["id"], "title": _title(meta["id"]), "created_at": started_at(path)},
        "agent": {"name": SOURCE},
        "threads": [primary, *sub_threads],
    }
