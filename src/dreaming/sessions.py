"""Read Claude Code session logs into the export shape render.py expects. codex.py does the same for Codex's."""

import hashlib
import json
import re
from itertools import islice
from pathlib import Path

from dreaming import formats

# The preview Claude Code leaves in the log when a tool result is too large to keep inline.
PERSISTED_OUTPUT = re.compile(
    r"<persisted-output>\s*[^\n]*Full output saved to: [^\n]*/tool-results/([^\s/]+)"
)
# The event that carries a user message's time. render_text has no label for it, so it prints the event as an
# "EVENT message.time:" line above the message, which is how the analyzer prompt describes it.
TIME_EVENT = "message.time"
# The event above a compaction summary in a log that lost messages it covers. render_text prints it
# as an "EVENT compaction.summary:" line, which the analyzer prompt describes, and the summary below it as an
# AGENT: block, since Claude Code's model wrote it.
SUMMARY_EVENT = "compaction.summary"

SOURCE = "claude-code"
TOOL = "Claude Code"  # as format notes name it

# Content blocks left out of a transcript on purpose: the model's thinking, as in the official export, and
# attached files.
LEFT_OUT_BLOCKS = {"thinking", "redacted_thinking", "image", "document"}
# Records that hold a message without being part of the conversation: Claude Code's copy of what it sent the API.
NOT_CONVERSATION = {"api-request-blob"}
# How a compaction summary opens. One not marked as a summary would read as something the user said.
SUMMARY_OPENING = "This session is being continued from a previous conversation"
# How many records into a log the folder it was started in is looked for, when looking through every project's.
RECORDS_TO_FOLDER = 50


def _records(path: Path):
    with path.open(encoding="utf-8") as lines:
        broken = False
        for line in lines:
            if not line.strip():
                continue
            if broken:  # a session still being written can end in a partial line, but only its last line
                formats.note(TOOL, "a line that is not JSON was skipped", path)
                broken = False
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                broken = True


def started_at(path: Path) -> str:
    """The log's first message time. These are UTC ISO timestamps, so they sort as strings."""
    for record in _records(path):
        if record.get("type") in ("user", "assistant"):
            return record.get("timestamp") or ""
    return ""


def _started_in(path: Path) -> str | None:
    """The folder the session was started in, if the log says."""
    return next((record["cwd"] for record in _records(path) if record.get("cwd")), None)


def session_files(project_root: Path) -> list[Path]:
    """The session logs Claude Code keeps for this project, earliest session first."""
    projects = Path.home() / ".claude" / "projects"
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project_root))
    logs = (projects / slug).glob("*.jsonl")
    # Two folders can share a slug, /a-b and /a/b for one, so each log is checked for the folder it names.
    found = sorted((log for log in logs if _started_in(log) in (None, str(project_root))), key=started_at)
    if not found:
        # Were Claude Code to name a project's folder another way, its sessions would never be found. So a
        # project with none is looked for in the other folders, by the folder each log names.
        for log in projects.glob("*/*.jsonl"):
            first = islice(_records(log), RECORDS_TO_FOLDER)
            started_in = next((record["cwd"] for record in first if record.get("cwd")), None)
            if log.parent.name != slug and started_in == str(project_root):
                where = f"projects/{log.parent.name}, not projects/{slug}"
                formats.note(TOOL, f"sessions started in this folder are kept under {where}, and were not read", log)
    return found


def session_id(path: Path) -> str:
    return path.stem


def files_stat(files: list[Path], folder: Path) -> tuple[str, float]:
    """A signature of the files a session's transcript is read from, and when the newest of them changed.

    The signature is a hash of each one's path under folder, size and modification time, so it changes when
    any of them is added, removed or written to, even if the sizes still add up to the same total.
    """
    files = sorted(files)
    stats = [file.stat() for file in files]
    lines = [
        f"{file.relative_to(folder)} {stat.st_size} {stat.st_mtime_ns}"
        for file, stat in zip(files, stats, strict=True)
    ]
    signature = hashlib.sha256("\n".join(lines).encode(errors="replace")).hexdigest()
    return signature, max(stat.st_mtime for stat in stats)


def log_stat(path: Path) -> tuple[str, float]:
    """files_stat of the session log, its subagents' logs and the tool outputs saved beside them."""
    session_dir = path.with_suffix("")
    files = [path, *session_dir.glob("subagents/**/agent-*"), *session_dir.glob("tool-results/*")]
    return files_stat(files, path.parent)


def _full_result(content, session_dir: Path, log: Path):
    """A tool result's complete output: the saved file replaces Claude Code's preview, and images are dropped."""
    if isinstance(content, str) and (match := PERSISTED_OUTPUT.match(content)):
        saved = session_dir / "tool-results" / match.group(1)
        if saved.is_file():
            return saved.read_text(encoding="utf-8", errors="replace")
    elif isinstance(content, str) and content.lstrip().startswith("<persisted-output>"):
        formats.note(TOOL, "a tool output saved to a file does not name its file, so only its preview was read", log)
    if isinstance(content, list):
        # Base64 images would otherwise be rendered into the transcript as text. A text block can hold a
        # preview too.
        return [
            {**block, "text": _full_result(block["text"], session_dir, log)}
            if isinstance(block.get("text"), str)
            else block
            for block in content
            if block.get("type") != "image"
        ]
    return content


def _events(record: dict, session_dir: Path, log: Path):
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
                "content": _full_result(block.get("content"), session_dir, log),
                "is_error": block.get("is_error"),
            }
        elif kind not in LEFT_OUT_BLOCKS:
            formats.note(TOOL, f'a content block of type "{kind}" was left out of the transcript', log)


def _preserved_by(records: list[dict], index: int) -> set:
    """The uuids of the messages that the compaction at records[index] kept as they were."""
    metadata = records[index].get("compactMetadata") or {}
    preserved = metadata.get("preservedMessages") or {}
    kept = {*(preserved.get("uuids") or []), *(preserved.get("allUuids") or [])}
    segment = metadata.get("preservedSegment") or {}
    # The segment runs from its tail back to its head along parent links. A message replayed later in the log
    # can carry a rewritten link, so each uuid's first record gives it; a walk that misses the head counts
    # for nothing.
    parents = {}
    for record in records[:index]:
        if record.get("uuid"):
            parents.setdefault(record["uuid"], record.get("parentUuid"))
    uuid, walked = segment.get("tailUuid"), set()
    while uuid and uuid not in walked:
        walked.add(uuid)
        if uuid == segment.get("headUuid"):
            return kept | walked
        uuid = parents.get(uuid)
    return kept


def _trimmed(records: list[dict]) -> bool:
    """Whether Claude Code deleted messages from this log, leaving its compaction summaries in their place.

    Its transcript GC deletes the messages before the last compaction except those that compaction preserved,
    and a Remote Control resume can rewrite the log from the server's copy, which starts near the last
    compaction. A log neither touched starts at the conversation's root, and before each compaction it holds
    messages that compaction did not preserve, at least the ones it summarized.

    Some links are not trusted. A compaction's logicalParentUuid can name a message that was never written to
    the log, and a later compaction or a resume can replay earlier messages, under their own uuids, after
    newer ones, so a parent can turn up further down the log and the records after a compaction can point to
    any message before it.
    """
    first = next((record for record in records if "parentUuid" in record), None)
    if first and first["parentUuid"]:
        return True  # the log starts part-way through the conversation
    for index, record in enumerate(records):
        if record.get("subtype") == "compact_boundary":
            preserved = _preserved_by(records, index)
            if not any(
                earlier.get("type") in ("user", "assistant")
                and not earlier.get("isCompactSummary")
                and earlier.get("uuid") not in preserved
                for earlier in records[:index]
            ):
                return True
    return False


def _log_events(path: Path, session_dir: Path) -> list[dict]:
    records = list(_records(path))
    # In a trimmed log every summary is kept. Each may be all that is left of what it summarized, and the GC
    # can keep an earlier summary while deleting the messages it summarized and those after it.
    trimmed = _trimmed(records)
    events, timed, untimed = [], None, False
    for record in records:
        if trimmed and record.get("isCompactSummary"):
            written = (record.get("timestamp") or "")[:19]
            events.append({"type": SUMMARY_EVENT, **({"utc": f"{written}Z"} if written else {})})
            events.extend(
                {"type": "agent.message", "content": event["content"]}
                for event in _events(record, session_dir, path)
                if event["type"] == "user.message"
            )
            continue
        # Text the user did not write: isMeta marks injected text such as skill bodies, and isCompactSummary
        # a model-written summary of earlier messages, which stay in the log unless it was trimmed.
        injected = record.get("isMeta") or record.get("isCompactSummary")
        if record.get("type") in ("user", "assistant") and not injected:
            new = list(_events(record, session_dir, path))
            if any(event["type"] == "user.message" and event["content"].startswith(SUMMARY_OPENING) for event in new):
                formats.note(TOOL, "a compaction summary is not marked as one, so it reads as a user message", path)
            untimed = untimed or not record.get("timestamp")
            # A session can run for days or be resumed later, so each user message gets its own time, to the
            # second. Records logged at the same moment share one.
            time = (record.get("timestamp") or "")[:19]
            if time and time != timed and any(event["type"] == "user.message" for event in new):
                events.append({"type": TIME_EVENT, "utc": f"{time}Z"})
                timed = time
            events.extend(new)
        elif record.get("type") not in ("user", "assistant", *NOT_CONVERSATION):
            message = record.get("message")
            if isinstance(message, dict) and message.get("role") in ("user", "assistant") and message.get("content"):
                kind = record.get("type")
                formats.note(TOOL, f'a record of type "{kind}" holds a message and was left out', path)
    if untimed:
        formats.note(TOOL, "a message has no timestamp, so the transcript does not say when it was sent", path)
    return events


def _subagent_threads(session_dir: Path) -> list[dict]:
    """Subagents log to their own files, under subagents/ directly or in a workflow's folder."""
    threads = []
    for log in sorted(session_dir.glob("subagents/**/agent-*.jsonl"), key=started_at):
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


def without_times(export: dict) -> dict:
    """A session's export with the time events taken out: what was said, whenever it was said."""
    threads = [
        {**thread, "events": [event for event in thread["events"] if event["type"] != TIME_EVENT]}
        for thread in export["threads"]
    ]
    return {**export, "threads": threads}


def load_session(path: Path) -> dict:
    """One session log and its subagents as the dict render_text takes, with a time event above user messages."""
    session_dir = path.with_suffix("")
    # A name the user gave the session comes before the one Claude Code generated from its first message, as
    # in Claude Code's own session list. Both are logged again as the session goes on, so the last one counts.
    custom_title = ai_title = ""
    for record in _records(path):
        if record.get("type") == "custom-title":
            custom_title = record.get("customTitle") or custom_title
        elif record.get("type") == "ai-title":
            ai_title = record.get("aiTitle") or ai_title
    primary = {
        "id": "primary",
        "parent_thread_id": None,
        "agent_name": SOURCE,
        "events": _log_events(path, session_dir),
    }
    return {
        # ai_title is not rendered. It is the title earlier versions read, which _gained_content needs.
        "session": {
            "id": path.stem,
            "title": custom_title or ai_title,
            "ai_title": ai_title,
            "created_at": started_at(path),
        },
        "agent": {"name": SOURCE},
        "threads": [primary, *_subagent_threads(session_dir)],
    }
