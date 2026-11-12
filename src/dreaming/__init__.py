"""Dream over this project's Claude Code and Codex sessions into a local memory directory."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from dotenv import load_dotenv

from dreaming import codex
from dreaming import sessions as claude_code
from dreaming.agent import MEMORY_PATH, UPLOADS_PATH, AgentRun, run_agent
from dreaming.render import render_text
from dreaming.report import usd, write_report
from dreaming.sessions import without_times

DREAMING_DIR = Path(__file__).resolve().parents[2]
load_dotenv(DREAMING_DIR / ".env")  # values already set in the environment win


def _whole_number(name: str, low: int, high: float = float("inf")) -> int | None:
    """A whole-number setting, or None when empty. A bad value stops the program before anything is spent."""
    text = os.getenv(name)
    if not text:
        return None
    if not (text.isdecimal() and low <= int(text) <= high):
        limits = f"from {low} to {high}" if high != float("inf") else f"of at least {low}"
        raise SystemExit(f"{name} must be a whole number {limits}, not {text!r}")
    return int(text)


# The defaults are what the official cookbook's agents ran with. It names no effort; its platform ran them at medium.
ANALYZER_MODEL = os.getenv("DREAMING_ANALYZER_MODEL") or "claude-haiku-5-5"
ORGANIZER_MODEL = os.getenv("DREAMING_ORGANIZER_MODEL") or "claude-haiku-5-5"
ANALYZER_EFFORT = os.getenv("DREAMING_ANALYZER_EFFORT") or "medium"
ORGANIZER_EFFORT = os.getenv("DREAMING_ORGANIZER_EFFORT") or "medium"
# An empty context window uses the model's own. The percent stays above 10 because compaction keeps the latest 10%.
ANALYZER_CONTEXT_WINDOW = _whole_number("DREAMING_ANALYZER_CONTEXT_WINDOW", 1000)
ORGANIZER_CONTEXT_WINDOW = _whole_number("DREAMING_ORGANIZER_CONTEXT_WINDOW", 1000)
COMPACT_AT_PERCENT = _whole_number("DREAMING_COMPACT_AT_PERCENT", 11, 100) or 85

# Neither tool's logs have a "finished" marker, so a session counts as finished once its log goes quiet.
SESSION_QUIET_SECONDS = 30 * 60
# An agent that keeps failing after reaching the model is not retried forever: every try costs money.
MAX_ATTEMPTS = 3
# Rolling back only ever needs a recent copy of memory. Reports and tool logs are kept for every run.
BACKUPS_TO_KEEP = 5

MEMORY_DIR = DREAMING_DIR / "memory"
STATE_DIR = DREAMING_DIR / ".state"
TRANSCRIPT_DIR = STATE_DIR / "transcripts"
RUNS_DIR = STATE_DIR / "runs"
STATE_FILE = STATE_DIR / "sessions.json"
LOCK_FILE = STATE_DIR / "lock"
LEGACY_PROCESSED_FILE = STATE_DIR / "processed.json"  # the first version's list of analyzed sessions
PROMPT_DIR = Path(__file__).parent / "prompts"

# The one-line file the organizer leaves behind when it merges or renames a file.
STUB = re.compile(r"MOVED TO \[\[([^\]\n]+)\]\]\.?")

ANALYZER_TASK = (
    "Please read the transcript at {path} and add the durable, verified facts it shows to the long-term "
    "memory directory at {memory}. Check what is already there before writing and add to or update existing "
    "files rather than overwrite them. Write only inside that directory, and do not modify the transcript. "
    "When done, submit a short summary of what you added."
)
ORGANIZER_TASK = "Every analyzer has finished. Organize the memory directory at {memory}."

INTERRUPTED = "interrupted: the run stopped before this session finished"


class Session(NamedTuple):
    """One session's log and the module that reads it: sessions.py for Claude Code's, codex.py for Codex's."""

    id: str
    path: Path
    reader: ModuleType


def _project_sessions(project: Path) -> list[Session]:
    """The sessions both tools ran in this project as one timeline, earliest session first."""
    found = [
        Session(reader.session_id(path), path, reader)
        for reader in (claude_code, codex)
        for path in reader.session_files(project)
    ]
    return sorted(found, key=lambda session: session.reader.started_at(session.path))


def _prompt(name: str) -> str:
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return text.replace("{mem_dir}", MEMORY_PATH).replace("{run_preferences}", "")


def _load_state() -> dict:
    """Each session's status (running, done, failed or skipped) and whether memory still needs organizing."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        for entry in state["sessions"].values():
            entry.pop("log_size", None)  # earlier versions kept the log's size where log_signature is now
        return state
    state = {"sessions": {}, "organizer": {"pending": False, "attempts": 0}}
    if LEGACY_PROCESSED_FILE.exists():
        for session_id in json.loads(LEGACY_PROCESSED_FILE.read_text()):
            state["sessions"][session_id] = {"status": "done", "attempts": 1}
    return state


def _save_state(state: dict) -> None:
    # Written to a temporary file first, so a run killed mid-write cannot leave a broken state file.
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(STATE_FILE)
    LEGACY_PROCESSED_FILE.unlink(missing_ok=True)


def _retry_note(attempts: int) -> str:
    if attempts < MAX_ATTEMPTS:
        return f"Retried on the next run ({attempts} of {MAX_ATTEMPTS} attempts used)."
    return (
        f"Not retried again: {MAX_ATTEMPTS} of {MAX_ATTEMPTS} attempts used. "
        f"To retry, lower its attempts in {STATE_FILE}."
    )


def _files(directory: Path) -> dict[str, bytes]:
    paths = (path for path in directory.rglob("*") if path.is_file())
    return {str(path.relative_to(directory)): path.read_bytes() for path in paths}


def _memory_changes(before: Path) -> dict[str, list[str]]:
    old, new = _files(before), _files(MEMORY_DIR)
    return {
        "added": sorted(new.keys() - old.keys()),
        "changed": sorted(name for name in new.keys() & old.keys() if new[name] != old[name]),
        "removed": sorted(old.keys() - new.keys()),
    }


def _remove_stubs() -> int:
    stubs = [
        path
        for path in MEMORY_DIR.rglob("*.md")
        if STUB.fullmatch(path.read_text(encoding="utf-8").strip())
    ]
    for path in stubs:
        path.unlink()
    return len(stubs)


def _prune_backups() -> None:
    # Run folders are named by start time, so sorting by path puts the oldest first.
    for backup in sorted(RUNS_DIR.glob("*/memory_before"))[:-BACKUPS_TO_KEEP]:
        shutil.rmtree(backup)


def _render(session: Session) -> tuple[str, str | None, str | None]:
    """A session's title, transcript text and digest. The last two are None when it has no conversation.

    The digest leaves the transcript's time lines out. It tells whether the conversation changed, and it stays
    what it was for sessions analyzed before transcripts had those lines.
    """
    export = session.reader.load_session(session.path)
    title = export["session"]["title"]
    if not any(thread["events"] for thread in export["threads"]):
        return title, None, None
    untimed = render_text(without_times(export))
    return title, render_text(export), hashlib.sha256(untimed.encode(errors="replace")).hexdigest()


def _gained_content(session: Session, entry: dict, signature: str) -> bool:
    """Whether a finished session's conversation has changed since it was analyzed. signature is log_stat's."""
    if entry.get("log_signature") == signature:  # no log file was touched, so the transcript is the same
        return False
    try:
        digest = _render(session)[2]
    except Exception:
        return False
    changed = "transcript_sha256" in entry and digest != entry["transcript_sha256"]
    if not changed:  # only metadata changed, or the session predates these fields: this is the new baseline
        entry.update(log_signature=signature, transcript_sha256=digest)
    return changed


def _analyze(session: Session, run_dir: Path) -> tuple[str, str | None, AgentRun | None]:
    """Analyze one session. Returns its title, transcript digest and the agent's run (None if no conversation)."""
    try:
        title, text, digest = _render(session)
        if text is None:
            return title, None, None
        (TRANSCRIPT_DIR / f"{session.id}.txt").write_text(text, encoding="utf-8")
    except Exception as error:
        return "", None, AgentRun(error=f"could not read the session log: {error!r}")
    task = ANALYZER_TASK.format(path=f"{UPLOADS_PATH}/{session.id}.txt", memory=MEMORY_PATH)
    system = _prompt("analyzer_system.md")
    label = f"analyzer {session.id}"
    run = run_agent(
        ANALYZER_MODEL,
        ANALYZER_EFFORT,
        system,
        task,
        TRANSCRIPT_DIR,
        MEMORY_DIR,
        run_dir,
        label,
        ANALYZER_CONTEXT_WINDOW,
        COMPACT_AT_PERCENT,
        _prompt("analyzer_compaction.md"),
    )
    return title, digest, run


def main() -> None:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    # Two runs at once would analyze the same sessions and overwrite each other's memory and state. The system
    # drops this lock when the process ends, however it ends, so a killed run never leaves it behind.
    with LOCK_FILE.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another dreaming run is still working in {DREAMING_DIR}. Run again when it ends.")
        _run()


def _run() -> None:
    state = _load_state()
    sessions, organizer = state["sessions"], state["organizer"]

    # A session still marked running means the previous run was killed while analyzing it.
    for entry in sessions.values():
        if entry["status"] == "running":
            entry.update(status="failed", attempts=entry["attempts"] + 1, error=INTERRUPTED)
            organizer.update(pending=True, attempts=0)
    if organizer.pop("running", False):
        organizer["attempts"] += 1
        organizer["error"] = INTERRUPTED

    cutoff = time.time() - SESSION_QUIET_SECONDS
    pending, not_run, resumed = [], [], set()
    abandoned = False  # a session out of attempts is work left undone, so the run must not exit as a success
    for session in _project_sessions(Path.cwd()):
        entry = sessions.get(session.id, {})
        # Both cover the subagent logs and saved tool outputs too: a transcript is built from all of them.
        signature, changed_at = session.reader.log_stat(session.path)
        finished = entry.get("status") in ("done", "skipped")
        gained = finished and _gained_content(session, entry, signature)
        if finished and not gained:
            continue
        if not gained and entry.get("attempts", 0) >= MAX_ATTEMPTS:
            reason = f"gave up after {MAX_ATTEMPTS} failed attempts; last error: {entry.get('error')}"
            not_run.append((session.id, reason))
            abandoned = True
        elif changed_at >= cutoff:
            not_run.append((session.id, "waiting: its log changed in the last 30 minutes"))
        else:
            pending.append(session)
            if gained:  # new content gets a fresh set of attempts
                resumed.add(session.id)
                entry["attempts"] = 0
    _save_state(state)
    for session_id, reason in not_run:
        print(f"Not run {session_id}: {reason}")

    organizer_gave_up = organizer["pending"] and organizer["attempts"] >= MAX_ATTEMPTS
    if not pending and (not organizer["pending"] or organizer_gave_up):
        if organizer_gave_up:
            print(f"Organizer gave up after {MAX_ATTEMPTS} failed attempts; last error: {organizer.get('error')}")
        print("Nothing to do.")
        if abandoned or organizer_gave_up:
            raise SystemExit(1)
        return

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    shutil.copytree(MEMORY_DIR, run_dir / "memory_before")
    record = {
        "id": run_id,
        "project": str(Path.cwd()),
        "models": {"analyzer": (ANALYZER_MODEL, ANALYZER_EFFORT), "organizer": (ORGANIZER_MODEL, ORGANIZER_EFFORT)},
        "compaction": {
            "percent": COMPACT_AT_PERCENT,
            "analyzer": ANALYZER_CONTEXT_WINDOW,
            "organizer": ORGANIZER_CONTEXT_WINDOW,
        },
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished": None,
        "sessions": [],
        "not_run": not_run,
        "organizer": {"status": "not needed"},
        "agents": [],  # (label, AgentRun) for the usage table
        "memory_changes": {},
    }

    def checkpoint() -> None:
        _save_state(state)
        record["memory_changes"] = _memory_changes(run_dir / "memory_before")
        write_report(run_dir / "report.md", record)

    print(f"Run {run_id}: {len(pending)} session(s) to analyze.")
    for number, session in enumerate(pending, 1):
        source = session.reader.SOURCE
        entry = sessions.setdefault(session.id, {"attempts": 0})
        result = {"id": session.id, "source": source, "status": "running", "previous_error": entry.pop("error", None)}
        if session.id in resumed:
            result["note"] = "Analyzed again: the conversation changed after its last analysis."
        record["sessions"].append(result)
        entry.update(status="running", last_run=run_id, source=source)
        checkpoint()
        print(f"[{number}/{len(pending)}] {source} {session.id}: ", end="", flush=True)

        memory_before = _files(MEMORY_DIR)
        # Taken before reading, so anything appended meanwhile counts as new.
        signature = session.reader.log_stat(session.path)[0]
        result["title"], digest, run = _analyze(session, run_dir)
        if run is None:
            entry["status"] = "skipped"
            result["note"] = "No conversation in this session."
            print("skipped (no conversation)")
        else:
            record["agents"].append((f"analyzer {session.id[:8]}", run))
            entry["status"] = "failed" if run.error else "done"
            if run.usage["requests"]:  # a try that reached the model cost money, so it counts
                entry["attempts"] += 1
            # The organizer follows every analyzed session, and a failed one that still changed memory.
            if not run.error or _files(MEMORY_DIR) != memory_before:
                organizer.update(pending=True, attempts=0)
            if run.error:
                entry["error"] = run.error
                result.update(error=run.error, note=_retry_note(entry["attempts"]))
                print(f"failed: {run.error}")
            else:
                result["summary"] = run.text
                print(f"done ({run.usage['requests']} requests, {usd(run.cost)})")
        if entry["status"] != "failed":
            entry.update(log_signature=signature, transcript_sha256=digest)
        result["status"] = entry["status"]
        checkpoint()

    if organizer["pending"] and organizer["attempts"] < MAX_ATTEMPTS:
        record["organizer"] = {"status": "running"}
        organizer["running"] = True  # still set on the next start means this run was killed here
        checkpoint()
        print("Organizer: ", end="", flush=True)
        task = ORGANIZER_TASK.format(memory=MEMORY_PATH)
        system = _prompt("organizer_system.md")
        run = run_agent(
            ORGANIZER_MODEL,
            ORGANIZER_EFFORT,
            system,
            task,
            TRANSCRIPT_DIR,
            MEMORY_DIR,
            run_dir,
            "organizer",
            ORGANIZER_CONTEXT_WINDOW,
            COMPACT_AT_PERCENT,
        )
        record["agents"].append(("organizer", run))
        del organizer["running"]
        if run.error:
            organizer["attempts"] += bool(run.usage["requests"])
            organizer["error"] = run.error
            note = _retry_note(organizer["attempts"])
            record["organizer"] = {"status": "failed", "error": run.error, "note": note}
            print(f"failed: {run.error}")
        else:
            organizer.pop("error", None)
            organizer.update(pending=False, attempts=0)
            note = f"Removed {_remove_stubs()} stub file(s)."
            record["organizer"] = {"status": "done", "result": run.text, "note": note}
            print(f"done ({run.usage['requests']} requests, {usd(run.cost)})")
    elif organizer["pending"]:
        record["organizer"] = {"status": "gave up", "error": organizer.get("error")}

    record["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    checkpoint()
    _prune_backups()
    print(f"Report: {run_dir / 'report.md'}")
    statuses = [record["organizer"]["status"], *(session["status"] for session in record["sessions"])]
    if abandoned or "failed" in statuses or "gave up" in statuses:
        raise SystemExit(1)
