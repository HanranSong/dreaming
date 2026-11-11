"""Dream over this project's Claude Code sessions into a local memory directory."""

import hashlib
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from dreaming.agent import MEMORY_PATH, UPLOADS_PATH, AgentRun, run_agent
from dreaming.render import render_text
from dreaming.report import usd, write_report
from dreaming.sessions import load_session, session_files

DREAMING_DIR = Path(__file__).resolve().parents[2]
load_dotenv(DREAMING_DIR / ".env")  # values already set in the environment win

# The default models are the official cookbook's. An empty effort leaves it to the API's default.
ANALYZER_MODEL = os.getenv("DREAMING_ANALYZER_MODEL") or "claude-haiku-5-5"
ORGANIZER_MODEL = os.getenv("DREAMING_ORGANIZER_MODEL") or "claude-haiku-5-5"
ANALYZER_EFFORT = os.getenv("DREAMING_ANALYZER_EFFORT") or None
ORGANIZER_EFFORT = os.getenv("DREAMING_ORGANIZER_EFFORT") or None

# Claude Code logs have no "finished" marker, so a session counts as finished once its log goes quiet.
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


def _prompt(name: str) -> str:
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return text.replace("{mem_dir}", MEMORY_PATH).replace("{run_preferences}", "")


def _load_state() -> dict:
    """Each session's status (running, done, failed or skipped) and whether memory still needs organizing."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
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


def _render(path: Path) -> tuple[str, str | None]:
    """A session's title and transcript text. The text is None when the session has no conversation."""
    export = load_session(path)
    title = export["session"]["title"]
    if not any(thread["events"] for thread in export["threads"]):
        return title, None
    return title, render_text(export)


def _digest(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode(errors="replace")).hexdigest()


def _gained_content(path: Path, entry: dict) -> bool:
    """Whether a finished session's conversation has changed since it was analyzed."""
    size = path.stat().st_size
    if entry.get("log_size") == size:
        return False
    try:
        digest = _digest(_render(path)[1])
    except Exception:
        return False
    changed = "transcript_sha256" in entry and digest != entry["transcript_sha256"]
    if not changed:  # only metadata grew, or the session predates these fields: this is the new baseline
        entry.update(log_size=size, transcript_sha256=digest)
    return changed


def _analyze(path: Path, tool_log: Path) -> tuple[str, str | None, AgentRun | None]:
    """Analyze one session. Returns its title, transcript digest and the agent's run (None if no conversation)."""
    try:
        title, text = _render(path)
        if text is None:
            return title, None, None
        (TRANSCRIPT_DIR / f"{path.stem}.txt").write_text(text, encoding="utf-8")
    except Exception as error:
        return "", None, AgentRun(error=f"could not read the session log: {error!r}")
    task = ANALYZER_TASK.format(path=f"{UPLOADS_PATH}/{path.stem}.txt", memory=MEMORY_PATH)
    system = _prompt("analyzer_system.md")
    label = f"analyzer {path.stem}"
    run = run_agent(ANALYZER_MODEL, ANALYZER_EFFORT, system, task, TRANSCRIPT_DIR, MEMORY_DIR, tool_log, label)
    return title, _digest(text), run


def main() -> None:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
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
    for path in session_files(Path.cwd()):
        entry = sessions.get(path.stem, {})
        finished = entry.get("status") in ("done", "skipped")
        gained = finished and _gained_content(path, entry)
        if finished and not gained:
            continue
        if not gained and entry.get("attempts", 0) >= MAX_ATTEMPTS:
            reason = f"gave up after {MAX_ATTEMPTS} failed attempts; last error: {entry.get('error')}"
            not_run.append((path.stem, reason))
        elif path.stat().st_mtime >= cutoff:
            not_run.append((path.stem, "waiting: its log changed in the last 30 minutes"))
        else:
            pending.append(path)
            if gained:  # new content gets a fresh set of attempts
                resumed.add(path.stem)
                entry["attempts"] = 0
    _save_state(state)
    for session_id, reason in not_run:
        print(f"Not run {session_id}: {reason}")

    organizer_gave_up = organizer["pending"] and organizer["attempts"] >= MAX_ATTEMPTS
    if not pending and (not organizer["pending"] or organizer_gave_up):
        if organizer_gave_up:
            print(f"Organizer gave up after {MAX_ATTEMPTS} failed attempts; last error: {organizer.get('error')}")
        print("Nothing to do.")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    shutil.copytree(MEMORY_DIR, run_dir / "memory_before")
    tool_log = run_dir / "tool_calls.jsonl"
    record = {
        "id": run_id,
        "project": str(Path.cwd()),
        "models": {"analyzer": (ANALYZER_MODEL, ANALYZER_EFFORT), "organizer": (ORGANIZER_MODEL, ORGANIZER_EFFORT)},
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
    for number, path in enumerate(pending, 1):
        entry = sessions.setdefault(path.stem, {"attempts": 0})
        result = {"id": path.stem, "status": "running", "previous_error": entry.pop("error", None)}
        if path.stem in resumed:
            result["note"] = "Analyzed again: the conversation changed after its last analysis."
        record["sessions"].append(result)
        entry.update(status="running", last_run=run_id)
        checkpoint()
        print(f"[{number}/{len(pending)}] {path.stem}: ", end="", flush=True)

        memory_before = _files(MEMORY_DIR)
        log_size = path.stat().st_size  # taken before reading, so anything appended meanwhile counts as new
        result["title"], digest, run = _analyze(path, tool_log)
        if run is None:
            entry["status"] = "skipped"
            result["note"] = "No conversation in this session."
            print("skipped (no conversation)")
        else:
            record["agents"].append((f"analyzer {path.stem[:8]}", run))
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
            entry.update(log_size=log_size, transcript_sha256=digest)
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
            ORGANIZER_MODEL, ORGANIZER_EFFORT, system, task, TRANSCRIPT_DIR, MEMORY_DIR, tool_log, "organizer"
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
    if "failed" in statuses:
        raise SystemExit(1)
