"""Dream over this project's Claude Code sessions into a local memory directory."""

import json
import re
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dreaming.agent import MEMORY_PATH, UPLOADS_PATH, run_agent
from dreaming.render import render_text
from dreaming.sessions import load_session, session_files

# The official cookbook's defaults.
ANALYZER_MODEL = "claude-haiku-5-5"
ORGANIZER_MODEL = "claude-haiku-5-5"

# Claude Code logs have no "finished" marker, so a session counts as finished once its log goes quiet.
SESSION_QUIET_SECONDS = 30 * 60

DREAMING_DIR = Path(__file__).resolve().parents[2]
MEMORY_DIR = DREAMING_DIR / "memory"
TRANSCRIPT_DIR = DREAMING_DIR / ".state" / "transcripts"
PROCESSED_FILE = DREAMING_DIR / ".state" / "processed.json"
BACKUP_DIR = DREAMING_DIR / ".state" / "backups"
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


def _prompt(name: str) -> str:
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return text.replace("{mem_dir}", MEMORY_PATH).replace("{run_preferences}", "")


def _remove_stubs() -> int:
    stubs = [
        path
        for path in MEMORY_DIR.rglob("*.md")
        if STUB.fullmatch(path.read_text(encoding="utf-8").strip())
    ]
    for path in stubs:
        path.unlink()
    return len(stubs)


def main() -> None:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    processed = set(json.loads(PROCESSED_FILE.read_text())) if PROCESSED_FILE.exists() else set()

    new = [path for path in session_files(Path.cwd()) if path.stem not in processed]
    cutoff = time.time() - SESSION_QUIET_SECONDS
    pending = [path for path in new if path.stat().st_mtime < cutoff]
    if len(pending) < len(new):
        print(f"Waiting on {len(new) - len(pending)} session(s) updated in the last 30 minutes.")
    if not pending:
        print("No new sessions to process.")
        return

    shutil.copytree(MEMORY_DIR, BACKUP_DIR / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    usage = Counter()
    analyzed = 0
    system = _prompt("analyzer_system.md")
    for path in pending:
        export = load_session(path)
        if not export["threads"][0]["events"]:
            print(f"Skipped {path.stem}: no conversation")
            continue
        (TRANSCRIPT_DIR / f"{path.stem}.txt").write_text(render_text(export), encoding="utf-8")
        task = ANALYZER_TASK.format(path=f"{UPLOADS_PATH}/{path.stem}.txt", memory=MEMORY_PATH)
        print(f"Analyzing {path.stem}...")
        try:
            summary, spent = run_agent(ANALYZER_MODEL, system, task, TRANSCRIPT_DIR, MEMORY_DIR)
        except Exception as error:  # one failed session should not stop the others; it is retried next run
            print(f"Failed {path.stem}: {error}")
            continue
        print(summary)
        usage += spent
        analyzed += 1
        processed.add(path.stem)
        PROCESSED_FILE.write_text(json.dumps(sorted(processed), indent=2))

    if analyzed:
        print("Organizing...")
        task = ORGANIZER_TASK.format(memory=MEMORY_PATH)
        result, spent = run_agent(
            ORGANIZER_MODEL, _prompt("organizer_system.md"), task, TRANSCRIPT_DIR, MEMORY_DIR
        )
        print(result)
        usage += spent
        print(f"Removed {_remove_stubs()} stub file(s).")

    print(f"Analyzed {analyzed} of {len(pending)} session(s). Tokens: {dict(usage)}")
