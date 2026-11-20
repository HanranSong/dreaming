"""What the readers met in a session log that this version of dreaming does not know.

Claude Code and Codex change their log formats without saying so. A reader skips what it does not recognize,
so a changed format would leave something out of a transcript, or misread it, in silence. Each such thing is
noted here instead, and a run that has any note stops before it analyzes a session, unless it was told to
carry on.
"""

from pathlib import Path

NOTICE = (
    "Log format: the session logs hold something this version of dreaming does not know. Claude Code or Codex "
    "may have changed its log format, so a transcript may be missing something or misreading it:"
)
IGNORE_FLAG = "--ignore-log-format"
STOPPED = (
    "Stopped: no session was analyzed and memory is as it was. Update dreaming, as its README says under "
    "Update, and run again. If it still stops, report it at https://github.com/HanranSong/dreaming/issues. "
    f"To analyze the sessions anyway, read as far as this version can read them, run again with {IGNORE_FLAG}."
)
CARRIED_ON = f"Carried on, as {IGNORE_FLAG} says. Check what these sessions added to memory."
LOGS_NAMED = 3  # of the logs a thing was met in, how many a note names

_notes: dict[str, dict[str, None]] = {}  # what was met -> the logs it was met in, in the order they were read


def note(tool: str, what: str, log: Path) -> None:
    """Note that a reader met something it does not know. The same thing in the same log is noted once."""
    _notes.setdefault(f"{tool}: {what}", {})[log.name] = None


def notes() -> list[str]:
    """One line for each kind of thing noted in this run, with the logs it was met in."""
    lines = []
    for what, logs in _notes.items():
        names = list(logs)
        more = f" and {len(names) - LOGS_NAMED} more" if len(names) > LOGS_NAMED else ""
        lines.append(f"{what} (in {', '.join(names[:LOGS_NAMED])}{more})")
    return lines


def clear() -> None:
    _notes.clear()
