"""Run dreaming every day at a set time from a line in the user's crontab, as the official cookbook does.

One dreaming folder has at most one schedule. It runs the same program a run by hand does, from the project's
folder, and adds what that prints to a log. When something it needs is missing, it says what and how to add it.
"""

import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

DEFAULT_TIME = "01:00"  # when the official cookbook's crontab line runs its nightly script

INSTALL_CRON = (
    "Install it, for example on Debian or Ubuntu with: sudo apt install cron. Elsewhere, install your system's "
    "cron package, such as cronie, and start its service. Then run this again."
)
START_CRON = (
    "Warning: no cron service seems to be running, so nothing runs at that time until one does. On Debian or "
    "Ubuntu, start it with: sudo systemctl enable --now cron (or, where systemd is not running, as on some WSL "
    "setups: sudo service cron start)\n"
)


class Schedule(NamedTuple):
    time: str  # HH:MM, in the machine's local time
    project: str


def _clock(at: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", at)
    if not match:
        raise SystemExit(f"--at takes a time of day as HH:MM, from 00:00 to 23:59, not {at!r}")
    return int(match[1]), int(match[2])


def _program() -> Path:
    """The dreaming command of the environment this run is in, so the schedule needs neither uv nor a PATH."""
    program = Path(sys.executable).parent / "dreaming"
    if not program.is_file():
        raise SystemExit(
            f"Cannot schedule: there is no dreaming command at {program}. "
            "Run `uv sync --project dreaming` from the project's folder, then try again."
        )
    return program


def _cron_running() -> bool:
    """Whether a cron service is running. Only Linux shows it; elsewhere, as on macOS, cron starts on demand."""
    processes = Path("/proc")
    if not sys.platform.startswith("linux") or not processes.is_dir():
        return True
    names = set()
    for name in processes.glob("[0-9]*/comm"):
        try:
            names.add(name.read_text().strip())
        except OSError:
            continue  # a process that ended meanwhile
    return bool(names & {"cron", "crond", "fcron"})


def _call(*command: str, text: str | None = None) -> None:
    done = subprocess.run(command, input=text, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"{' '.join(command)} failed: {(done.stderr or done.stdout).strip()}")


def _cron_text(text: str) -> str:
    return text.replace("%", r"\%")  # cron reads a bare % as the end of the command


def _marker(dreaming_dir: Path) -> str:
    """The comment this dreaming folder's line ends with, by which it is found again."""
    return _cron_text(f" # dreaming: {dreaming_dir}")


def _crontab() -> list[str]:
    listed = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if listed.returncode == 0:
        return listed.stdout.splitlines()
    if "no crontab" in listed.stderr:
        return []
    # Anything else must stop here: carrying on would replace a crontab that could not be read.
    raise SystemExit(f"crontab -l failed: {listed.stderr.strip()}")


def _write_crontab(lines: list[str]) -> None:
    _call("crontab", "-", text="".join(f"{line}\n" for line in lines))


def status(dreaming_dir: Path) -> Schedule | None:
    if not shutil.which("crontab"):
        return None
    for line in _crontab():
        if line.endswith(_marker(dreaming_dir)):
            words = shlex.split(line.replace(r"\%", "%"))  # minute hour * * * cd <project> && ...
            return Schedule(f"{int(words[1]):02}:{int(words[0]):02}", words[6])
    return None


def turn_on(dreaming_dir: Path, project: Path, log: Path, at: str | None) -> str:
    """Schedule a daily run over project, replacing this dreaming folder's schedule if it has one."""
    hour, minute = _clock(at or DEFAULT_TIME)
    if not shutil.which("crontab"):
        raise SystemExit(f"Cannot schedule: cron is not installed. {INSTALL_CRON}")
    program = _program()
    log.parent.mkdir(parents=True, exist_ok=True)
    project_, program_, log_ = (shlex.quote(str(path)) for path in (project, program, log))
    # date goes first so that each run's lines in the log start with when it ran.
    command = f"cd {project_} && {{ date; {program_}; }} >> {log_} 2>&1"
    ours = f"{minute} {hour} * * * {_cron_text(command)}{_marker(dreaming_dir)}"
    _write_crontab([*(line for line in _crontab() if not line.endswith(_marker(dreaming_dir))), ours])
    return (
        f"Schedule on: dreaming runs over {project} every day at {hour:02}:{minute:02}, this machine's time.\n"
        f"{'' if _cron_running() else START_CRON}"
        f"Each run adds what it prints to {log}"
    )


def turn_off(dreaming_dir: Path) -> str:
    if not status(dreaming_dir):
        return "Schedule off: there was none."
    _write_crontab([line for line in _crontab() if not line.endswith(_marker(dreaming_dir))])
    return "Schedule off."


def describe(dreaming_dir: Path, log: Path) -> str:
    found = status(dreaming_dir)
    if not found:
        return "Schedule off."
    return (
        f"Schedule on: dreaming runs over {found.project} every day at {found.time}, this machine's time.\n"
        f"{'' if _cron_running() else START_CRON}"
        f"Log: {log}"
    )
