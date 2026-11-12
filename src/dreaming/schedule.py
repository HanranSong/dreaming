"""Run dreaming every day at a set time: a crontab line where cron is installed, a systemd user timer elsewhere.

One dreaming folder has at most one schedule. It runs the same program a run by hand does, from the project's
folder, and adds what that prints to a log.
"""

import getpass
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

DEFAULT_TIME = "01:00"  # when the official cookbook's crontab line runs its nightly script


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
        raise SystemExit(f"Cannot schedule: there is no dreaming command at {program}")
    return program


def _backend() -> str:
    if shutil.which("crontab"):
        return "cron"
    if shutil.which("systemctl"):
        return "systemd"
    raise SystemExit("Cannot schedule: this machine has neither crontab nor systemctl.")


def _call(*command: str, text: str | None = None) -> None:
    done = subprocess.run(command, input=text, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"{' '.join(command)} failed: {(done.stderr or done.stdout).strip()}")


# cron: one line in the user's crontab, found again by the comment it ends with.


def _cron_text(text: str) -> str:
    return text.replace("%", r"\%")  # cron reads a bare % as the end of the command


def _cron_marker(dreaming_dir: Path) -> str:
    return _cron_text(f" # dreaming: {dreaming_dir}")


def _crontab() -> list[str]:
    listed = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if listed.returncode == 0:
        return listed.stdout.splitlines()
    if "no crontab" in listed.stderr:
        return []
    # Anything else must stop here: carrying on would replace a crontab that could not be read.
    raise SystemExit(f"crontab -l failed: {listed.stderr.strip()}")


def _cron_off(dreaming_dir: Path) -> bool:
    lines = _crontab()
    kept = [line for line in lines if not line.endswith(_cron_marker(dreaming_dir))]
    if kept != lines:
        _call("crontab", "-", text="".join(f"{line}\n" for line in kept))
    return kept != lines


def _cron_on(dreaming_dir: Path, project: Path, program: Path, log: Path, hour: int, minute: int) -> None:
    project_, program_, log_ = (shlex.quote(str(path)) for path in (project, program, log))
    # date goes first so that each run's lines in the log start with when it ran.
    command = f"cd {project_} && {{ date; {program_}; }} >> {log_} 2>&1"
    ours = f"{minute} {hour} * * * {_cron_text(command)}{_cron_marker(dreaming_dir)}"
    others = [line for line in _crontab() if not line.endswith(_cron_marker(dreaming_dir))]
    _call("crontab", "-", text="".join(f"{line}\n" for line in [*others, ours]))


def _cron_status(dreaming_dir: Path) -> Schedule | None:
    for line in _crontab():
        if line.endswith(_cron_marker(dreaming_dir)):
            words = shlex.split(line.replace(r"\%", "%"))  # minute hour * * * cd <project> && ...
            return Schedule(f"{int(words[1]):02}:{int(words[0]):02}", words[6])
    return None


# systemd: a service and a timer in the user's own unit folder, named after the dreaming folder.


def _units(dreaming_dir: Path) -> tuple[Path, Path]:
    name = f"dreaming-{hashlib.sha256(str(dreaming_dir).encode()).hexdigest()[:8]}"
    folder = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user"
    return folder / f"{name}.service", folder / f"{name}.timer"


def _unit_text(text: str) -> str:
    return text.replace("%", "%%")  # systemd reads a bare % as the start of a specifier


def _unit_word(text: str) -> str:
    """One quoted word of an ExecStart line."""
    return '"' + _unit_text(text).replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$") + '"'


def _systemd_off(dreaming_dir: Path) -> bool:
    service, timer = _units(dreaming_dir)
    if not (service.exists() or timer.exists()):
        return False
    # The timer may never have started, so a failure to stop it is not one to report.
    subprocess.run(["systemctl", "--user", "disable", "--now", timer.name], capture_output=True)
    service.unlink(missing_ok=True)
    timer.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return True


def _systemd_on(dreaming_dir: Path, project: Path, program: Path, log: Path, hour: int, minute: int) -> None:
    service, timer = _units(dreaming_dir)
    service.parent.mkdir(parents=True, exist_ok=True)
    service.write_text(
        "[Unit]\n"
        f"Description=Dreaming over {_unit_text(str(project))}\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={_unit_text(str(project))}\n"
        "ExecStartPre=/bin/date\n"
        f"ExecStart={_unit_word(str(program))}\n"
        f"StandardOutput=append:{_unit_text(str(log))}\n"
        f"StandardError=append:{_unit_text(str(log))}\n",
        encoding="utf-8",
    )
    # Persistent: a run missed while the machine was off happens when it is next on.
    timer.write_text(
        "[Unit]\n"
        f"Description=Daily dreaming over {_unit_text(str(project))}\n\n"
        "[Timer]\n"
        f"OnCalendar=*-*-* {hour:02}:{minute:02}:00\n"
        "Persistent=true\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n",
        encoding="utf-8",
    )
    try:
        _call("systemctl", "--user", "daemon-reload")
        _call("systemctl", "--user", "enable", "--now", timer.name)
    except SystemExit:
        _systemd_off(dreaming_dir)  # leave no files that would read as a schedule that is on
        raise


def _systemd_status(dreaming_dir: Path) -> Schedule | None:
    service, timer = _units(dreaming_dir)
    if not (service.exists() and timer.exists()):
        return None
    time = re.search(r"^OnCalendar=\S+ (\d\d:\d\d):", timer.read_text(encoding="utf-8"), re.MULTILINE)
    project = re.search(r"^WorkingDirectory=(.*)$", service.read_text(encoding="utf-8"), re.MULTILINE)
    return Schedule(time[1], project[1].replace("%%", "%")) if time and project else None


def _linger_note() -> str:
    """systemd runs a user's timers only while that user is logged in, unless lingering is on for them."""
    user = getpass.getuser()
    shown = subprocess.run(
        ["loginctl", "show-user", user, "--property=Linger", "--value"], capture_output=True, text=True
    )
    if shown.stdout.strip() == "yes":
        return ""
    turned_on = subprocess.run(["loginctl", "--no-ask-password", "enable-linger", user], capture_output=True)
    if turned_on.returncode == 0:
        return f"Turned on lingering for {user}, so the run also happens while you are logged out.\n"
    return (
        "The run happens only while you are logged in. To have it happen while you are logged out too, "
        f"run: sudo loginctl enable-linger {user}\n"
    )


def status(dreaming_dir: Path) -> Schedule | None:
    return _cron_status(dreaming_dir) if _backend() == "cron" else _systemd_status(dreaming_dir)


def turn_on(dreaming_dir: Path, project: Path, log: Path, at: str | None) -> str:
    """Schedule a daily run over project, replacing this dreaming folder's schedule if it has one."""
    hour, minute = _clock(at or DEFAULT_TIME)
    program, backend = _program(), _backend()
    log.parent.mkdir(parents=True, exist_ok=True)
    note = ""
    if backend == "cron":
        _cron_on(dreaming_dir, project, program, log, hour, minute)
    else:
        _systemd_on(dreaming_dir, project, program, log, hour, minute)
        note = _linger_note()
    return (
        f"Schedule on: dreaming runs over {project} every day at {hour:02}:{minute:02}, this machine's time.\n"
        f"{note}"
        f"Each run adds what it prints to {log}"
    )


def turn_off(dreaming_dir: Path) -> str:
    removed = _cron_off(dreaming_dir) if _backend() == "cron" else _systemd_off(dreaming_dir)
    return "Schedule off." if removed else "Schedule off: there was none."


def describe(dreaming_dir: Path, log: Path) -> str:
    found = status(dreaming_dir)
    if not found:
        return "Schedule off."
    return (
        f"Schedule on: dreaming runs over {found.project} every day at {found.time}, this machine's time.\n"
        f"Log: {log}"
    )
