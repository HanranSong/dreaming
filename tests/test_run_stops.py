"""A run stops before it analyzes anything when a session log holds something the readers do not know."""

import json
import os
import re
import time
from collections import Counter

import pytest
from conftest import write_jsonl
from test_sessions import assistant, user

import dreaming
from dreaming import formats
from dreaming.agent import UPLOADS_PATH, AgentRun

UNKNOWN = 'Claude Code: a content block of type "server_tool_use" was left out of the transcript (in odd.jsonl)'


@pytest.fixture
def project(tmp_path, home, codex_home, monkeypatch):
    """A project folder to run dreaming from, with its own state and memory and no model behind it."""
    folder = tmp_path / "shop"
    folder.mkdir()
    monkeypatch.chdir(folder)
    state = tmp_path / "dreaming" / ".state"
    for name, path in {
        "MEMORY_DIR": tmp_path / "dreaming" / "memory",
        "STATE_DIR": state,
        "TRANSCRIPT_DIR": state / "transcripts",
        "RUNS_DIR": state / "runs",
        "STATE_FILE": state / "sessions.json",
        "LOCK_FILE": state / "lock",
        "RESTORE_FILE": state / "restore.json",
        "LEGACY_PROCESSED_FILE": state / "processed.json",
    }.items():
        monkeypatch.setattr(dreaming, name, path)
    dreaming.TRANSCRIPT_DIR.mkdir(parents=True)
    dreaming.MEMORY_DIR.mkdir()
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "not-a-key")
    return folder


@pytest.fixture
def agents(monkeypatch) -> list[str]:
    """Stands in for the model: every agent succeeds at once. Returns the labels of the agents that ran."""
    ran = []

    def run_agent(model, effort, system, task, transcripts, memory, run_dir, label, *rest):
        ran.append(label.split()[0])
        read = {f"{UPLOADS_PATH}/{transcript.name}" for transcript in transcripts.iterdir()}
        return AgentRun(text="done", usage=Counter(requests=1), cost=0.0, read=read)

    monkeypatch.setattr(dreaming, "run_agent", run_agent)
    return ran


def session(project, name, records):
    """A session started in the project, finished long enough ago to be analyzed."""
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project))
    log = write_jsonl(project.parent / ".claude" / "projects" / slug / f"{name}.jsonl", records)
    hour_ago = time.time() - 3600
    os.utime(log, (hour_ago, hour_ago))


def known(project, name="plain"):
    session(project, name, [user("u1", "run the tests", cwd=project)])


def unknown(project, name="odd"):
    blocks = [{"type": "text", "text": "Searching."}, {"type": "server_tool_use", "id": "s1", "name": "web_search"}]
    session(project, name, [user("u1", "look it up", cwd=project), assistant("a1", blocks, parent="u1")])


def statuses() -> dict[str, str]:
    saved = json.loads(dreaming.STATE_FILE.read_text())
    return {session_id: entry["status"] for session_id, entry in saved["sessions"].items()}


def test_a_log_with_something_unknown_stops_the_run_before_any_session_is_analyzed(project, agents):
    known(project)
    unknown(project)

    with pytest.raises(SystemExit) as stopped:
        dreaming._run()

    assert str(stopped.value) == "\n".join([formats.NOTICE, f"- {UNKNOWN}", formats.STOPPED])
    # Not the session whose log is fine either: nothing was spent, written or recorded.
    assert agents == []
    assert not dreaming.RUNS_DIR.exists()
    assert list(dreaming.MEMORY_DIR.iterdir()) == []
    assert statuses() == {}


def test_told_to_carry_on_the_run_analyzes_the_sessions_and_reports_what_it_met(project, agents, capsys):
    known(project)
    unknown(project)

    dreaming._run(ignore_format=True)

    assert agents == ["analyzer", "analyzer", "organizer"]
    assert statuses() == {"plain": "done", "odd": "done"}
    notice = "\n".join([formats.NOTICE, f"- {UNKNOWN}", formats.CARRIED_ON])
    assert notice in capsys.readouterr().out
    (run,) = dreaming.RUNS_DIR.iterdir()
    assert f"## Log format\n\n{notice}\n" in (run / "report.md").read_text()


def test_logs_the_readers_know_are_analyzed_without_a_word_about_their_format(project, agents, capsys):
    known(project)

    dreaming._run()

    assert agents == ["analyzer", "organizer"]
    assert "Log format" not in capsys.readouterr().out
    (run,) = dreaming.RUNS_DIR.iterdir()
    assert "Log format" not in (run / "report.md").read_text()


def test_sessions_that_cannot_be_found_where_expected_stop_the_run_too(project, agents):
    elsewhere = project.parent / ".claude" / "projects" / "renamed"
    log = write_jsonl(elsewhere / "abc.jsonl", [user("u1", "hi", cwd=project)])

    with pytest.raises(SystemExit) as stopped:
        dreaming._run()

    assert "are kept under projects/renamed" in str(stopped.value)
    assert formats.STOPPED in str(stopped.value)
    assert agents == [] and log.exists()


def test_a_log_that_turns_unknown_during_the_run_is_left_for_the_next_run(project, agents, monkeypatch):
    known(project)
    read = dreaming.claude_code.load_session

    def read_after_it_changed(path):
        formats.note("Claude Code", "something new", path)
        return read(path)

    # A session is read once when the run starts, and again when it is analyzed.
    reads = iter([read, read_after_it_changed])
    monkeypatch.setattr(dreaming.claude_code, "load_session", lambda path: next(reads)(path))

    with pytest.raises(SystemExit) as failed:
        dreaming._run()

    assert failed.value.code == 1
    assert agents == []
    assert statuses() == {"plain": "failed"}
