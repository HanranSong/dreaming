"""The sessions of both tools as one timeline."""

from conftest import PROJECT, write_jsonl
from test_codex import meta, rollout
from test_sessions import user

import dreaming


def test_the_sessions_of_both_tools_come_in_the_order_they_started(home, codex_home):
    folder = home / ".claude" / "projects" / "-work-shop"
    write_jsonl(folder / "morning.jsonl", [user("u1", "first", time="2026-11-01T08:00:00.000Z")])
    write_jsonl(folder / "evening.jsonl", [user("u1", "third", time="2026-11-01T20:00:00.000Z")])
    rollout(codex_home, "noon", [meta("noon", time="2026-11-01T12:00:00.000Z")])

    found = dreaming._project_sessions(PROJECT)

    assert [(session.id, session.reader.SOURCE) for session in found] == [
        ("morning", "claude-code"),
        ("noon", "codex"),
        ("evening", "claude-code"),
    ]
