"""Reading Codex's session logs."""

import json
import os
from compression import zstd

from conftest import PROJECT, write_jsonl

from dreaming import codex, formats
from dreaming.render import render_text

DAY = "sessions/2026/11/01"


def meta(thread_id, *, time="2026-11-01T09:00:00.000Z", cwd=PROJECT, **extra) -> dict:
    payload = {"id": thread_id, "timestamp": time, "cwd": str(cwd), "source": "cli", **extra}
    return {"timestamp": time, "type": "session_meta", "payload": payload}


def item(payload: dict, *, time="2026-11-01T09:00:10.000Z") -> dict:
    return {"timestamp": time, "type": "response_item", "payload": payload}


def message(role, *texts, item_id=None, **extra) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    content = [{"type": kind, "text": text} for text in texts]
    return item({"type": "message", "id": item_id, "role": role, "content": content}, **extra)


def spawned_by(parent_id, agent_path) -> dict:
    """The source of a sub-agent the model started."""
    return {"subagent": {"thread_spawn": {"parent_thread_id": parent_id, "agent_path": agent_path}}}


def rollout(home, thread_id, records, *, folder=DAY, mtime=None, suffix=".jsonl"):
    log = write_jsonl(home / folder / f"rollout-2026-11-01T09-00-00-{thread_id}{suffix}", records)
    if mtime:
        os.utime(log, (mtime, mtime))
    return log


def test_finds_the_sessions_run_in_the_project_earliest_first(codex_home):
    # Codex files logs by date, not by project, and here the later session's log is the older file.
    late = rollout(codex_home, "late", [meta("late", time="2026-11-01T12:00:00.000Z")], mtime=1_000)
    early = rollout(codex_home, "early", [meta("early", time="2026-11-01T08:00:00.000Z")], mtime=2_000)
    rollout(codex_home, "elsewhere", [meta("elsewhere", cwd="/work/other")])
    rollout(codex_home, "sub", [meta("sub", source=spawned_by("early", "/root/reader"))])
    # Threads Codex runs for itself.
    rollout(codex_home, "review", [meta("review", source={"subagent": {"other": "guardian"}})])
    rollout(codex_home, "compact", [meta("compact", source={"subagent": "compact"})])
    rollout(codex_home, "memory", [meta("memory", thread_source="memory_consolidation")])
    # Not sessions: a first record that is no session_meta, an id that could not name a file, another file.
    rollout(codex_home, "headless", [message("user", "no meta above me")])
    rollout(codex_home, "odd", [meta("../odd")])
    write_jsonl(codex_home / DAY / "notes.jsonl", [meta("notes")])

    assert codex.session_files(PROJECT) == [early, late]
    assert codex.session_id(early) == "early"
    assert codex.started_at(late) == "2026-11-01T12:00:00.000Z"


def test_a_log_becomes_the_events_of_its_conversation(codex_home):
    log = rollout(
        codex_home,
        "abc",
        [
            meta("abc"),
            {"timestamp": "2026-11-01T09:00:01.000Z", "type": "turn_context", "payload": {"model": "gpt-6-luna"}},
            message("developer", "You are Codex."),
            message("user", "<environment_context>\n  <cwd>/work/shop</cwd>\n</environment_context>"),
            item(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "fix the build"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                        {"type": "input_text", "text": "<note>the lockfile is stale</note>"},  # a tag of the user's
                    ],
                }
            ),
            {"timestamp": "2026-11-01T09:00:10.000Z", "type": "event_msg", "payload": {"type": "user_message"}},
            item({"type": "reasoning", "encrypted_content": "gAAAA", "summary": []}),
            item({"type": "function_call", "name": "shell", "call_id": "c1", "arguments": '{"command": "uv lock"}'}),
            item({"type": "function_call_output", "call_id": "c1", "output": "Resolved 40 packages"}),
            item({"type": "custom_tool_call", "name": "exec", "call_id": "c2", "input": "text(1)"}),
            item({"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "1"}]}),
            item({"type": "web_search_call", "id": "ws1", "status": "completed", "action": {"query": "uv lock"}}),
            message("assistant", "The lockfile is current again."),
            # Kinds this reader has no name for are kept rather than dropped.
            item({"type": "plan_update", "id": "p1", "steps": ["lock", "test"]}),
            {"timestamp": "2026-11-01T09:00:20.000Z", "type": "new_record", "payload": {"note": "kept"}},
            message("user", "thanks", time="2026-11-01T09:05:00.000Z"),
        ],
    )
    write_jsonl(
        codex_home / "session_index.jsonl",
        [
            {"id": "abc", "thread_name": "Fix build"},
            {"id": "other", "thread_name": "Something else"},
            {"id": "abc", "thread_name": "Fix the stale lockfile"},
        ],
    )

    export = codex.load_session(log)

    assert export["session"] == {
        "id": "abc",
        "title": "Fix the stale lockfile",
        "created_at": "2026-11-01T09:00:00.000Z",
    }
    assert export["threads"] == [
        {
            "id": "primary",
            "parent_thread_id": None,
            "agent_name": "codex",
            "events": [
                {"type": "message.time", "utc": "2026-11-01T09:00:10Z"},
                {
                    "type": "user.message",
                    "content": [
                        {"type": "text", "text": "fix the build"},
                        {"type": "text", "text": "<note>the lockfile is stale</note>"},
                    ],
                },
                {"type": "agent.tool_use", "id": "c1", "name": "shell", "input": {"command": "uv lock"}},
                {"type": "agent.tool_result", "tool_use_id": "c1", "content": "Resolved 40 packages"},
                {"type": "agent.tool_use", "id": "c2", "name": "exec", "input": "text(1)"},
                {"type": "agent.tool_result", "tool_use_id": "c2", "content": [{"type": "text", "text": "1"}]},
                {"type": "agent.tool_use", "id": None, "name": "web_search", "input": {"action": {"query": "uv lock"}}},
                {"type": "agent.message", "content": [{"type": "text", "text": "The lockfile is current again."}]},
                {"type": "codex.plan_update", "steps": ["lock", "test"]},
                {"type": "codex.new_record", "payload": {"note": "kept"}},
                {"type": "message.time", "utc": "2026-11-01T09:05:00Z"},
                {"type": "user.message", "content": [{"type": "text", "text": "thanks"}]},
            ],
        }
    ]
    # What the reader kept without knowing it is reported, so that it does not pass unseen.
    assert formats.notes() == [
        f'Codex: a response item of type "plan_update" is not known and was kept as an EVENT line (in {log.name})',
        f'Codex: a record of type "new_record" is not known and was kept as an EVENT line (in {log.name})',
    ]


def test_sub_agents_are_threads_under_the_session_without_what_they_inherited(codex_home):
    task = {"type": "agent_message", "author": "/root", "recipient": "/root/reader"}
    report = {"type": "agent_message", "author": "/root/reader", "recipient": "/root"}
    parent = rollout(
        codex_home,
        "main",
        [
            meta("main"),
            message("user", "read the constants", item_id="m1"),
            item({**task, "id": "m2", "content": [{"type": "encrypted_content", "encrypted_content": "gAAAA"}]}),
        ],
    )
    child = rollout(
        codex_home,
        "child",
        [
            meta(
                "child",
                time="2026-11-01T09:01:00.000Z",
                forked_from_id="main",
                source=spawned_by("main", "/root/reader"),
            ),
            # A forked thread's log opens with a copy of the thread it was forked from.
            message("user", "read the constants", item_id="m1"),
            item({**task, "id": "m2", "content": [{"type": "encrypted_content", "encrypted_content": "gAAAA"}]}),
            item({**report, "id": "c1", "content": [{"type": "output_text", "text": "PORT is 8080"}]}),
        ],
    )
    rollout(
        codex_home,
        "grandchild",
        [
            meta("grandchild", time="2026-11-01T09:02:00.000Z", source=spawned_by("child", "/root/reader/checker")),
            message("assistant", "8080 confirmed"),
        ],
    )

    primary, reader, checker = codex.load_session(parent)["threads"]

    assert primary["events"][-1] == {
        "type": "agent.thread_message_sent",
        "to_agent_name": "/root/reader",
        "content": [{"type": "text", "text": "[encrypted by Codex]"}],
    }
    assert reader == {
        "id": "child",
        "parent_thread_id": "primary",
        "agent_name": "/root/reader",
        "events": [
            {
                "type": "agent.thread_message_sent",
                "to_agent_name": "/root",
                "content": [{"type": "text", "text": "PORT is 8080"}],
            }
        ],
    }
    assert (checker["id"], checker["agent_name"]) == ("grandchild", "/root/reader/checker")

    # A change to a sub-agent's log is a change to the session.
    signature = codex.log_stat(parent)[0]
    with child.open("a", encoding="utf-8") as log:
        log.write(json.dumps(message("assistant", "one more thing")) + "\n")
    assert codex.log_stat(parent)[0] != signature


def test_a_message_from_another_agent_is_marked_as_received(codex_home):
    log = rollout(
        codex_home,
        "child",
        [
            meta("child", source=spawned_by("main", "/root/reader")),
            item(
                {
                    "type": "agent_message",
                    "author": "/root",
                    "recipient": "/root/reader",
                    "content": [{"type": "input_text", "text": "read constants.py"}],
                }
            ),
        ],
    )

    assert codex._log_events(log) == [
        {
            "type": "agent.thread_message_received",
            "from_agent_name": "/root",
            "content": [{"type": "text", "text": "read constants.py"}],
        }
    ]


def test_a_compressed_log_is_read(codex_home):
    log = codex_home / DAY / "rollout-2026-11-01T09-00-00-abc.jsonl.zst"
    log.parent.mkdir(parents=True)
    with zstd.open(log, "wt", encoding="utf-8") as lines:
        for record in (meta("abc"), message("user", "fix the build")):
            lines.write(json.dumps(record) + "\n")

    assert codex.session_files(PROJECT) == [log]
    assert codex.load_session(log)["threads"][0]["events"][-1] == {
        "type": "user.message",
        "content": [{"type": "text", "text": "fix the build"}],
    }


def test_a_thread_with_two_logs_is_read_from_the_newer(codex_home):
    rollout(codex_home, "abc", [meta("abc"), message("user", "first half")], mtime=1_000)
    archived = rollout(
        codex_home,
        "abc",
        [meta("abc"), message("user", "first half"), message("assistant", "second half")],
        folder="archived_sessions",
        mtime=2_000,
    )

    assert codex.session_files(PROJECT) == [archived]


def test_a_log_that_cannot_be_read_does_not_hide_the_others(codex_home, capsys):
    good = rollout(codex_home, "good", [meta("good")])
    broken = codex_home / DAY / "rollout-2026-11-01T09-00-00-broken.jsonl.zst"
    broken.write_bytes(b"this is not zstd")

    assert codex.session_files(PROJECT) == [good]
    assert f"Left out a Codex log that could not be read: {broken}" in capsys.readouterr().err


def test_the_transcript_reads_as_the_analyzer_is_told_it_does(codex_home):
    log = rollout(
        codex_home,
        "abc",
        [
            meta("abc"),
            message("user", "fix the build"),
            item({"type": "function_call", "name": "shell", "call_id": "c1", "arguments": '{"command": "uv lock"}'}),
            item({"type": "function_call_output", "call_id": "c1", "output": "Resolved 40 packages"}),
            message("assistant", "Done."),
        ],
    )

    lines = render_text(codex.load_session(log)).splitlines()

    for line in (
        "=== primary thread: codex ===",
        'EVENT message.time: {"utc":"2026-11-01T09:00:10Z"}',
        "USER:",
        "fix the build",
        "TOOL CALL shell:",
        "TOOL RESULT (shell):",
        "Resolved 40 packages",
        "AGENT:",
    ):
        assert line in lines, line
