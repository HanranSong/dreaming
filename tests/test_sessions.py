"""Reading Claude Code's session logs."""

import os

from conftest import PROJECT, write_jsonl

from dreaming import formats, sessions
from dreaming.render import render_text

SUMMARY = "This session is being continued from a previous conversation. Summary: the build uses uv."


def user(uuid, content, *, parent=None, time="2026-11-01T10:00:00.000Z", cwd=PROJECT, **extra) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": time,
        "cwd": str(cwd),
        "message": {"role": "user", "content": content},
        **extra,
    }


def assistant(uuid, content, *, parent, time="2026-11-01T10:00:05.000Z") -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": time,
        "cwd": str(PROJECT),
        "message": {"role": "assistant", "content": content},
    }


def boundary(uuid, *, preserved_segment=None, preserved_uuids=()) -> dict:
    metadata = {"trigger": "auto", "preTokens": 150000}
    if preserved_segment:
        metadata["preservedSegment"] = dict(zip(("headUuid", "tailUuid"), preserved_segment, strict=True))
    if preserved_uuids:
        metadata["preservedMessages"] = {"uuids": list(preserved_uuids)}
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": uuid,
        "parentUuid": None,
        "logicalParentUuid": "never-written",
        "timestamp": "2026-11-01T11:00:00.000Z",
        "compactMetadata": metadata,
    }


def summary(uuid, *, parent) -> dict:
    return user(uuid, SUMMARY, parent=parent, time="2026-11-01T11:00:01.000Z", isCompactSummary=True)


def events_of(log) -> list[dict]:
    return sessions.load_session(log)["threads"][0]["events"]


def test_finds_the_logs_started_in_the_project_earliest_first(home):
    folder = home / ".claude" / "projects" / "-work-a-b"
    late = write_jsonl(folder / "late.jsonl", [user("u1", "second", cwd="/work/a-b", time="2026-11-02T08:00:00.000Z")])
    early = write_jsonl(folder / "early.jsonl", [user("u1", "first", cwd="/work/a-b", time="2026-11-01T08:00:00.000Z")])
    # /work/a/b gets the same folder name as /work/a-b, so its log sits beside theirs.
    write_jsonl(folder / "other.jsonl", [user("u1", "elsewhere", cwd="/work/a/b", time="2026-11-01T09:00:00.000Z")])
    write_jsonl(home / ".claude" / "projects" / "-work-c" / "c.jsonl", [user("u1", "another project", cwd="/work/c")])

    assert sessions.session_files(sessions.Path("/work/a-b")) == [early, late]
    assert sessions.session_id(early) == "early"


def test_a_log_becomes_the_events_of_its_conversation(tmp_path):
    log = write_jsonl(
        tmp_path / "abc.jsonl",
        [
            {"type": "ai-title", "aiTitle": "Run the tests"},
            user("u1", "run the tests"),
            assistant(
                "a1",
                [
                    {"type": "thinking", "thinking": "private reasoning"},
                    {"type": "text", "text": "Running."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pytest"}},
                ],
                parent="u1",
            ),
            user(
                "u2",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": False,
                        "content": [
                            {"type": "text", "text": "2 passed"},
                            {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                        ],
                    }
                ],
                parent="a1",
                time="2026-11-01T10:00:09.000Z",
            ),
            user("u3", "Base directory for this skill: /skills/x", parent="u2", isMeta=True),
            {"type": "custom-title", "customTitle": "Test run"},
        ],
        partial_line='{"type": "user", "message": {"content": "still being wri',
    )

    export = sessions.load_session(log)

    assert export["session"] == {
        "id": "abc",
        "title": "Test run",  # the name the user gave it, not the generated one
        "ai_title": "Run the tests",
        "created_at": "2026-11-01T10:00:00.000Z",
    }
    assert export["threads"] == [
        {
            "id": "primary",
            "parent_thread_id": None,
            "agent_name": "claude-code",
            "events": [
                {"type": "message.time", "utc": "2026-11-01T10:00:00Z"},
                {"type": "user.message", "content": "run the tests"},
                {"type": "agent.message", "content": "Running."},
                {"type": "agent.tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pytest"}},
                {
                    "type": "agent.tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "2 passed"}],
                    "is_error": False,
                },
            ],
        }
    ]
    # Everything in this log is something the reader knows, even what it leaves out.
    assert formats.notes() == []


def test_each_user_message_gets_its_time_once_per_second(tmp_path):
    log = write_jsonl(
        tmp_path / "abc.jsonl",
        [
            user("u1", "first", time="2026-11-01T10:00:00.100Z"),
            user("u2", "same second", parent="u1", time="2026-11-01T10:00:00.900Z"),
            user("u3", "two days later", parent="u2", time="2026-11-03T07:30:00.000Z"),
        ],
    )

    assert [event.get("utc") or event["content"] for event in events_of(log)] == [
        "2026-11-01T10:00:00Z",
        "first",
        "same second",
        "2026-11-03T07:30:00Z",
        "two days later",
    ]


def test_a_tool_output_saved_to_a_file_replaces_its_preview(tmp_path):
    def result(tool_id: str) -> dict:
        preview = (
            "<persisted-output>\nOutput too large (678.3KB). Full output saved to: "
            f"/home/someone/.claude/projects/-work-shop/abc/tool-results/{tool_id}.txt\n\nPreview (first 2KB):\nline 1"
        )
        return {"type": "tool_result", "tool_use_id": tool_id, "content": preview}

    log = write_jsonl(tmp_path / "abc.jsonl", [user("u1", [result("toolu_1"), result("toolu_2")])])
    saved = tmp_path / "abc" / "tool-results" / "toolu_1.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    kept, lost = events_of(log)

    assert kept["content"] == "line 1\nline 2\nline 3\n"
    assert lost["content"].startswith("<persisted-output>")  # its file is gone, so the preview is all there is


def test_a_complete_log_leaves_its_compaction_summaries_out(tmp_path):
    log = write_jsonl(
        tmp_path / "abc.jsonl",
        [
            user("u1", "set up the build"),
            assistant("a1", [{"type": "text", "text": "Done with uv."}], parent="u1"),
            boundary("b1"),
            summary("s1", parent="b1"),
            user("u2", "now add tests", parent="s1", time="2026-11-01T11:05:00.000Z"),
        ],
    )

    assert [event["type"] for event in events_of(log)] == [
        "message.time",
        "user.message",
        "agent.message",
        "message.time",
        "user.message",
    ]
    assert SUMMARY not in str(events_of(log))


def test_a_log_claude_code_deleted_messages_from_keeps_its_summaries(tmp_path):
    after = user("u2", "now add tests", parent="s1", time="2026-11-01T11:05:00.000Z")
    kept = [
        {"type": "compaction.summary", "utc": "2026-11-01T11:00:01Z"},
        {"type": "agent.message", "content": SUMMARY},
        {"type": "message.time", "utc": "2026-11-01T11:05:00Z"},
        {"type": "user.message", "content": "now add tests"},
    ]

    # Everything before the compaction was deleted.
    deleted = write_jsonl(tmp_path / "deleted.jsonl", [boundary("b1"), summary("s1", parent="b1"), after])
    assert events_of(deleted) == kept

    # The log was rewritten from part-way through the conversation.
    partway = write_jsonl(tmp_path / "partway.jsonl", [summary("s1", parent="not-in-this-log"), after])
    assert events_of(partway) == kept

    # The only messages left before the compaction are the ones it preserved, by a head-to-tail segment and by name.
    preserved = write_jsonl(
        tmp_path / "preserved.jsonl",
        [
            user("p1", "keep the port at 8080"),
            assistant("p2", [{"type": "text", "text": "Noted."}], parent="p1"),
            user("p3", "and the host", parent="p2", time="2026-11-01T10:00:00.500Z"),
            boundary("b1", preserved_segment=("p1", "p2"), preserved_uuids=["p3"]),
            summary("s1", parent="b1"),
            after,
        ],
    )
    assert [event for event in events_of(preserved) if event["type"] == "compaction.summary"] == kept[:1]

    # One message before the compaction that it did not preserve: nothing was deleted.
    complete = write_jsonl(
        tmp_path / "complete.jsonl",
        [
            user("p1", "keep the port at 8080"),
            assistant("p2", [{"type": "text", "text": "Noted."}], parent="p1"),
            boundary("b1", preserved_segment=("p2", "p2")),
            summary("s1", parent="b1"),
            after,
        ],
    )
    assert not [event for event in events_of(complete) if event["type"] == "compaction.summary"]


def test_subagents_are_threads_under_the_session(tmp_path):
    log = write_jsonl(tmp_path / "abc.jsonl", [user("u1", "look into the cache")])
    subagents = tmp_path / "abc" / "subagents"
    write_jsonl(subagents / "agent-late.jsonl", [user("u1", "read cache.py", time="2026-11-01T10:02:00.000Z")])
    (subagents / "agent-late.meta.json").write_text('{"agentType": "Explore", "description": "Read the cache"}')
    # One started by a workflow, with nothing to name it.
    write_jsonl(
        subagents / "workflows" / "wf_1" / "agent-early.jsonl",
        [user("u1", "list the files", time="2026-11-01T10:01:00.000Z")],
    )

    threads = sessions.load_session(log)["threads"]

    assert [(thread["id"], thread["parent_thread_id"], thread["agent_name"]) for thread in threads] == [
        ("primary", None, "claude-code"),
        ("early", "primary", "subagent"),
        ("late", "primary", "Read the cache"),
    ]
    assert threads[2]["events"][-1] == {"type": "user.message", "content": "read cache.py"}


def test_the_signature_changes_when_any_file_of_the_session_does(tmp_path):
    log = write_jsonl(tmp_path / "abc.jsonl", [user("u1", "look into the cache")])
    os.utime(log, (1_000_000, 1_000_000))
    signature, changed = sessions.log_stat(log)
    assert changed == 1_000_000
    assert sessions.log_stat(log) == (signature, changed)

    subagent = write_jsonl(tmp_path / "abc" / "subagents" / "agent-x.jsonl", [user("u1", "read cache.py")])
    with_subagent, changed = sessions.log_stat(log)
    assert with_subagent != signature
    assert changed == subagent.stat().st_mtime

    saved = tmp_path / "abc" / "tool-results" / "toolu_1.txt"
    saved.parent.mkdir()
    saved.write_text("output")
    assert sessions.log_stat(log)[0] not in (signature, with_subagent)


def test_the_transcript_reads_as_the_analyzer_is_told_it_does(tmp_path):
    call = {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}}
    result = {"type": "tool_result", "tool_use_id": "t1", "content": "2 passed", "is_error": True}
    log = write_jsonl(
        tmp_path / "abc.jsonl",
        [
            boundary("b1"),
            summary("s1", parent="b1"),
            user("u1", "run the tests", parent="s1", time="2026-11-01T11:05:00.000Z"),
            assistant("a1", [call], parent="u1"),
            user("u2", [result], parent="a1"),
        ],
    )
    export = sessions.load_session(log)

    text = render_text(export)

    # The label lines analyzer_system.md describes.
    for line in (
        'EVENT compaction.summary: {"utc":"2026-11-01T11:00:01Z"}',
        "AGENT:",
        'EVENT message.time: {"utc":"2026-11-01T11:05:00Z"}',
        "USER:",
        "TOOL CALL Bash:",
        "TOOL RESULT (Bash) [error]:",
    ):
        assert line in text.splitlines(), line
    assert "message.time" not in render_text(sessions.without_times(export))
