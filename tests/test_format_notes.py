"""What a run says when a log holds something the readers do not know."""

from conftest import PROJECT, write_jsonl
from test_codex import DAY, item, message, meta, rollout
from test_sessions import assistant, user

import dreaming
from dreaming import codex, formats, sessions
from dreaming.report import write_report


def claude_notes(tmp_path, records, **written) -> list[str]:
    sessions.load_session(write_jsonl(tmp_path / "abc.jsonl", records, **written))
    return formats.notes()


def codex_notes(codex_home, records) -> list[str]:
    codex.load_session(rollout(codex_home, "abc", [meta("abc"), *records]))
    return formats.notes()


def test_claude_code_a_content_block_of_an_unknown_type(tmp_path):
    blocks = [{"type": "text", "text": "Searching."}, {"type": "server_tool_use", "id": "s1", "name": "web_search"}]

    assert claude_notes(tmp_path, [user("u1", "look it up"), assistant("a1", blocks, parent="u1")]) == [
        'Claude Code: a content block of type "server_tool_use" was left out of the transcript (in abc.jsonl)'
    ]


def test_claude_code_a_message_in_a_record_of_an_unknown_type(tmp_path):
    moved = {"type": "turn", "uuid": "t1", "message": {"role": "assistant", "content": "The port is 8080."}}
    # Claude Code's copy of a request it sent holds a message too, and is no part of the conversation.
    request = {"type": "api-request-blob", "hash": "h1", "message": {"role": "user", "content": "what port?"}}

    assert claude_notes(tmp_path, [user("u1", "what port?"), moved, request]) == [
        'Claude Code: a record of type "turn" holds a message and was left out (in abc.jsonl)'
    ]


def test_claude_code_a_broken_line_that_is_not_the_last(tmp_path):
    log = tmp_path / "abc.jsonl"
    whole = write_jsonl(tmp_path / "whole.jsonl", [user("u1", "first"), user("u2", "second", parent="u1")]).read_text()
    first, second = whole.splitlines(keepends=True)

    log.write_text(first + second[:40])  # the last line is still being written
    sessions.load_session(log)
    assert formats.notes() == []

    log.write_text(first[:40] + "\n" + second)
    sessions.load_session(log)
    assert formats.notes() == ["Claude Code: a line that is not JSON was skipped (in abc.jsonl)"]


def test_claude_code_a_saved_tool_output_named_another_way(tmp_path):
    preview = "<persisted-output>\nOutput too large. Stored as attachment toolu_1.\n\nPreview:\nline 1"
    result = {"type": "tool_result", "tool_use_id": "toolu_1", "content": preview}

    assert claude_notes(tmp_path, [user("u1", [result])]) == [
        "Claude Code: a tool output saved to a file does not name its file, so only its preview was read (in abc.jsonl)"
    ]


def test_claude_code_a_compaction_summary_not_marked_as_one(tmp_path):
    text = "This session is being continued from a previous conversation that ran out of context."

    assert claude_notes(tmp_path, [user("u1", "set up the build"), user("u2", text, parent="u1")]) == [
        "Claude Code: a compaction summary is not marked as one, so it reads as a user message (in abc.jsonl)"
    ]


def test_claude_code_a_message_with_no_timestamp(tmp_path):
    untimed = {key: value for key, value in user("u1", "set up the build").items() if key != "timestamp"}

    assert claude_notes(tmp_path, [untimed]) == [
        "Claude Code: a message has no timestamp, so the transcript does not say when it was sent (in abc.jsonl)"
    ]


def test_claude_code_sessions_kept_under_another_folder_name(home):
    projects = home / ".claude" / "projects"
    write_jsonl(projects / "work-shop-1f3a" / "abc.jsonl", [user("u1", "set up the build")])
    write_jsonl(projects / "-work-other" / "def.jsonl", [user("u1", "another project", cwd="/work/other")])

    assert sessions.session_files(PROJECT) == []
    assert formats.notes() == [
        "Claude Code: sessions started in this folder are kept under projects/work-shop-1f3a, not "
        "projects/-work-shop, and were not read (in abc.jsonl)"
    ]

    # Sessions where they are expected are simply read.
    formats.clear()
    expected = write_jsonl(projects / "-work-shop" / "ghi.jsonl", [user("u1", "add tests")])
    assert sessions.session_files(PROJECT) == [expected]
    assert formats.notes() == []


def test_codex_records_and_items_of_unknown_types(codex_home):
    records = [
        message("user", "fix the build"),
        item({"type": "plan_update", "steps": ["lock", "test"]}),
        {"timestamp": "2026-11-01T09:00:20.000Z", "type": "new_record", "payload": {"note": "kept"}},
    ]

    assert codex_notes(codex_home, records) == [
        'Codex: a response item of type "plan_update" is not known and was kept as an EVENT line '
        "(in rollout-2026-11-01T09-00-00-abc.jsonl)",
        'Codex: a record of type "new_record" is not known and was kept as an EVENT line '
        "(in rollout-2026-11-01T09-00-00-abc.jsonl)",
    ]


def test_codex_a_user_message_wholly_inside_an_unknown_tag(codex_home):
    records = [
        message("user", "<environment_context>\n  <cwd>/work/shop</cwd>\n</environment_context>"),
        message("user", "<session_recap>\nEarlier you fixed the build.\n</session_recap>"),
        # Not named as Codex names its own tags: something the user pasted.
        message("user", "<config>\n  <port>8080</port>\n</config>"),
        message("user", "wrap it in <b>bold</b> please"),
    ]

    assert codex_notes(codex_home, records) == [
        "Codex: a user message wholly inside a <session_recap> tag was read as the user's; Codex may have added it "
        "(in rollout-2026-11-01T09-00-00-abc.jsonl)"
    ]


def test_codex_logs_that_cannot_be_matched_to_a_session(codex_home):
    rollout(codex_home, "headless", [message("user", "no session_meta above me")], mtime=1_000)
    rollout(codex_home, "nowhere", [meta("nowhere", cwd="")], mtime=2_000)
    write_jsonl(codex_home / DAY / "rollout-2026-11-01T09-00-00-empty.jsonl", [])  # nothing written to it yet

    assert codex.session_files(PROJECT) == []
    assert formats.notes() == [
        "Codex: a log does not open with a session_meta naming its thread, and was left out "
        "(in rollout-2026-11-01T09-00-00-headless.jsonl)",
        "Codex: a session's session_meta names no folder (cwd), so it belongs to no project "
        "(in rollout-2026-11-01T09-00-00-nowhere.jsonl)",
    ]


def test_codex_an_index_of_sessions_with_no_log_to_be_found(codex_home):
    write_jsonl(codex_home / "session_index.jsonl", [{"id": "abc", "thread_name": "Fix build"}])
    write_jsonl(codex_home / "threads" / "2026" / "abc.jsonl", [meta("abc")])  # where this version does not look

    assert codex.session_files(PROJECT) == []
    assert formats.notes() == [
        "Codex: its index lists sessions, but no log was found under sessions/ or archived_sessions/ "
        "(in session_index.jsonl)"
    ]


def test_a_note_names_the_first_logs_it_was_met_in(tmp_path):
    for name in ("a", "b", "c", "d", "e"):
        formats.note("Codex", "something new", tmp_path / f"{name}.jsonl")
    formats.note("Codex", "something new", tmp_path / "a.jsonl")
    formats.note("Codex", "something else", tmp_path / "a.jsonl")

    assert formats.notes() == [
        "Codex: something new (in a.jsonl, b.jsonl, c.jsonl and 2 more)",
        "Codex: something else (in a.jsonl)",
    ]


def test_a_report_holds_what_a_run_met_and_carried_on_past(tmp_path):
    record = {
        "id": "20261101-010000",
        "project": str(PROJECT),
        "models": {"analyzer": ("claude-haiku-5-5", "medium"), "organizer": ("claude-haiku-5-5", "medium")},
        "compaction": {"percent": 85, "analyzer": None, "organizer": None},
        "started": "2026-11-01T01:00:00+00:00",
        "finished": "2026-11-01T01:00:05+00:00",
        "sessions": [],
        "not_run": [],
        "organizer": {"status": "not needed"},
        "agents": [],
        "memory_changes": {},
    }
    report = tmp_path / "report.md"

    assert dreaming._format_notice(stopped=False) == []
    write_report(report, {**record, "format_notice": []})
    assert "Log format" not in report.read_text()

    formats.note("Codex", "something new", tmp_path / "a.jsonl")
    notice = dreaming._format_notice(stopped=False)
    assert notice == [formats.NOTICE, "- Codex: something new (in a.jsonl)", formats.CARRIED_ON]
    assert dreaming._format_notice(stopped=True)[-1] == formats.STOPPED
    write_report(report, {**record, "format_notice": notice})
    assert "\n## Log format\n\n" + "\n".join(notice) + "\n\n## Sessions\n" in report.read_text()
